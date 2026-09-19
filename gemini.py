import asyncio
import base64
import logging
from typing import Dict, List, Optional, Tuple

import aiohttp
from config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

SOLVE_INSTRUCTION = (
    "Ты помогаешь школьнику (8 класс) быстро и четко списать и решить задания из тестов, ВПР, контрольных и домашки. "
    "Отвечай МАКСИМАЛЬНО КРАТКО, КОМПАКТНО, БЕЗ ВОДЫ, БЕЗ СМАЙЛИКОВ И БЕЗ ДЛИННЫХ СОЧИНЕНИЙ.\n\n"
    "ПРАВИЛА:\n"
    "1. Всегда пиши четкий список ответов по порядку номеров:\n"
    "   Задание 1: Ответ\n"
    "   Задание 2: Ответ\n"
    "   Задание 3: Ответ\n"
    "2. Краткое пояснение (если требуется по заданию) пиши МАКСИМУМ В 1-2 СТРОКИ прямо под номером задания, без лишней теории.\n"
    "3. Если прислано несколько фото/страниц, объедини все задания в один последовательный список (от меньшего номера к большему).\n"
    "4. СТРОГО ЗАПРЕЩЕНО писать знаки $, LaTeX-код (\\frac, \\sqrt) — формулы пиши обычным текстом: (a - 2)² / 60.\n"
    "5. Строго без эмодзи и без приветствий."
)

QUIZ_INSTRUCTION = (
    "Ты готовишь школьника к проверочной или контрольной (8 класс) по теме или фото учебника. "
    "Строго БЕЗ смайликов, БЕЗ знаков $, БЕЗ LaTeX, простым и понятным языком.\n\n"
    "СТРУКТУРА ОТВЕТА:\n"
    "1. КРАТКИЙ ПЕРЕСКАЗ (самая суть темы в 4-6 ключевых пунктах: главные факты, даты, формулы без воды).\n"
    "2. ТЕСТ ДЛЯ САМОПРОВЕРКИ (4-5 тестовых вопросов с вариантами А, Б, В, Г).\n"
    "3. ПРАВИЛЬНЫЕ ОТВЕТЫ (в самом низу для самопроверки)."
)

# Хранилище истории диалогов и режимов: user_id -> dict
user_histories: Dict[int, List[dict]] = {}
user_modes: Dict[int, str] = {}  # "solve" | "quiz"
MAX_HISTORY_LEN = 12


def set_user_mode(user_id: int, mode: str):
    """Устанавливает режим работы для пользователя ('solve' или 'quiz')."""
    user_modes[user_id] = mode
    clear_user_history(user_id)


def get_user_mode(user_id: int) -> str:
    """Возвращает текущий режим пользователя."""
    return user_modes.get(user_id, "solve")


def clear_user_history(user_id: int):
    """Сбрасывает контекст диалога для пользователя."""
    if user_id in user_histories:
        del user_histories[user_id]


async def ask_gemini(
    user_id: int,
    prompt: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
    images: Optional[List[Tuple[bytes, str]]] = None,
    mode: Optional[str] = None
) -> str:
    """Отправляет запрос в Google Gemini с поддержкой одного или нескольких фото."""
    current_mode = mode or get_user_mode(user_id)
    system_instruction = QUIZ_INSTRUCTION if current_mode == "quiz" else SOLVE_INSTRUCTION
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    if user_id not in user_histories:
        user_histories[user_id] = []

    parts = []

    # Добавляем изображения в запрос
    all_images = list(images) if images else []
    if image_bytes:
        all_images.append((image_bytes, image_mime))

    for img_b, img_m in all_images:
        encoded_image = base64.b64encode(img_b).decode("utf-8")
        parts.append({
            "inline_data": {
                "mime_type": img_m,
                "data": encoded_image
            }
        })

    # Текстовый запрос
    if prompt and prompt.strip():
        text_prompt = prompt.strip()
    elif all_images:
        text_prompt = "Составь краткий пересказ сути и тест с вопросами по этому материалу." if current_mode == "quiz" else "Реши все задания на этих фотографиях по порядку номеров кратко и четко."
    else:
        text_prompt = "Привет!"

    parts.append({"text": text_prompt})

    # Текущее сообщение пользователя
    user_message = {"role": "user", "parts": parts}
    
    # Формируем цепочку с историей (для картинок историю берем умеренно, чтобы не раздувать запрос)
    current_contents = list(user_histories[user_id][-MAX_HISTORY_LEN:])
    current_contents.append(user_message)

    payload = {
        "system_instruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": current_contents,
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 4096
        }
    }

    # Список моделей по приоритету: если основная перегружена (503, 429) или недоступна, переключаемся на следующую
    candidate_models = [
        GEMINI_MODEL,
        "gemini-3.6-flash",
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.5-pro"
    ]
    models_to_try = []
    for m in candidate_models:
        if m and m not in models_to_try:
            models_to_try.append(m)

    last_error_code = None

    try:
        async with aiohttp.ClientSession() as session:
            for model_name in models_to_try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"

                for attempt in range(2):
                    try:
                        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=35)) as resp:
                            if resp.status == 200:
                                result = await resp.json()
                                candidates = result.get("candidates", [])
                                if not candidates:
                                    continue

                                parts_response = candidates[0].get("content", {}).get("parts", [])
                                text_response = "".join(p.get("text", "") for p in parts_response if "text" in p)

                                if not text_response:
                                    continue

                                # Сохраняем текстовый след в историю
                                user_histories[user_id].append({"role": "user", "parts": [{"text": text_prompt}]})
                                user_histories[user_id].append({"role": "model", "parts": [{"text": text_response}]})

                                if len(user_histories[user_id]) > MAX_HISTORY_LEN * 2:
                                    user_histories[user_id] = user_histories[user_id][-MAX_HISTORY_LEN * 2:]

                                return text_response

                            last_error_code = resp.status
                            if resp.status in (500, 503, 429):
                                logger.warning(f"Модель {model_name} вернула код {resp.status} (перегрузка), попытка {attempt + 1}")
                                await asyncio.sleep(1)
                            else:
                                err_body = await resp.text()
                                logger.error(f"Gemini API {model_name} Error {resp.status}: {err_body}")
                                break
                    except asyncio.TimeoutError:
                        logger.warning(f"Таймаут запроса к {model_name}, пробуем следующую модель...")
                        break

            return f"❌ Серверы нейросети временно перегружены (код {last_error_code or 503}). Повторите запрос через 5 секунд."

    except aiohttp.ClientConnectorError as cce:
        logger.error(f"Ошибка подключения к Gemini: {cce}")
        return "❌ Ошибка связи с сервером нейросети. Проверьте интернет."
    except Exception as e:
        logger.error(f"Исключение при обращении к Gemini: {e}", exc_info=True)
        return f"❌ Произошла ошибка: {e}"
