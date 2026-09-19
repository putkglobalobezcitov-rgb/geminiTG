import base64
import logging
from typing import Dict, List, Optional

import aiohttp
from config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "Ты — умнейший академический ИИ-ассистент для помощи в учебе, тестах и домашних заданиях. "
    "Ты обладаешь зрением высшего уровня и безошибочно распознаешь любой текст с фотографий: "
    "рукописный почерк, формулы, печатные варианты, скриншоты, доску и учебники.\n\n"
    "ПРАВИЛА ОТВЕТА:\n"
    "1. Будь максимально точным, не фантазируй и перепроверяй расчеты.\n"
    "2. Сначала пиши ЧЕТКИЙ ИТОГОВЫЙ ОТВЕТ (например: '🎯 Ответ: 25' или '🎯 Правильный вариант: Б').\n"
    "3. Затем давай понятное, компактное пошаговое решение.\n"
    "4. Если на фото несколько номеров и нет конкретного вопроса, реши задания по порядку (Номер 1, Номер 2...).\n"
    "5. Используй структурированное форматирование: списки, жирный шрифт и понятные обозначения."
)

# Хранилище истории диалогов пользователей: user_id -> list of contents
user_histories: Dict[int, List[dict]] = {}
MAX_HISTORY_LEN = 12


def clear_user_history(user_id: int):
    """Сбрасывает контекст диалога для пользователя."""
    if user_id in user_histories:
        del user_histories[user_id]


async def ask_gemini(
    user_id: int,
    prompt: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg"
) -> str:
    """Отправляет запрос в Google Gemini с учетом контекста и фото."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    if user_id not in user_histories:
        user_histories[user_id] = []

    parts = []

    # Если передано изображение, добавляем его в формате inline_data
    if image_bytes:
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        parts.append({
            "inline_data": {
                "mime_type": image_mime,
                "data": encoded_image
            }
        })

    # Текстовый запрос
    text_prompt = prompt.strip() if prompt else ("Реши задания на этой фотографии подробно и понятно." if image_bytes else "Привет!")
    parts.append({"text": text_prompt})

    # Текущее сообщение пользователя
    user_message = {"role": "user", "parts": parts}
    
    # Формируем цепочку с историей (для картинок историю берем умеренно, чтобы не раздувать запрос)
    current_contents = list(user_histories[user_id][-MAX_HISTORY_LEN:])
    current_contents.append(user_message)

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}]
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
