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

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=45)) as resp:
                if resp.status != 200:
                    err_body = await resp.text()
                    logger.error(f"Gemini API Error {resp.status}: {err_body}")
                    return f"❌ Ошибка нейросети (код {resp.status}). Попробуйте еще раз через пару секунд."

                result = await resp.json()
                candidates = result.get("candidates", [])
                if not candidates:
                    return "⚠️ Нейросеть не смогла сгенерировать ответ. Попробуйте сфотографировать четче."

                parts_response = candidates[0].get("content", {}).get("parts", [])
                # Извлекаем текст ответа (игнорируя технические поля)
                text_response = "".join(p.get("text", "") for p in parts_response if "text" in p)

                if not text_response:
                    return "⚠️ Пустой ответ от модели. Попробуйте переформулировать."

                # Сохраняем текстовый след в историю диалога
                # (картинки в историю не сохраняем, чтобы не перегружать токены в последующих запросах)
                user_histories[user_id].append({"role": "user", "parts": [{"text": text_prompt}]})
                user_histories[user_id].append({"role": "model", "parts": [{"text": text_response}]})

                # Ограничиваем размер истории
                if len(user_histories[user_id]) > MAX_HISTORY_LEN * 2:
                    user_histories[user_id] = user_histories[user_id][-MAX_HISTORY_LEN * 2:]

                return text_response

    except aiohttp.ClientConnectorError as cce:
        logger.error(f"Ошибка подключения к Gemini: {cce}")
        return "❌ Ошибка связи с сервером нейросети. Проверьте интернет."
    except Exception as e:
        logger.error(f"Исключение при обращении к Gemini: {e}", exc_info=True)
        return f"❌ Произошла ошибка: {e}"
