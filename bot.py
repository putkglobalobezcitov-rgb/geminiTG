import asyncio
import io
import logging
import os
import sys

from aiohttp import web
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from config import BOT_TOKEN, PORT
from gemini import ask_gemini, clear_user_history, get_user_mode, set_user_mode

# Настройка UTF-8 для Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
dp = Dispatcher()


# Веб-сервер для бесплатного тарифа Render Web Service
async def health_handler(request):
    return web.Response(text="Gemini School Helper Bot is LIVE!", status=200)


async def start_health_server():
    app = web.Application()
    app.router.add_get("/", health_handler)
    app.router.add_get("/healthz", health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    logger.info(f"Health-check сервер запущен на порту {PORT}")


async def send_smart_message(message: Message, text: str):
    """Отправляет сообщение частями (до 4000 символов) с безопасным парсингом Markdown."""
    chunk_size = 3900
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        try:
            await message.answer(chunk, parse_mode=ParseMode.MARKDOWN)
        except TelegramBadRequest:
            # Если в тексте формул сломался Markdown, отправляем как обычный текст
            await message.answer(chunk, parse_mode=None)


@dp.message(CommandStart())
async def cmd_start(message: Message):
    welcome_text = (
        "Привет. Я помогу с учебой и подготовкой к контрольным.\n\n"
        "• Обычный режим (по умолчанию): отправь фото задания или напиши вопрос — я сразу решу и объясню.\n"
        "• Режим подготовки (/quiz или /history): отправь тему или фото учебника — сделаю краткий пересказ сути и дам тест для самопроверки.\n"
        "• /new — сбросить контекст и вернуться к обычному решению."
    )
    await message.answer(welcome_text, parse_mode=None)


@dp.message(Command("quiz"))
@dp.message(Command("history"))
@dp.message(Command("test"))
@dp.message(Command("prep"))
async def cmd_quiz(message: Message):
    set_user_mode(message.from_user.id, "quiz")
    
    # Проверяем, передал ли пользователь тему прямо в команде (например: /history Северная война)
    args = message.text.split(maxsplit=1)
    if len(args) > 1 and args[1].strip():
        topic = args[1].strip()
        await bot.send_chat_action(message.chat.id, ChatAction.TYPING)
        response = await ask_gemini(user_id=message.from_user.id, prompt=topic, mode="quiz")
        await send_smart_message(message, response)
    else:
        text = (
            "Включен режим подготовки к проверочной.\n"
            "Отправь фото страницы учебника/конспекта или напиши тему текстом.\n"
            "Я сделаю краткий пересказ сути и составлю тест для самопроверки.\n"
            "Чтобы вернуться к обычному решению задач, напиши /new"
        )
        await message.answer(text, parse_mode=None)


@dp.message(Command("new"))
@dp.message(Command("clear"))
@dp.message(Command("solve"))
async def cmd_clear(message: Message):
    set_user_mode(message.from_user.id, "solve")
    clear_user_history(message.from_user.id)
    await message.answer("Диалог очищен. Включен обычный режим решения. Отправь фото или вопрос.", parse_mode=None)


@dp.message(F.photo)
async def handle_photo(message: Message):
    """Обработчик фотографий (задач, тестов, конспектов)."""
    await bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    
    # Берем фото максимального разрешения
    photo = message.photo[-1]
    
    # Скачиваем фото в память
    file_io = io.BytesIO()
    await bot.download(photo, destination=file_io)
    image_bytes = file_io.getvalue()
    
    # Подпись к фото (если пользователь что-то написал к фотке)
    prompt = message.caption

    # Запрашиваем ответ у Gemini
    response = await ask_gemini(
        user_id=message.from_user.id,
        prompt=prompt,
        image_bytes=image_bytes,
        image_mime="image/jpeg"
    )

    await send_smart_message(message, response)


@dp.message(F.document)
async def handle_document(message: Message):
    """Обработчик документов (если фото отправлено файлом)."""
    mime = message.document.mime_type or ""
    if mime.startswith("image/"):
        await bot.send_chat_action(message.chat.id, ChatAction.TYPING)
        file_io = io.BytesIO()
        await bot.download(message.document, destination=file_io)
        image_bytes = file_io.getvalue()

        response = await ask_gemini(
            user_id=message.from_user.id,
            prompt=message.caption,
            image_bytes=image_bytes,
            image_mime=mime
        )
        await send_smart_message(message, response)
    else:
        await message.answer("⚠️ Пожалуйста, отправьте изображение (JPEG, PNG).")


@dp.message(F.text)
async def handle_text(message: Message):
    """Обработчик обычных текстовых сообщений."""
    await bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    
    response = await ask_gemini(
        user_id=message.from_user.id,
        prompt=message.text
    )

    await send_smart_message(message, response)


async def main():
    logger.info("Запуск Gemini School Helper бота...")
    try:
        me = await bot.get_me()
        logger.info(f"Бот успешно авторизован: @{me.username} ({me.first_name})")
        print(f"\n>>> Бот @{me.username} ГОТОВ К РАБОТЕ! <<<")
        
        # Запускаем веб-сервер для Render
        await start_health_server()

        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
