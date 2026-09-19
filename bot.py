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
from aiogram.types import (
    BotCommand,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)

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

# Удобная клавиатура для телефона
KEYBOARD = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text="/a Решить задачу"),
            KeyboardButton(text="/b Тест и пересказ"),
        ],
        [
            KeyboardButton(text="/new Очистить"),
        ]
    ],
    resize_keyboard=True
)


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
        "Привет. Выбирай нужный режим кнопками внизу или командами:\n\n"
        "• /a — Решить задачу (скинь фото или текст — дам четкий ответ и решение)\n"
        "• /b — Тест и пересказ (скинь фото учебника или тему — сделаю пересказ и тест для самопроверки)\n"
        "• /new — Сбросить диалог"
    )
    await message.answer(welcome_text, parse_mode=None, reply_markup=KEYBOARD)


# Режим подготовки (пересказ + тест): команды /b, /quiz, /history, /test, /prep или нажатие кнопки
@dp.message(Command("b"))
@dp.message(Command("quiz"))
@dp.message(Command("history"))
@dp.message(Command("test"))
@dp.message(Command("prep"))
@dp.message(F.text.startswith("/b"))
@dp.message(F.text == "/b Тест и пересказ")
async def cmd_quiz(message: Message):
    set_user_mode(message.from_user.id, "quiz")
    
    # Извлекаем тему, если пользователь ввел сразу с текстом (например: /b Северная война)
    raw_text = message.text or ""
    parts = raw_text.split(maxsplit=1)
    
    if len(parts) > 1 and parts[1].strip() and not parts[1].startswith("Тест и пересказ"):
        topic = parts[1].strip()
        await bot.send_chat_action(message.chat.id, ChatAction.TYPING)
        response = await ask_gemini(user_id=message.from_user.id, prompt=topic, mode="quiz")
        await send_smart_message(message, response)
    else:
        text = (
            "Включен режим: ТЕСТ И ПЕРЕСКАЗ (/b).\n"
            "Отправь фото параграфа/конспекта или напиши тему.\n"
            "Я сделаю краткий пересказ сути и составлю тест для самопроверки.\n\n"
            "Чтобы вернуться к обычному решению, нажми /a или /new"
        )
        await message.answer(text, parse_mode=None, reply_markup=KEYBOARD)


# Обычный режим решения задач: команды /a, /solve или нажатие кнопки
@dp.message(Command("a"))
@dp.message(Command("solve"))
@dp.message(F.text.startswith("/a"))
@dp.message(F.text == "/a Решить задачу")
async def cmd_solve(message: Message):
    set_user_mode(message.from_user.id, "solve")
    
    raw_text = message.text or ""
    parts = raw_text.split(maxsplit=1)
    
    if len(parts) > 1 and parts[1].strip() and not parts[1].startswith("Решить задачу"):
        question = parts[1].strip()
        await bot.send_chat_action(message.chat.id, ChatAction.TYPING)
        response = await ask_gemini(user_id=message.from_user.id, prompt=question, mode="solve")
        await send_smart_message(message, response)
    else:
        await message.answer("Включен режим: РЕШЕНИЕ ЗАДАЧ (/a). Отправь фото или напиши вопрос.", parse_mode=None, reply_markup=KEYBOARD)


# Сброс контекста: команды /new, /clear, /c или нажатие кнопки
@dp.message(Command("new"))
@dp.message(Command("clear"))
@dp.message(Command("c"))
@dp.message(F.text.startswith("/new"))
@dp.message(F.text == "/new Очистить")
async def cmd_clear(message: Message):
    set_user_mode(message.from_user.id, "solve")
    clear_user_history(message.from_user.id)
    await message.answer("Диалог очищен. Включен обычный режим (/a). Отправь фото или вопрос.", parse_mode=None, reply_markup=KEYBOARD)


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
        await message.answer("Пожалуйста, отправь изображение (JPEG, PNG).")


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
        print(f"\n>>> БОТ @{me.username} ГОТОВ К РАБОТЕ! <<<\n")
        
        # Регистрируем системные команды Telegram для всплывающего меню на телефоне
        await bot.set_my_commands([
            BotCommand(command="a", description="Решить задачу / домашку"),
            BotCommand(command="b", description="Тест и краткий пересказ темы"),
            BotCommand(command="new", description="Очистить диалог"),
        ])
        
        # Запускаем веб-сервер для Render
        await start_health_server()

        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
