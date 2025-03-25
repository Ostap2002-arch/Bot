import asyncio

import pandas as pd
import psutil
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, Message

from app.mother_data import GetCoins
from app.queries.orm import ge_and_count_important_level

keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Список монет")],  # Первая строка с одной кнопкой
        [KeyboardButton(text="Список отслеживающих монет")],  # Вторая строка с одной кнопкой
        [KeyboardButton(text="📊 Статистика бота")]
    ],
    resize_keyboard=True  # Опционально: автоматически изменять размер клавиатуры
)

dp = Dispatcher()


@dp.message(F.text == "Список монет")
async def handle_hello(message: Message):
    answer = await GetCoins.get_coins()
    answer = '\n'.join(map(lambda x: x.symbol, answer))
    await message.answer(f"Список монет \n {answer}", reply_markup=keyboard)


@dp.message(F.text == "Список отслеживающих монет")
async def handle_hello(message: Message):
    answer = await ge_and_count_important_level()
    df = pd.DataFrame(columns=['id', 'time_frame', 'coin', 'count_level'], data=answer)
    answer = df[['coin', 'time_frame', 'count_level']]
    await message.answer(f"Список уровней \n {answer}", reply_markup=keyboard)


@dp.message(F.text == "📊 Статистика бота")
async def send_bot_stats(message: Message):
    # Получаем информацию о процессах
    process = psutil.Process()

    # Активные задачи asyncio
    tasks = [t for t in asyncio.all_tasks() if not t.done()]

    # Системная информация
    cpu_percent = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()

    # Собираем сообщение
    stats_message = (
        "📊 <b>Статистика бота:</b>\n"
        f"• <b>Задачи:</b> {len(tasks)} активных\n"
        f"• <b>CPU:</b> {cpu_percent}%\n"
        f"• <b>Память:</b> {memory_info.used / 1024 / 1024:.2f} MB / {memory_info.total / 1024 / 1024:.2f} MB\n"
        f"• <b>Загрузка памяти:</b> {memory_info.percent}%\n"
        f"• <b>Память бота:</b> {process.memory_info().rss / 1024 / 1024:.2f} MB"
    )

    await message.answer(stats_message, parse_mode="HTML")


# Запуск бота
async def start_bot(bot: Bot):
    await dp.start_polling(bot)
