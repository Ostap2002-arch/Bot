import asyncio
import time

import logging

# from pybit import HTTP
from aiogram import Bot, types

from app.bot import start_bot
from app.check_tasks import TaskMonitor
from app.models import CoinsTable
from app.mother_data import GetBars, GetCoins, CheckPrice, GetLevels, CheckPriceSpam
from app.pool import SuspendedTaskPool
from config import List_coins

# Настройки
API_KEY = 'YOUR_BYBIT_API_KEY'
API_SECRET = 'YOUR_BYBIT_API_SECRET'
TELEGRAM_BOT_TOKEN = '7972460979:AAGKfa0jPPWBrD1aKtLrR2p9QGxnZnqEuqg'
TELEGRAM_BOT_TOKEN_UTILS = '7666917890:AAE_PwW2daxVH5VCcLapyjS6wAwzm5vJeW8'
TELEGRAM_BOT_TOKEN_SPAM = '7579275056:AAHmP48y1pud7_DXxGcMpusIKt5kB_YPl1U'
TELEGRAM_CHAT_ID = '5182604068'

bot = Bot(token=TELEGRAM_BOT_TOKEN)
bot_utils = Bot(token=TELEGRAM_BOT_TOKEN_UTILS)
bot_spam = Bot(token=TELEGRAM_BOT_TOKEN_SPAM)


async def main():
    await GetCoins(List_coins, bot_utils=bot_utils, chat_id=TELEGRAM_CHAT_ID).loading_coins()

    Coins = await GetCoins.get_coins()
    List_time_frame = ['240'] * len(Coins) + ['720'] * len(Coins)
    Coins = Coins + Coins

    GetList = [GetBars(coin=coin, time_frame=time_frame, bot_utils=bot_utils, chat_id=TELEGRAM_CHAT_ID) for coin, time_frame in zip(Coins, List_time_frame)]

    CheckList = [CheckPrice(coin=coin,
                            time_frame=timeframe,
                            bot=bot,
                            CHAT_ID=TELEGRAM_CHAT_ID) for coin, timeframe in zip(Coins, List_time_frame)]

    CheckList_Spam = [CheckPriceSpam(coin=coin,
                            time_frame=timeframe,
                            bot=bot_spam,
                            CHAT_ID=TELEGRAM_CHAT_ID) for coin, timeframe in zip(Coins, List_time_frame)]

    async def check():
        monitor = TaskMonitor()
        await monitor.start()
        while True:
            monitor.print_stats()
            yield asyncio.sleep(30)


    Tasks = ([get.get_data for get in GetList] +
             [check.check for check in CheckList] +
             [check.check for check in CheckList_Spam]
              )

    pool = SuspendedTaskPool(max_concurrent=5)

    for task in Tasks:
        await pool.add_task(task)

    await asyncio.gather(*[pool.run(), start_bot(bot)])


asyncio.run(main())
