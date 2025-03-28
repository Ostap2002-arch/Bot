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
from config import List_coins, settings

TELEGRAM_BOT_TOKEN = settings.TELEGRAM_BOT_TOKEN
TELEGRAM_BOT_TOKEN_UTILS = settings.TELEGRAM_BOT_TOKEN_UTILS
TELEGRAM_BOT_TOKEN_SPAM = settings.TELEGRAM_BOT_TOKEN_SPAM
TELEGRAM_CHAT_ID = settings.TELEGRAM_CHAT_ID

bot = Bot(token=TELEGRAM_BOT_TOKEN)
bot_utils = Bot(token=TELEGRAM_BOT_TOKEN_UTILS)
bot_spam = Bot(token=TELEGRAM_BOT_TOKEN_SPAM)


async def prepare_coins():
    await GetCoins(List_coins, bot_utils=bot_utils, chat_id=TELEGRAM_CHAT_ID).loading_coins()
    return await GetCoins.get_coins()


async def run(Coins):
    List_time_frame = ['240'] * len(Coins) + ['720'] * len(Coins)
    Coins = Coins + Coins

    GetList = [GetBars(coin=coin, time_frame=time_frame, bot_utils=bot_utils, chat_id=TELEGRAM_CHAT_ID) for
               coin, time_frame in zip(Coins, List_time_frame)]

    CheckList = [CheckPrice(coin=coin,
                            time_frame=timeframe,
                            bot=bot,
                            CHAT_ID=TELEGRAM_CHAT_ID) for coin, timeframe in zip(Coins, List_time_frame)]

    CheckList_Spam = [CheckPriceSpam(coin=coin,
                                     time_frame=timeframe,
                                     bot=bot_spam,
                                     CHAT_ID=TELEGRAM_CHAT_ID) for coin, timeframe in zip(Coins, List_time_frame)]

    Tasks = ([get.get_data for get in GetList] +
             [check.check for check in CheckList] +
             [check.check for check in CheckList_Spam]
             )

    pool = SuspendedTaskPool(max_concurrent=5)

    for task in Tasks:
        await pool.add_task(task)

    await asyncio.gather(*[pool.run(), start_bot(bot)])


async def main_async():
    Coins = await prepare_coins()

    while True:
        try:
            await run(Coins)
        except Exception as e:
            await bot.send_message(TELEGRAM_CHAT_ID,
                                   f'Произошла ошибка. Бот перезугружается. Перерыв 30 минут. Ошибка {e}')
            await asyncio.sleep(1800)
            continue


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
