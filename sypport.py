import asyncio

from app.models import CoinsTable, PricesTable
from app.queries.orm import select_level_by_coin_and_model


async def main():
    time_frame = '240'
    coin = await CoinsTable.get_by_symbol('BTCUSDT')
    result = await select_level_by_coin_and_model(model=PricesTable, Coin=coin, time_frame=time_frame)
    print(result)

asyncio.run(main())