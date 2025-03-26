import asyncio
import json
import logging
import sys
from datetime import datetime
from os.path import dirname, abspath
from typing import List

from aiogram import Bot
from pybit.unified_trading import HTTP
import pandas as pd

sys.path.insert(0, dirname(abspath(__file__)))

from levels import find_levels
from models import CoinsTable, PricesTable, LevelsTable, HideLevelsTable, PriorityLevelsTable, UserLevelsTable
from queries.orm import insert_info_array, select_data_orm, select_level, select_coins, select_priority_level, \
    select_level_by_coin_and_model, delete_objects
from utils import transfer_data_to_dict, find_quantile


class GetBars:
    dict_time_frame = {
        '240': 186,
        '720': 155}

    def __init__(self,
                 coin: CoinsTable,
                 bot_utils: Bot,
                 chat_id: str,
                 time_frame: int | str = 240,
                 category: str = 'linear',
                 ):
        self.coin = coin
        self.time_frame = time_frame
        self.category = category
        self.limit = self.dict_time_frame[time_frame]
        self.bot_utils = bot_utils
        self.chat_id = chat_id

    async def get_data(self):
        """"Функция запроса данных.
        Используется при запуске приложения и заполняет всю базу данных за определённый промежуток времени
        """
        old_time = None
        timer = 14400
        while True:
            now_time = asyncio.get_event_loop().time()
            if not old_time is None and now_time - old_time < timer:
                yield asyncio.sleep(0)
                continue
            try:
                session = HTTP()
                request = session.get_index_price_kline(
                    category=self.category,
                    symbol=self.coin.symbol,
                    interval=self.time_frame,
                    limit=self.limit,
                )
            except Exception as e:
                await self.bot_utils.send_message(self.chat_id,
                                                  f'Ошибка при запросе данных о монете {self.coin} '
                                                  f'тайм-фрейм {self.time_frame} повторная попытка через 5 минут',
                                                  request_timeout=30)
                timer = 1800
                old_time = asyncio.get_event_loop().time()
                yield asyncio.sleep(0)
                continue

            if request['retMsg'] == 'OK':
                request = request['result']['list']
                List_prices = [PricesTable(
                    Date=datetime.fromtimestamp(int(x[0]) / 1000),
                    time_frame=str(self.time_frame),
                    Open=float(x[1]),
                    High=float(x[2]),
                    Low=float(x[3]),
                    Close=float(x[4]),
                    # coin=self.coin,
                    id_coin=self.coin.id
                ) for x in request]
                await PricesTable.clear_by_model(self.coin, self.time_frame)
                await insert_info_array(List_prices)
                await GetLevels(coin=self.coin, time_frame=self.time_frame).find_levels()
                try:
                    await self.bot_utils.send_message(self.chat_id,
                                                  f'Загружены цены и уровни для монеты {self.coin} тайм-фрейм {self.time_frame}',
                                                      request_timeout=30
                                                      )
                except Exception:
                    #print(f'Загружены цены и уровни для монеты {self.coin} тайм-фрейм {self.time_frame}')
                    timer = 1800
                    old_time = asyncio.get_event_loop().time()
                    yield asyncio.sleep(0)
                    continue
            else:
                try:
                    await self.bot_utils.send_message(self.chat_id,
                                                  f'Ошибка при запросе данных о монете {self.coin} тайм-фрейм {self.time_frame} '
                                                  f'со стороны сервера повторная попытка через 5 минут',
                                                      request_timeout=30)
                    timer = 1800
                    old_time = asyncio.get_event_loop().time()
                    yield asyncio.sleep(0)
                    continue
                except Exception:
                    timer = 1800
                    old_time = asyncio.get_event_loop().time()
                    yield asyncio.sleep(0)
                    continue
            timer = 14400
            old_time = asyncio.get_event_loop().time()
            yield asyncio.sleep(0)
            continue



class GetLevels:
    dict_time_frame = {
        '240': 186,
        '720': 155}

    def __init__(self,
                 coin: CoinsTable,
                 time_frame: str = '240',
                 ):
        self.coin = coin
        self.time_frame = time_frame

    async def find_levels(self):
        """Функция создания уровней"""
        data = await select_data_orm(self.coin, self.time_frame)
        data = pd.DataFrame(transfer_data_to_dict(data))
        Levels = list()
        for index, level in find_levels(data).iterrows():
            Levels.append(LevelsTable(id_coin=self.coin.id,
                                      time_frame=str(self.time_frame),
                                      Level=level['Level'],
                                      Mother_date=json.dumps(list(map(str, level['mother_date']))),
                                      Mother_prices=json.dumps(list(map(str, level['mother_prices']))),
                                      Source=json.dumps(list(map(str, level['Source']))),
                                      ))
        await LevelsTable.clear_by_model(self.coin, self.time_frame)
        await insert_info_array(Levels)

    async def get_levels(self):
        """Функция получения уровней"""
        return await select_level_by_coin_and_model(LevelsTable, self.coin, self.time_frame)


class GetCoins:

    def __init__(self,
                 coins: List[str],
                 bot_utils: Bot,
                 chat_id: str
                 ):
        self.coins = coins
        self.bot_utils = bot_utils
        self.chat_id = chat_id

    async def loading_coins(self):
        """"Функция записи списка монет для парсера.
        Используется при запуске приложения и заполняет всю базу данных инициалами монет
        """
        while True:
            await PricesTable.total_clear()
            await LevelsTable.total_clear()
            try:
                await CoinsTable.total_clear()
                List_coin = [CoinsTable(symbol=symbol) for symbol in self.coins]
                await insert_info_array(List_coin)
                try:
                    await self.bot_utils.send_message(self.chat_id,
                                                      f'Список монет успешно загружен в базу данных',
                                                      request_timeout=30)
                except Exception:
                    pass
                break
            except Exception as e:
                try:
                    await self.bot_utils.send_message(self.chat_id,
                                                  f'При записи монет произошла ошибка {e} через 5 минут будет повторная попытка записи монет',
                                                  request_timeout=30)
                except Exception:
                    pass
                await asyncio.sleep(300)

    @classmethod
    async def get_coins(cls):
        try:
            return await select_coins()
        except Exception as e:
            return []


class GetPriorityLevels:
    model = PriorityLevelsTable

    def __init__(self,
                 coin: CoinsTable,
                 time_frame: str = '240',
                 ):
        self.coin = coin
        self.time_frame = time_frame

    async def get_levels(self):
        """Функция получения уровней"""
        result = await select_level_by_coin_and_model(self.__class__.model, self.coin, self.time_frame)
        return result

    def add_levels(self, levels: List[float]):
        """Метод добавления уровней
        Функция принимает  список уровней для добавления в бд.

        Args:
            levels (List[float]): список уровней

        :return None
        """
        List_priority_levels = [self.__class__.model(id_coin=self.coin.id,
                                                     time_frame=self.time_frame,
                                                     Level=level,
                                                     coin=self.coin) for level in levels]
        insert_info_array(List_priority_levels)

    def deleted_levels(self, deleted_levels: List[float], ):
        """Метод удаления уровней
        Функция принимает монету, таймфрем и список уровней.

        Args:
            deleted_levels (List[float]): список уровней

        :return None
        """
        List_levels = self.get_levels()
        deleted_levels = [level for level in List_levels if level.Level in deleted_levels]
        delete_objects(deleted_levels)

    def all_deleted_levels(self):
        List_levels = self.get_levels()
        delete_objects(List_levels)


class GetHideLevels(GetPriorityLevels):
    model = HideLevelsTable


class GetUserLevels(GetPriorityLevels):
    model = UserLevelsTable


class CheckPrice:
    def __init__(self,
                 coin: CoinsTable,
                 bot: Bot,
                 CHAT_ID: str,
                 time_frame: str = '240',
                 category: str = 'linear',
                 ):
        self.coin = coin
        self.time_frame = str(time_frame)
        self.category = category
        self.bot = bot
        self.CHAT_ID = CHAT_ID

    async def check(self):
        old_time = None
        timer = 1800
        while True:
            #print(f'приоритет {self.coin}, {self.time_frame}')
            now_time = asyncio.get_event_loop().time()
            if not old_time is None and now_time - old_time < timer:
                yield asyncio.sleep(1)
                continue
            try:
                session = HTTP()
                request = session.get_index_price_kline(
                    category=self.category,
                    symbol=self.coin.symbol,
                    interval=self.time_frame,
                    limit=1,
                )
            except:
                try:
                    await self.bot.send_message(self.CHAT_ID,
                                            f'Ошибка при получении данных (Check) к монете {self.coin.symbol} по тайм-фрему {self.time_frame}',
                                                request_timeout=30)
                except Exception:
                    pass
                timer = 1800
                old_time = asyncio.get_event_loop().time()
                yield asyncio.sleep(1)
                continue
            if request['retMsg'] == 'OK':
                price = float(request['result']['list'][0][4])
            else:
                try:
                    await self.bot.send_message(self.CHAT_ID,
                                            f'Ошибка при получении данных (Check) к монете {self.coin.symbol} на стороне сервера по тайм-фрему {self.time_frame}',
                                                request_timeout=30)
                except Exception:
                    pass
                timer = 1800
                old_time = asyncio.get_event_loop().time()
                yield asyncio.sleep(1)
                continue
            Levels = (await GetPriorityLevels(coin=self.coin, time_frame=self.time_frame).get_levels() +
                      await GetUserLevels(coin=self.coin, time_frame=self.time_frame).get_levels())
            Levels = list(map(lambda x: x.Level, Levels))
            if not bool(Levels):
                timer = 900
                old_time = asyncio.get_event_loop().time()
                yield asyncio.sleep(1)
                continue
            else:
                level = list(sorted(Levels, key=lambda x: abs(x - price)))[0]
                percent = round((level - price) / level * 100, 2)
                if abs(percent) < 0.5:
                    messange = f'''Монета {self.coin.symbol}
Цена {price}
Тайм-фрейме {self.time_frame}
Уровень {level:6}
Расстояние {percent} %
----------------------------------------------'''
                    try:
                        await self.bot.send_message(self.CHAT_ID, messange, request_timeout=30)
                    except Exception:
                        pass
                    timer = 1800
                    old_time = asyncio.get_event_loop().time()
                    yield asyncio.sleep(0)
                    continue
            timer = 1800
            old_time = asyncio.get_event_loop().time()
            yield asyncio.sleep(1)
            continue


class CheckPriceSpam:
    def __init__(self,
                 coin: CoinsTable,
                 bot: Bot,
                 CHAT_ID: str,
                 time_frame:  str = '240',
                 category: str = 'linear',
                 ):
        self.coin = coin
        self.time_frame = str(time_frame)
        self.category = category
        self.bot = bot
        self.CHAT_ID = CHAT_ID

    async def check(self):
        old_time = None
        timer = 1800
        while True:
            now_time = asyncio.get_event_loop().time()
            #print(f'спам {self.coin}, {self.time_frame}')
            if not old_time is None and now_time - old_time < timer:
                yield asyncio.sleep(1)
                continue
            try:
                session = HTTP()
                request = session.get_index_price_kline(
                    category=self.category,
                    symbol=self.coin.symbol,
                    interval=self.time_frame,
                    limit=1,
                )
            except:
                try:
                    await self.bot.send_message(self.CHAT_ID,
                                            f'Ошибка при получении данных (Spam) к монете {self.coin.symbol} по тайм-фрему {self.time_frame}',
                                                request_timeout=30)
                except Exception:
                    pass
                timer = 1800
                old_time = asyncio.get_event_loop().time()
                yield asyncio.sleep(1)
                continue
            if request['retMsg'] == 'OK':
                price = float(request['result']['list'][0][4])
            else:
                try:
                    await self.bot.send_message(self.CHAT_ID,
                                            f'Ошибка при получении данных (Spam) к монете {self.coin.symbol} на стороне сервера по тайм-фрему {self.time_frame}')
                except Exception:
                    pass
                timer = 1800
                old_time = asyncio.get_event_loop().time()
                yield asyncio.sleep(1)
                continue
            Levels = await GetLevels(coin=self.coin, time_frame=self.time_frame).get_levels()
            LevelsLevel = list(map(lambda x: x.Level, Levels))
            if not bool(LevelsLevel):
                timer = 900
                old_time = asyncio.get_event_loop().time()
                yield asyncio.sleep(1)
                continue
            else:
                level = list(sorted(LevelsLevel, key=lambda x: abs(x - price)))[0]
                percent = round((level - price) / level * 100, 2)
                if abs(percent) < 0.1:
                    Q, confirmation = find_quantile(Levels, level)
                    # Поиск соседних уровней
                    for i, lev in enumerate(Levels):
                        if lev.Level == level:
                            try:
                                level_upp = Levels[i + 1]
                                level_upp = level_upp.Level
                            except Exception:
                                level_upp = None
                            try:
                                level_down = Levels[i - 1]
                                level_down = level_down.Level
                            except Exception:
                                level_down = None
                        else:
                            continue
                        if level_upp is not None:
                            delta_upp = round(((level_upp - level) / level) * 100, 2)
                            Q_upp, confirmation_upp = find_quantile(Levels, level_upp)
                        else:
                            delta_upp = None
                            Q_upp, confirmation_upp = None, None
                        if level_down is not None:
                            delta_down = round(((level - level_down) / level) * 100, 2)
                            Q_down, confirmation_down = find_quantile(Levels, level_down)
                        else:
                            delta_down = None
                            Q_down, confirmation_down = None, None

                        messange = f'''Монета {self.coin.symbol} \n
Цена {price} \n
Тайм-фрейме {self.time_frame} \n
Уровень {level} Сила {Q} Подтверждений {confirmation} Расстояние {percent} % \n
Уровень выше {level_upp} Сила {Q_upp} Подтверждений {confirmation_upp} Расстояние {delta_upp} % \n
Уровень ниже {level_down} Сила {Q_down} Подтверждений {confirmation_down} Расстояние {delta_down} %'''
                        try:
                            await self.bot.send_message(self.CHAT_ID, messange, request_timeout=30)
                        except Exception:
                            pass
                        timer = 1800
                        old_time = asyncio.get_event_loop().time()
                        yield asyncio.sleep(1)
                        continue
                timer = 1800
                old_time = asyncio.get_event_loop().time()
                yield asyncio.sleep(1)
                continue
