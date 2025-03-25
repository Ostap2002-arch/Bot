import logging
import sys
from datetime import datetime
from os.path import dirname, abspath
from typing import Optional, Type, TypeVar, List

import pandas as pd
from sqlalchemy import select, update, and_, union_all, func, cast, String
from sqlalchemy.orm import joinedload

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from app.utils import transfer_data_to_dict
from app.models import Base, CoinsTable, PricesTable, LevelsTable, PriorityLevelsTable, UserLevelsTable
from database import engine as sync_engine
from database import session_maker as sync_session


def create_table_all():
    """
    Функция создаёт все необходимые таблицы
    """
    Base.metadata.create_all(sync_engine)


def drop_table_all():
    """
    Функция удаляет все таблицы
    """
    Base.metadata.drop_all(sync_engine)


def drop_table(table) -> None:
    """Функция удаления определённой таблицы
    Функция получает на вход декларативную таблицу и удаляется. Функция ничего не возвращает.

    Args:
        table (Base): Декларативная база данных, на основе класса Base

    """
    with sync_engine.connect() as conn:
        table.__table__.drop(conn, checkfirst=True)
        conn.commit()


def create_table(table) -> None:
    """Функция создания определённой таблицы
    Функция получает на вход декларативную таблицу и создаёт её. Функция ничего не возвращает.

    Args:
        table (Base): Декларативная база данных, на основе класса Base

    """
    with sync_engine.connect() as conn:
        table.__table__.create(conn, checkfirst=True)
        conn.commit()


async def select_data_orm(Coin: CoinsTable,
                        time_frame: int = '240'):
    """Функция возвращает всю информацию о баре
    Функция принимает на вход монету и тайм-фрейм

    Args:
        Coin (CoinsTable): монета
        time_frame (int): тайм-фрейм
    """
    id_coin = Coin.id
    async with sync_session() as session:
        async with session.begin():
            query = select(PricesTable).filter(
                and_(PricesTable.id_coin == id_coin,
                     PricesTable.time_frame == cast(time_frame, String))).order_by(PricesTable.Date)
            result = await session.execute(query)
            return result.scalars().all()


async def select_level_by_coin_and_model(model: Base,
                                         Coin: CoinsTable,
                                         time_frame: str = '240'):
    """Функция возвращает уровни
        Функция принимает на вход монету и тайм-фрейм и модель из которой берутся уровни

        Args:
            model (Base): модель
            Coin (CoinsTable): монета
            time_frame (str): тайм-фрейм
        """
    id_coin = Coin.id
    async with sync_session() as session:
        async with session.begin():
            query = select(model).filter(
                and_(model.id_coin == id_coin,
                     model.time_frame == cast(time_frame, String))
            ).options(joinedload(model.coin))
            result = await session.execute(query)
            result = result.scalars().all()
            print(result)
    return result


async def select_level(Coin: CoinsTable,
                 time_frame: str = '240', ):
    """Функция возвращает уровни
    Функция принимает на вход монету и тайм-фрейм

    Args:
        Coin (CoinsTable): монета
        time_frame (int): тайм-фрейм
    """
    id_coin = Coin.id
    async with sync_session() as session:
        async with session.begin():
            query = select(LevelsTable).filter(
                and_(LevelsTable.id_coin == id_coin,
                     LevelsTable.time_frame == cast(time_frame, String))
            ).options(joinedload(LevelsTable.coin))
            result = session.execute(query).scalars().all()
            return result


async def select_priority_level(Coin: CoinsTable,
                          time_frame: str = '240', ):
    """Функция возвращает уровни
    Функция принимает на вход монету и тайм-фрейм

    Args:
        Coin (CoinsTable): монета
        time_frame (int): тайм-фрейм
    """
    id_coin = Coin.id
    async with sync_session() as session:
        async with session.begin():
            query = select(PriorityLevelsTable).filter(
                and_(PriorityLevelsTable.id_coin == id_coin,
                     PriorityLevelsTable.time_frame == cast(time_frame, String))
            ).options(joinedload(PriorityLevelsTable.coin))
            result = session.execute(query).scalars().all()
            return result


async def select_coins():
    """Функция возвращает монеты.
    Функция на вход ничего не принимает и и возвращает список монет.

    """
    async with sync_session() as session:
        async with session.begin():
            query = select(CoinsTable)
            result = await session.execute(query)
            result = result.scalars().all()
            return result


async def select_data_by_date(Coin: CoinsTable,
                        time_frame: str = '240',
                        start: Optional[datetime] = None,
                        stop: Optional[datetime] = None) -> list:
    """Функция возвращает список бар

    Функция принимает на вход монету, тайм-фрейм(необязательно) и две даты (необязательно заданы) и возвращается список всех бар между указанными числами.
    Если на вход не будут заданны start и stop функция вернёт все записи, как это бы сделала select_data_orm.

    Args:
        Coin (CoinsTable): монета
        time_frame (int): тайм-фрейм
        start (Optional[datetime]): Дата начала списка (опционально)
        stop (Optional[datetime]): Дата конца списка (опционально)

    Returns:
        list: Список бар между датами
    """
    async with sync_session() as session:
        async with session.begin():
            id_coin = Coin.id
            if start and stop:
                query = select(PricesTable).filter(
                    and_(start <= PricesTable.Date,
                         PricesTable.Date <= stop,
                         PricesTable.id_coin == id_coin,
                         PricesTable.time_frame == cast(time_frame, String)
                         )
                ).order_by(PricesTable.Date)
            elif start:
                query = select(PricesTable).filter(
                    and_(start <= PricesTable.Date,
                         PricesTable.id_coin == id_coin,
                         PricesTable.time_frame == cast(time_frame, String)
                         )
                ).order_by(PricesTable.Date)
            elif stop:
                query = select(PricesTable).filter(
                    and_(stop >= PricesTable.Date,
                         PricesTable.id_coin == id_coin,
                         PricesTable.time_frame == cast(time_frame, String)
                         )
                ).order_by(PricesTable.Date)
            else:
                return await select_data_orm(Coin, time_frame)
            result = await session.execute(query)
            return result.scalars().all()


async def clear_table_orm(Table_orm: Type[Base]) -> None:
    """Функция очищения таблицы базы данных.
    Функция принимает декларативную таблицу и очищает её. Сама функция ничего не возвращает.

    Args:
        Table_orm (Base): декларативная база данных
    """
    async with sync_session() as session:
        async with session.begin():
            session.execute(Table_orm.__table__.delete())
            session.commit()


async def insert_info_table_orm(Model_info_add: Base, **kwargs) -> None:
    """" Функция вставки данных в таблицу.
    Функция принимает модель декларативной таблицы и kwargs, которые представляют ключ-значение для записи.

    Args:
        Model_info_add (Base): модель декларативной таблицы
        kwargs (dict): значения для записи в виде dict, где key - название столбца, value - новая запись

    """
    async with sync_session() as session:
        async with session.begin():
            new_record = Model_info_add(**kwargs)
            await session.add(new_record)
            await session.commit()


async def insert_info_array(List_table: list) -> None:
    """"Функция вставки массива данных в таблицу.
        Функция принимает список моделей декларативной таблицы и записывает их в таблицу.

        Args:
            List_table (list): список моделей декларативной таблицы
    """
    async with sync_session() as session:
        async with session.begin():
            session.add_all(List_table)
            await session.commit()


async def update_table(Model_table_update: Base, date: datetime, values: dict) -> None:
    """Функция обновления записей в базе данных.
    Функция принимает модель таблицы базы данных, дату бара и значения.

    Args:
        Model_table_update (Base): модель декларативный таблицы.
        date (datetime): дата изменяемого бара.
        values (dict): новые значения.
    """
    async with sync_session() as session:
        async with session.begin():
            stmt = update(Model_table_update).where(Model_table_update.Date == date).values(**values)
            session.execute(stmt)
            session.commit()


async def find_last_and_first_by_date(Coin: CoinsTable,
                                time_frame: str = '240') -> List[PricesTable]:
    """Фукция поиска последнего и первого элемента по дате
    Функция возвращает список из двух элементов первого и конечного элемента.

    :param Coin: Монета
    :param time_frame: data-frame
    :return:
    """
    id_coin = Coin.id
    async with sync_session() as session:
        async with session.begin():
            query = select(PricesTable).filter(and_(
                PricesTable.id_coin == id_coin,
                PricesTable.time_frame == cast(time_frame, String)
            )).order_by(PricesTable.Date).options(joinedload(PricesTable.coin))
            first_elem = session.execute(query).scalars().first()
            query = select(PricesTable).filter(and_(
                PricesTable.id_coin == id_coin,
                PricesTable.time_frame == cast(time_frame, String)
            )).order_by(PricesTable.Date.desc()).options(joinedload(PricesTable.coin))
            last_elem = session.execute(query).scalars().first()
            return [first_elem, last_elem]


async def deleted_element(List_elem: list[PricesTable]) -> None:
    """Функция удаления списка элементов из базы данных

    Args:
        List_elem: список элементов записей из базы данных которые надо удалить
    """
    if List_elem.count(None) == len(List_elem):
        pass
    async with sync_session() as session:
        async with session.begin():
            for elem in List_elem:
                session.delete(elem)
                session.commit()


def check_integrity(Coin: CoinsTable, time_frame: str = '240') -> bool:
    """Функция проверки целостности записей в базе данных
    Функция принимает на вход монету и time-frame
    Args:
        Coin: монета
        time_frame: тайм-фрейм
    """
    df = pd.DataFrame(transfer_data_to_dict(select_data_by_date(Coin, time_frame)))
    df['time_frame'] = df['time_frame'].astype(int)
    df['delta_time'] = df['Date'].diff()
    df['delta_time'] = df['delta_time'].apply(lambda x: int(x.total_seconds() // 60) if pd.notna(x) else None)
    df['cor_to_time'] = df['delta_time'] == df['time_frame']
    count = (df['cor_to_time'] == False).sum()
    if count > 1:
        return False
    else:
        return True


async def clear_table_by_coin(Coin: CoinsTable, time_frame: str = '240') -> None:
    """Функция очистки базы данных от всех записей по монете и тайм-фрейму
    Функция принимает на вход монету и time-frame и очищает бд
    Args:
        Coin: монета
        time_frame: тайм-фрейм
    """
    Deleted_list = select_data_orm(Coin, time_frame)

    async with sync_session() as session:
        async with session.begin():
            for elem in Deleted_list:
                session.delete(elem)
                session.commit()


async def delete_objects(list_objects: list) -> None:
    async with sync_session() as session:
        async with session.begin():
            for elem in list_objects:
                session.delete(elem)
                session.commit()


async def ge_and_count_important_level():
    async with sync_session() as session:
        async with session.begin():
            stmt1 = select(
                PriorityLevelsTable.id_coin,
                PriorityLevelsTable.Level,
                PriorityLevelsTable.time_frame
            )

            stmt2 = select(
                UserLevelsTable.id_coin,
                UserLevelsTable.Level,
                UserLevelsTable.time_frame
            )

            # Объединяем подзапросы с помощью union_all
            union_query = union_all(stmt1, stmt2).alias('union_table')

            # Создаем основной запрос для группировки и подсчета
            grouped_query = select(
                union_query.c.id_coin,
                union_query.c.time_frame,
                CoinsTable.symbol,
                func.count(union_query.c.Level).label('level_count')
            ).select_from(
                union_query.outerjoin(CoinsTable, union_query.c.id_coin == CoinsTable.id)
            ).group_by(
                union_query.c.id_coin,
                union_query.c.time_frame,
                CoinsTable.symbol
            )
            # Выполняем запрос
            result = await session.execute(grouped_query)

            # Возвращаем результаты
            return result.all()
