import datetime
import sys
from os.path import dirname, abspath
from typing import Annotated

from sqlalchemy import Integer, DateTime, Float, ForeignKey, String, JSON, select, and_, delete
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

sys.path.insert(0, dirname(abspath(__file__)))

from database import session_maker as sync_session


class Base(DeclarativeBase):
    @classmethod
    async def clear_by_model(cls, coin, time_frame):
        """Функция очистка таблицы"""
        async with sync_session() as session:
            async with session.begin():  # Контекстный менеджер для транзакции
                stmt = delete(cls).where(
                    and_(
                        cls.id_coin == coin.id,
                        cls.time_frame == time_frame
                    )
                )
                await session.execute(stmt)

    @classmethod
    async def total_clear(cls):
        """Функция очистка таблицы"""
        async with sync_session() as session:
            async with session.begin():
                await session.execute(delete(cls))


intpk = Annotated[int, mapped_column(primary_key=True)]


class PricesTable(Base):
    __tablename__ = 'PricesTable'
    id: Mapped[intpk] = mapped_column(Integer)
    id_coin: Mapped[int] = mapped_column(Integer, ForeignKey('CoinsTable.id'))
    time_frame: Mapped[str] = mapped_column(String)
    Date: Mapped[datetime.datetime] = mapped_column(DateTime)
    Open: Mapped[float] = mapped_column(Float)
    Close: Mapped[float] = mapped_column(Float)
    High: Mapped[float] = mapped_column(Float)
    Low: Mapped[float] = mapped_column(Float)
    coin: Mapped['CoinsTable'] = relationship('CoinsTable', back_populates='prices')

    def __str__(self):
        return f'{self.coin}({self.Date})'

    def __repr__(self):
        return self.__str__()


class LevelsTable(Base):
    __tablename__ = 'LevelsTable'
    id: Mapped[intpk] = mapped_column(Integer)
    id_coin: Mapped[int] = mapped_column(Integer, ForeignKey('CoinsTable.id'))
    time_frame: Mapped[str] = mapped_column(String)
    Level: Mapped[float] = mapped_column(Float)
    Mother_date: Mapped[JSON] = mapped_column(JSON)
    Mother_prices: Mapped[JSON] = mapped_column(JSON)
    Source: Mapped[JSON] = mapped_column(JSON)
    coin: Mapped['CoinsTable'] = relationship('CoinsTable', back_populates='levels')

    def __str__(self):
        return f'Level_{self.coin}({self.Level})'

    def __repr__(self):
        return self.__str__()


class PriorityLevelsTable(Base):
    __tablename__ = 'PriorityLevelsTable'
    id: Mapped[intpk] = mapped_column(Integer)
    id_coin: Mapped[int] = mapped_column(Integer, ForeignKey('CoinsTable.id'))
    time_frame: Mapped[str] = mapped_column(String)
    Level: Mapped[float] = mapped_column(Float)
    coin: Mapped['CoinsTable'] = relationship('CoinsTable', back_populates='priority_levels')

    def __str__(self):
        return f'Level_{self.coin}({self.Level})'

    def __repr__(self):
        return self.__str__()


class HideLevelsTable(Base):
    __tablename__ = 'HideLevelsTable'
    id: Mapped[intpk] = mapped_column(Integer)
    id_coin: Mapped[int] = mapped_column(Integer, ForeignKey('CoinsTable.id'))
    time_frame: Mapped[str] = mapped_column(String)
    Level: Mapped[float] = mapped_column(Float)
    coin: Mapped['CoinsTable'] = relationship('CoinsTable', back_populates='hide_levels')

    def __str__(self):
        return f'Hide_Level_{self.coin}({self.Level})'

    def __repr__(self):
        return self.__str__()


class UserLevelsTable(Base):
    __tablename__ = 'UserLevelsTable'
    id: Mapped[intpk] = mapped_column(Integer)
    id_coin: Mapped[int] = mapped_column(Integer, ForeignKey('CoinsTable.id'))
    time_frame: Mapped[str] = mapped_column(String)
    Level: Mapped[float] = mapped_column(Float)
    coin: Mapped['CoinsTable'] = relationship('CoinsTable', back_populates='user_levels')

    def __str__(self):
        return f'Hide_Level_{self.coin}({self.Level})'

    def __repr__(self):
        return self.__str__()


class CoinsTable(Base):
    __tablename__ = 'CoinsTable'
    id: Mapped[intpk] = mapped_column(Integer)
    symbol: Mapped[str] = mapped_column(String, unique=True)
    prices: Mapped[list[PricesTable]] = relationship('PricesTable', back_populates='coin')
    levels: Mapped[list[LevelsTable]] = relationship('LevelsTable', back_populates='coin')
    priority_levels: Mapped[list[LevelsTable]] = relationship('PriorityLevelsTable', back_populates='coin')
    hide_levels: Mapped[list[LevelsTable]] = relationship('HideLevelsTable', back_populates='coin')
    user_levels: Mapped[list[LevelsTable]] = relationship('UserLevelsTable', back_populates='coin')

    def __str__(self):
        return f'Coin({self.symbol})'

    def __repr__(self):
        return self.__str__()

    @classmethod
    async def get_by_symbol(cls, symbol):
        async with sync_session() as session:
            async with session.begin():
                query = select(cls).filter_by(symbol=symbol)
                coin = await session.execute(query)
                return coin.scalars().first()
