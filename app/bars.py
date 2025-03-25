# Пользовательские классы, которые будут передаваться между функциями во всё проекте
import sys
from datetime import datetime
from os.path import dirname, abspath
from typing import Optional
from functools import total_ordering

import pandas as pd


from utils import transfer_data_to_dict

sys.path.insert(0, dirname(abspath(__file__)))

from app.queries.orm import select_data_by_date
from models import CoinsTable


@total_ordering
class Bar:
    """
    Класс бара, содержит информацию об определенной баре.

    Атрибуты:
        date (datetime): время открытия бара
        open_prise (float): цена открытия
        height_prise (float): цена максимальная цена
        low_prise (float): цена минимальная цена
        close_prise (float): цена закрытия
        body (float): тело свечи
        shadow (float): тень свечи
    Методы:

    """

    def __init__(self, data: pd.Series):
        self.date = data.name.to_pydatetime()
        self.open_prise = data.iloc[0]
        self.height_prise = data.iloc[1]
        self.low_prise = data.iloc[2]
        self.close_prise = data.iloc[3]
        self.body = abs(self.open_prise - self.close_prise)
        self.shadow = abs(self.height_prise - self.low_prise)

    def __str__(self):
        return f"Bar({self.date.strftime('%Y-%m-%d %H:%M:%S')})"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.date == other.date
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, self.__class__):
            return self.date > other.date
        else:
            return NotImplemented


class Bars:
    """
    Класс баров, даёт возможность быстро и просто получать информацию для работы с ней

    Атрибуты:
        base_data (pd.DataFrame): Вся информация за данный период в формате DataFrame
    Методы:
        get_bar(self, date): Возвращает класс Bar по заданной дате
        get_data_for_mplfinance(self): Возвращает данные для построения графиков в mplfinance.

    """

    def __init__(self,
                 Coin: CoinsTable,
                 time_frame: int = 240,
                 date_start: Optional[datetime] = None,
                 date_stop: Optional[datetime] = None):
        """ Метод инициализации класса и создания атрибутов.
        На вход принимает монету, тайм-фрейм, начало и конец диапазона в датах. Если не задано начало, берется с самого начала бд,
        если не задан конец, до конца бд.

        :param
            Coin (CoinsTable): монета, по которой происходит сбор данных
            time_frame (int): тайм-фрейм (в минутах), по умолчанию 240 (4H)
            date_start (Optional[datetime]): начала диапазона (опционально)
            date_stop (Optional[datetime]): конец диапазона (опционально)

        """
        self.Coin = Coin
        self.time_frame = time_frame
        self.date_start = date_start
        self.date_stop = date_stop

    async def get_bar(self, date: datetime) -> Bar:
        """
        Метод получения определённого бара из списка бар по дате.

        Args:
            date (datetime): Дата получаемого бара

        Returns:
            Bar: Бар по данной дате

        Raises:
            ValuerError: Возникает, когда запрашиваемой даты нет в диапазоне

        """
        inter_data = await self.get_data_for_mplfinance()
        Series = inter_data.loc[pd.Timestamp(date)]
        return Bar(Series)

    async def get_data_for_mplfinance(self) -> pd.DataFrame:
        """
        Метод возвращает данные для построения графиков в mplfinance.

        Returns:
            pd.DataFrame: Возвращает DataFrame с index = Date

        """
        data_models = await select_data_by_date(self.Coin, self.time_frame, self.date_start, self.date_stop)
        data_dict = transfer_data_to_dict(data_models)
        base_data = pd.DataFrame(data_dict)
        inter_data = base_data.copy()
        inter_data = inter_data[['Date', 'Open', 'High', 'Low', 'Close']]
        inter_data.set_index('Date', inplace=True)
        return inter_data

    def __str__(self):
        return str(f'Bars({self.Coin})')

