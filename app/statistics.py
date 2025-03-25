import sys
from os.path import dirname, abspath
from typing import List, Tuple, Literal

import numpy as np
from pandas import Series
import pandas as pd

sys.path.insert(0, dirname(abspath(__file__)))

from bars import Bars


def find_local_date_max(data: pd.DataFrame,
                        window: int = 3,
                        by: Literal['body', 'shadow'] = 'body') -> List[pd.Timestamp]:
    """Функция поиска локальных максимумов
    Функция принимает на вход данные о цене и возвращает даты локальных максимумов
    Args:
        data (pd.DataFrame): данные о цене
        window (int): размер скользящего экрана
    """
    data = data.assign(top_price=None)
    data['top_price'] = data[['Open', 'Close']].max(axis=1)
    rolling_shadow_max = data['High'].rolling(window=window, center=True).max()
    rolling_body_max = data['top_price'].rolling(window=window, center=True).max()
    data['rolling_shadow_max'] = (data['High'] == rolling_shadow_max)
    data['rolling_body_max'] = (data['top_price'] == rolling_body_max)
    data = data[data[f'rolling_{by}_max'] == True]
    return data['Date']


def find_local_id_max(data: pd.DataFrame,
                      window: int = 3,
                      by: Literal['body', 'shadow'] = 'body') -> List[pd.Timestamp]:
    """Функция поиска локальных максимумов
    Функция принимает на вход данные о цене и возвращает даты локальных максимумов
    Args:
        data (pd.DataFrame): данные о цене
        window (int): размер скользящего экрана
    """
    data = data.assign(top_price=None)
    data['top_price'] = data[['Open', 'Close']].max(axis=1)
    rolling_shadow_max = data['High'].rolling(window=window, center=True).max()
    rolling_body_max = data['top_price'].rolling(window=window, center=True).max()
    data['rolling_shadow_max'] = (data['High'] == rolling_shadow_max)
    data['rolling_body_max'] = (data['top_price'] == rolling_body_max)
    data = data[data[f'rolling_{by}_max'] == True]
    return data['id']


def find_local_date_min(data: pd.DataFrame,
                        window: int = 3,
                        by: Literal['body', 'shadow'] = 'body') -> pd.Series:
    """Функция поиска локальных минимумов
    Функция принимает на вход данные о цене и возвращает даты локальных минимумов

    Args:
        data (pd.DataFrame): данные о цене
        window (int): размер скользящего экрана
        by (Literal['body', 'shadow'] = 'body')) : По чему ищется относительный минимум (по телу или тени)

    Returns:
        pd.Series: Серию pandas содержащую pd.TimeStamp
    """
    data = data.copy()
    data['lower_price'] = data[['Open', 'Close']].min(axis=1)
    rolling_shadow_min = data['Low'].rolling(window=window, center=True).min()
    rolling_body_min = data['lower_price'].rolling(window=window, center=True).min()
    data['rolling_shadow_min'] = (data['Low'] == rolling_shadow_min)
    data['rolling_body_min'] = (data['lower_price'] == rolling_body_min)
    data = data[data[f'rolling_{by}_min'].isin([True])]
    return data['Date']


def find_local_id_min(data: pd.DataFrame,
                      window: int = 3,
                      by: Literal['body', 'shadow'] = 'body') -> pd.Series:
    """Функция поиска локальных минимумов
    Функция принимает на вход данные о цене и возвращает даты локальных минимумов

    Args:
        data (pd.DataFrame): данные о цене
        window (int): размер скользящего экрана
        by (Literal['body', 'shadow'] = 'body')) : По чему ищется относительный минимум (по телу или тени)

    Returns:
        pd.Series: Серию pandas содержащую pd.TimeStamp
    """
    data = data.copy()
    data['lower_price'] = data[['Open', 'Close']].min(axis=1)
    rolling_shadow_min = data['Low'].rolling(window=window, center=True).min()
    rolling_body_min = data['lower_price'].rolling(window=window, center=True).min()
    data['rolling_shadow_min'] = (data['Low'] == rolling_shadow_min)
    data['rolling_body_min'] = (data['lower_price'] == rolling_body_min)
    data = data[data[f'rolling_{by}_min'].isin([True])]
    return data['id']


def find_anomaly_min_bars(data: pd.DataFrame,
                          N: int = 30,
                          by: Literal['body', 'shadow'] = 'body',
                          p: float = 10.0) -> List[pd.Timestamp]:
    """"Функция поиска аномально маленьких баров
    Функция принимает информацию о цене и возвращает список дат, когда бары были аномально маленькие.

    Args:
        data (pd.DataFrame): Данные о цене
        N (int): Период
        by (str): По какому параметру смотреть аномальность. Принимает значение body или shadow.
        p (float): Процент покрытия по мат распределению, чем больше тем больше аномальных баров, но точность
        поиска будет снижена.

    Returns:
        List[pd.Timestamp] - список дат, когда бары были аномально маленькими
    """

    data = data.copy()
    if by == 'body':
        data['by'] = data['Open'] - data['Close']
    else:
        data['by'] = data['High'] - data['Low']
    data['by'] = data['by'].abs()

    def find_border_left(series: pd.Series) -> float:
        """Вспомогательная функция поиска минимального размера бара
        Функция принимает серию о цене и находит значение меньше которого бар считается аномально маленьким.

        Args:
            series (pd.Series): Срез данных о цене при агрегации

        Returns:
            float: критическая минимальная цена на эту дату
        """
        data = pd.DataFrame(series, columns=['by'])
        N = len(data)
        data.sort_values(by='by', inplace=True)
        data.reset_index(drop=True, inplace=True)
        percent = Series([index / N * 100 for index in range(1, N + 1)])
        data['percent'] = percent

        data_min = data[data['percent'] <= p]
        index_min_left = data_min.index[-1]
        min_left = data.iloc[index_min_left]
        if min_left.percent == p:
            left_border = min_left.by
        else:
            min_right = data.iloc[index_min_left + 1]
            left_border = min_left.by + (p - min_left.percent) * (min_right.by - min_left.by) / (
                    min_right.percent - min_left.percent)

        return left_border

    statistic = data[['by']].rolling(window=N).agg([find_border_left, ])
    statistic.columns = ['left_border']
    data['left_border'] = statistic['left_border']
    return data[data['by'] < data['left_border']]['Date']


def find_anomaly_max_bars(data: pd.DataFrame,
                          N: int = 30,
                          by: Literal['body', 'shadow'] = 'body',
                          p: float = 10.0) -> pd.DataFrame:
    """"Функция поиска аномально больших баров
    Функция принимает информацию о цене и возвращает DataFrame, когда бары были аномально большие.

    Args:
        data (pd.DataFrame): Данные о цене
        N (int): Период
        by (str): По какому параметру смотреть аномальность. Принимает значение body или shadow.
        p (float): Процент покрытия по мат распределению, чем больше тем больше аномальных баров, но точность
        поиска будет снижена.

    Returns:
        List[pd.Timestamp] - список дат, когда бары были аномально большие
    """

    data = data.copy()
    if by == 'body':
        data['by'] = data['Open'] - data['Close']
    else:
        data['by'] = data['High'] - data['Low']
    data['by'] = data['by'].abs()

    def find_border_right(series: pd.Series) -> float:
        """Вспомогательная функция поиска максимального размера бара
        Функция принимает серию о цене и находит значение больше которого бар считается аномально большим.

        Args:
            series (pd.Series): Срез данных о цене при агрегации

        Returns:
            float: критическая максимальная цена на эту дату
        """
        data = pd.DataFrame(series, columns=['by'])
        N = len(data)
        data.sort_values(by='by', inplace=True)
        data.reset_index(drop=True, inplace=True)
        percent = Series([index / N * 100 for index in range(1, N + 1)])
        data['percent'] = percent

        data_max = data[data['percent'] >= 100 - p]
        index_max_right = data_max.index[0]
        max_right = data.iloc[index_max_right]
        if max_right.percent == 100 - p:
            right_border = max_right.by
        else:
            max_left = data.iloc[index_max_right - 1]
            right_border = max_left.by + (100 - p - max_left.percent) * (max_right.by - max_left.by) / (
                    max_right.percent - max_left.percent)
        return right_border

    statistic = data[['by']].rolling(window=N).agg([find_border_right, ])
    statistic.columns = ['right_border']
    data['right_border'] = statistic['right_border']
    return data[data['by'] > data['right_border']]


def find_anomal_size_date(data: pd.DataFrame,
                          N: int = 30,
                          by: Literal['body', 'shadow'] = 'body',
                          p: float = 10.0,
                          ) -> Tuple[List[pd.Timestamp]]:
    """Функция определения аномальных баров
    Функция принимает данные о цене и возвращает даты аномально больших или маленьких баров.

    Args:
        data (pd.DataFrame): Данные о ценах
        N (int): Период
        by (str): По какому параметру смотреть аномальность. Принимает значение body или shadow.

    Returns:
        Tuple[List[pd.Timestamp]]: Даты аномально максимальных и аномально минимальных баров
    """

    date_min = find_anomaly_min_bars(data=data,
                                     N=N,
                                     p=p,
                                     by=by)
    date_max = find_anomaly_max_bars(data=data,
                                     N=N,
                                     p=p,
                                     by=by)

    return date_max, date_min


def find_levels_with_show_mother_bar(Data: Bars,
                                     percent: float = 1.0,
                                     threshold: float = 0.5) \
        -> Tuple[List[float], List[pd.Timestamp]] | None:
    """Функция поиска уровней сопротивлений и поддержки
    Функция принимает Bars с информацией о ценах и возвращает словарь с уровнями сопротивления и
    поддержки и родительские бары

    Args:
        Data (Bars): Данные о ценах
        percent (float): Максимальная разница между двумя ценами, при которой существует уровень. Указывается в
        процентах по умолчанию равен 1.0
        threshold (float): Максимальный процент при котором можно "склеивать" уровни

    Returns:
        Tuple[List[float], List[pd.Timestamp]]: Кортеж из списка уровней и родительских точек последнего уровня
    """
    data = Data.base_data
    data = data.assign(body=lambda x: x['Open'] - x['Close'])
    data = data.assign(shadow=lambda x: abs(x['High'] - x['Low']))

    df_Open = data[['Date', 'Open', 'body', 'shadow']].copy()
    df_Open.rename(columns={'Open': 'Prices'}, inplace=True)

    df_Close = data[['Date', 'Close', 'body', 'shadow']].copy()
    df_Close.rename(columns={'Close': 'Prices'}, inplace=True)

    df_Low = data[['Date', 'Low', 'body', 'shadow']].copy()
    df_Low.rename(columns={'Low': 'Prices'}, inplace=True)

    df_High = data[['Date', 'High', 'body', 'shadow']].copy()
    df_High.rename(columns={'High': 'Prices'}, inplace=True)

    # Создаём данные для создания уровней поддержки и сопротивления
    support = pd.concat([
        df_Close.assign(Source='Close'),
        df_Low.assign(Source='Low')
    ])
    resistance = pd.concat([
        df_High.assign(Source='High'),
        df_Open.assign(Source='Open'),
    ])

    # Сортируем данные от меньшего к большему
    support.sort_values(by='Prices', inplace=True)
    support = support[support['body'] > 0]

    resistance.sort_values(by='Prices', inplace=True)
    resistance = resistance[resistance['body'] > 0]

    # Обнуление индексов
    support.reset_index(drop=True, inplace=True)
    resistance.reset_index(drop=True, inplace=True)

    # Создание столбца с разницей между ближайшими числами
    diff = support['Prices'].copy().diff().abs()
    support['diff'] = diff

    diff = resistance['Prices'].copy().diff().abs()
    resistance['diff'] = diff

    # Помещение дополнительной информации о вышестоящих барах
    candle_sizes_from_above = support[['body', 'shadow', 'Date']].shift(1)
    candle_sizes_from_above.rename(columns={'body': 'body_above', 'shadow': 'shadow_above', 'Date': 'date_above'},
                                   inplace=True)
    support = pd.concat([
        support,
        candle_sizes_from_above
    ], axis=1)

    candle_sizes_from_above = resistance[['body', 'shadow', 'Date']].shift(1)
    candle_sizes_from_above.rename(columns={'body': 'body_above', 'shadow': 'shadow_above', 'Date': 'date_above'},
                                   inplace=True)
    resistance = pd.concat([
        resistance,
        candle_sizes_from_above
    ], axis=1)

    # Определение разности между ценами в %
    support = support.assign(percent_max=lambda x: x['diff'] / x['body'] * 100)
    support = support.assign(percent_min=lambda x: x['diff'] / x['shadow'] * 100)
    support = support.assign(percent_above_max=lambda x: x['diff'] / x['body_above'] * 100)
    support = support.assign(percent_above_min=lambda x: x['diff'] / x['shadow_above'] * 100)

    resistance = resistance.assign(percent_max=lambda x: x['diff'] / x['body'] * 100)
    resistance = resistance.assign(percent_min=lambda x: x['diff'] / x['shadow'] * 100)
    resistance = resistance.assign(percent_above_max=lambda x: x['diff'] / x['body_above'] * 100)
    resistance = resistance.assign(percent_above_min=lambda x: x['diff'] / x['shadow_above'] * 100)

    # Фильтрация цен, разность которых очень большая
    support = support.query(f'percent_max < {percent} | percent_min < {percent} | '
                            f'percent_above_max < {percent} | percent_above_min < {percent}')
    resistance = resistance.query(f'percent_max < {percent} | percent_min < {percent} | '
                                  f'percent_above_max < {percent} | percent_above_min < {percent}')

    # Проверка, есть вообще уровни
    N = len(support) + len(resistance)
    if N == 0:
        return None

    # Объединяем уровни
    levels = pd.concat([support, resistance])

    # Добавляем новую колонку с уровнями выше и находим разницу между двумя соседними уровнями
    diff_levels = levels['Prices'].copy().diff().abs()
    levels['diff_levels'] = diff_levels

    prices_prices_above = levels['Prices'].shift(1)
    levels['Prices_above'] = prices_prices_above

    # Находим разницу в % между уровнями и создаём словари для материнских точек и уровней
    levels = levels.assign(percent_levels=lambda x: x['diff_levels'] / x['Prices'] * 100)
    levels = levels.assign(prices_list=[[] for i in range(N)])
    levels = levels.assign(mother_date=[[] for i in range(N)])
    levels.reset_index(drop=True, inplace=True)

    # Проводим склейку между найденными уровнями, при условии, что разница между ними не превышает threshold %
    del_index = list()
    for index, row in levels.iterrows():
        row['prices_list'].append(row['Prices'])
        row['mother_date'].append(row['Date'])
        row['mother_date'].append(row['date_above'])
        if not row['percent_levels'] is None and row['percent_levels'] < threshold:
            del_index.append(index - 1)
            row['prices_list'].extend(levels.iloc[index - 1]['prices_list'])
            row['mother_date'].extend(levels.iloc[index - 1]['mother_date'])
    for i in del_index:
        levels.drop(i, inplace=True)

    levels.sort_values(by='Date', inplace=True)
    levels.reset_index(drop=True, inplace=True)
    prices = list(map(lambda x: np.mean(x), levels['prices_list'].tolist()))
    last_item = levels.iloc[-1]

    return prices, last_item['mother_date']
