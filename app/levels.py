import sys
from os.path import dirname, abspath
from typing import List, Tuple, Literal

import pandas as pd
import numpy as np
from datetime import timedelta

sys.path.insert(0, dirname(abspath(__file__)))

from statistics import find_local_id_min, find_local_id_max, find_anomaly_max_bars
from bars import Bars


def finding_neighbor_in_df(df: pd.DataFrame, time_frame: timedelta = timedelta(days=1)) -> None:
    """Функция поиска соседей в DataFrame
    Функция принимает на вход DataFrame с колонками ['Date', 'Prices', 'body', 'shadow'] упорядочивает и добавляет в
    этот DataFrame новые колонки
    ['Date_neighbor', 'Prices_neighbor']

    Args:
        df (pd.DataFrame): информация о цене в формате ['Date', 'Prices', 'body', 'shadow'],
        time_frame (datetime.timedelta): Тайм-фрейм в котором мы работаем

    Returns:
        None
    """
    df.sort_values(by='Prices', inplace=True)
    df.reset_index(drop=True, inplace=True)

    def finding_neighbor(slice_df: np.array) -> np.array:
        """Функция поиска соседа для бара (принцип поиска - снизу вверх)
        Функция принимает на вход двумерный массив numpy и возвращает одномерный массив - информацию о соседе последнего бара

        Args:
            slice_df (np.array): данные о цене в виде двумерного массива (slice_df.shape -> (1...9, 2)) - где первая колонка Date (в наносекундах),
            вторая колонка Prices
        Returns:
            (np.array): данные о соседе в виде одномерного массива (slice_df.shape -> (2)) - где первая колонка Date (в наносекундах),
            вторая колонка Prices
        """
        # Делаем запасной DataFrame
        old_df = slice_df.copy()

        # Получаем информацию о главном элементе
        main_elem = slice_df[-1]
        main_price = main_elem[1]
        main_date = main_elem[0]

        # Добавляем разницу между элементами, index Diff == 2
        Prices = slice_df[:, 1]
        Diff = np.abs(Prices - main_price)
        Diff = Diff.reshape(-1, 1)
        slice_df = np.hstack((slice_df, Diff))

        # Создаём фильтр дат для массива List[bool]

        yesterday = main_date - delta_frame
        tomorrow = main_date + delta_frame

        Dates = slice_df[:, 0]
        Filter_dates_1 = Dates != main_date
        Filter_dates_2 = Dates != yesterday
        Filter_dates_3 = Dates != tomorrow

        Filter_dates = np.logical_and(np.logical_and(Filter_dates_1, Filter_dates_2), Filter_dates_3)

        # Применяем созданный фильтр по датам
        slice_df = slice_df[Filter_dates]

        # если DataFrame пустой, то возращаем гланый DataFrame (теоретически это работает для самого первого
        # элемента)
        if slice_df.size == 0:
            # не самое правильно решение, в дальше это фиксится в DataFrame
            # Сразу заполнить это None нельзя, так как работаем в numba
            return old_df[-1, 0:2]

        # Находим минимальную разницу цен
        Diff = slice_df[:, 2]
        min_Diff = np.min(slice_df[:, 2])

        # Создаём фильтр для поиска пары, которая образует минимальную разницу
        Filter_diff = Diff == min_Diff
        # Применяем созданный фильтр
        neighbor = slice_df[Filter_diff][-1]

        # Возвращаем ближайшую цену и соответствующею ей дату
        neighbor = neighbor.reshape(-1)
        return neighbor[0:2]

    # создаём расходные данные, на основе которых будем искать соседей
    consumption_data = df[['Date', 'Prices']].copy()
    consumption_data['Date'] = consumption_data['Date'].astype('int64')

    # Это тайм-фрейм в наносекундах, для поиска соседей
    delta_frame = np.float64(time_frame.total_seconds() * 1_000_000_000)

    neighbors_df = consumption_data[['Date', 'Prices']].rolling(window=9,
                                                                method='table',
                                                                min_periods=2).apply(finding_neighbor,
                                                                                     engine="numba",
                                                                                     raw=True)

    neighbors_df['Date'] = pd.to_datetime(neighbors_df['Date'])
    df[['Date_neighbors', 'Prices_neighbors']] = neighbors_df[['Date', 'Prices']]

    # Фиксим баг из finding_neighbor
    df.loc[(df['Date'] == df['Date_neighbors']) & (df['Prices'] == df['Prices_neighbors']), ['Date_neighbors',
                                                                                             'Prices_neighbors']] = None


def add_info_df(df: pd.DataFrame) -> None:
    """Функция добавления информации в DataFrame
    Функция принимает на вход DataFrame с колонками ['Date', 'Prices', 'body', 'shadow', 'Date_neighbors', 'Prices_neighbors']
    упорядочивает и добавляет в этот DataFrame новые колонки ['Diff', 'body_neighbors', 'shadow_neighbors', '%_min', '%_max', '%_min_neighbors', '%_max_neighbors',
    'Level', 'mother_date', 'mother_prices']
    Использование этой функции к DataFrame предполагает, что к этому же DataFrame было применена функция finding_neighbor_in_df.
    Args:
        df (pd.DataFrame): информация о цене в формате ['Date', 'Prices', 'body', 'shadow', 'Date_neighbors', 'Prices_neighbors']

    Returns:
        None
    """
    # Добавление mother_date и mother_prices
    df['mother_date'] = df.apply(lambda row: [row['Date'], row['Date_neighbors']], axis=1)
    df['mother_prices'] = df.apply(lambda row: [row['Prices'], row['Prices_neighbors']], axis=1)

    # Находим разницу между ценами
    df['Diff'] = df['Prices'] - df['Prices_neighbors']

    # Добавление уровни (средне арифметический метод)
    df['Level'] = df.apply(lambda row: np.mean(row['mother_prices']), axis=1)

    # Добавляем колонку body_neighbors и shadow_neighbors
    def find_body_neighbors(row):
        """Функция поиска тела соседнего бара
        Функция принимает на вход строку из DataFrame и находит размер соседнего бара для этой строки

        Args:
            row (pd.Series): Строка из DataFrame

        """
        neighbors = df[(df['Date'] == row['Date_neighbors']) & (df['Prices'] == row['Prices_neighbors'])]
        try:
            return abs(neighbors['body'].tolist()[0])  # Или любой другой столбец, который вам нужен
        except:
            return None

    # Используем apply для применения функции к каждой строке
    df['body_neighbors'] = df.apply(find_body_neighbors, axis=1)

    def find_shadow_neighbors(row):
        """Функция поиска тела соседнего бара
        Функция принимает на вход строку из DataFrame и находит размер соседнего бара для этой строки

        Args:
            row (pd.Series): Строка из DataFrame

        """
        neighbors = df[(df['Date'] == row['Date_neighbors']) & (df['Prices'] == row['Prices_neighbors'])]
        try:
            return abs(neighbors['shadow'].tolist()[0])  # Или любой другой столбец, который вам нужен
        except:
            return None

    # Используем apply для применения функции к каждой строке
    df['shadow_neighbors'] = df.apply(find_shadow_neighbors, axis=1)

    # Добавляем проценты
    df['%_min'] = df['Diff'] / df['shadow'] * 100
    df['%_max'] = df['Diff'] / df['body'] * 100
    df['%_min_neighbors'] = df['Diff'] / df['shadow_neighbors'] * 100
    df['%_max_neighbors'] = df['Diff'] / df['body_neighbors'] * 100


def preparation_for_unification(list_df: List[pd.DataFrame],
                                percent: float = 1.0,
                                by: Literal['body', 'shadow'] = 'body'):
    """Функция подготавливающая передеанныe DataFrames к процессу слияния уроней и их объединению
    Использование данной функции предполагается после функции add_info_df

    Args:
        list_df (List[pd.DataFrame]): список DataFrame для подготовки и объединению
        percent (float): процент при котором уровень считается ликвидным (чем меньше процент - темь меньше уровней, но их качесво выше)
        by (str): параметр который отвечает за условие выбора процента для фильтрации 'body' или 'shadow',
    Returns:
        pd.DataFrame: Датафрейм готовый к объединению
    """
    # Какие колонки используем для фильтрации (в зависимости от by)
    columns_filter = ['%_max', '%_max_neighbors'] if by == 'body' else ['%_min', '%_min_neighbors']

    # Фильтруем каждый переданный фильтр
    for index, df in enumerate(list_df):
        list_df[index] = df[(df[columns_filter[0]] < percent) & (df[columns_filter[1]] < percent)]

    # Список столбцов, которые нужно оставить
    columns_to_keep = ['Level', 'mother_date', 'mother_prices', 'Source']

    # Удаление всех столбцов, кроме указанных
    for index, df in enumerate(list_df):
        list_df[index] = df.drop(columns=[col for col in df.columns if col not in columns_to_keep])

    result_df = pd.concat(list_df)

    # Сортируем и обновляем индексы
    result_df.sort_values(by='Level', inplace=True)
    result_df.reset_index(drop=True, inplace=True)

    # #Добавляем разницу между соседними уровнями
    # Diff = result_df['Level'].copy().diff().abs()
    # result_df['Diff'] = Diff
    # result_df['%_Diff'] = result_df['Diff'] / result_df['Level'] * 100
    return result_df


def unification(Levels: pd.DataFrame, c: float = 0.5) -> pd.DataFrame:
    """Функция слияния уровней
    Функция принимает DataFrame и склеивает близкие уровни

    Args:
        Levels (pd.DataFrame): Список уровней, предполагая, что в него передаётся DataFrame из функции preparation_for_unification
        c (float): Процент при котором два уровня сливаются
    """
    copy_levels = Levels.copy()
    copy_levels.sort_values(by='Level', inplace=True)
    copy_levels.reset_index(drop=True, inplace=True)

    def condition(level1: pd.Series, level2: pd.Series):
        if abs(level1['Level'] - level2['Level']) / level1['Level'] * 100 < c:
            return True
        else:
            return False

    def combine_to_list(obj1, obj2):
        result = []

        # Проверяем первый объект
        if isinstance(obj1, str):
            result.append(obj1)
        elif isinstance(obj1, list):
            result.extend(obj1)

        # Проверяем второй объект
        if isinstance(obj2, str):
            result.append(obj2)
        elif isinstance(obj2, list):
            result.extend(obj2)

        return result

    i = 0
    while i < len(copy_levels) - 1:

        # Проверяем условие между текущим элементом и следующим
        if condition(copy_levels.iloc[i], copy_levels.iloc[i + 1]):
            # Заменяем текущий элемент на новый
            copy_levels.at[i, 'mother_date'] = copy_levels.at[i, 'mother_date'] + copy_levels.at[
                i + 1, 'mother_date']
            copy_levels.at[i, 'mother_prices'] = copy_levels.at[i, 'mother_prices'] + copy_levels.at[
                i + 1, 'mother_prices']
            copy_levels.at[i, 'Source'] = copy_levels.at[i, 'Source'] + copy_levels.loc[i + 1, 'Source']
            copy_levels.at[i, 'Level'] = np.mean(copy_levels.at[i, 'mother_prices'])
            # Удаляем следующий элемент
            copy_levels.drop(index=(i + 1), inplace=True)
            # Сбрасываем индекс, чтобы проверить снова с текущим элементом
            i = max(i - 1, 0)
            copy_levels.reset_index(drop=True, inplace=True)
        else:
            i += 1

    return copy_levels


def search_for_nearest_prices(Data: pd.DataFrame,
                              percent: float = 1.0,
                              c: float = 0.5,
                              by: Literal['body', 'shadow', 'all'] = 'all',
                              time_frame: timedelta = timedelta(days=1), ) -> pd.DataFrame:
    """Функция поиска уровня по методу ближайших цен

    :param Data: Информация о ценах
    :param percent: Процент, при котором две цены считаются близкими
    :param c: Процент при котором обнаруженные уровни можно объединить
    :param by: Какие цены мы рассматриваем
    :param time_frame: Тайм фрейм
    :return: pd.DataFrame c columns = ['Level', 'mother_date', 'mother_prices', 'Source']
    """

    data = Data.copy()
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

    # Соединяем в 4 DataFrame Все данные о минимальных значениях и максимальных

    if by == 'body':
        list_positive_bars_support = [
            df_Open.assign(Source='Open'),
        ]
        list_negative_bars_support = [
            df_Close.assign(Source='Close'),
        ]
        list_positive_bars_resist = [
            df_Close.assign(Source='Close'),
        ]
        list_negative_bars_resist = [
            df_Open.assign(Source='Open'),
        ]

    if by == 'shadow':
        list_positive_bars_support = [
            df_Low.assign(Source='Low')
        ]
        list_negative_bars_support = [
            df_Low.assign(Source='Low')
        ]
        list_positive_bars_resist = [
            df_High.assign(Source='High'),
        ]
        list_negative_bars_resist = [
            df_High.assign(Source='High'),
        ]
    else:
        list_positive_bars_support = [
            df_Open.assign(Source='Open'),
            df_Low.assign(Source='Low')
        ]
        list_negative_bars_support = [
            df_Close.assign(Source='Close'),
            df_Low.assign(Source='Low')
        ]
        list_positive_bars_resist = [
            df_High.assign(Source='High'),
            df_Close.assign(Source='Close'),
        ]
        list_negative_bars_resist = [
            df_High.assign(Source='High'),
            df_Open.assign(Source='Open'),
        ]

    positive_bars_support = pd.concat(list_positive_bars_support)
    positive_bars_support = positive_bars_support[positive_bars_support['body'] < 0]

    negative_bars_support = pd.concat(list_negative_bars_support)
    negative_bars_support = negative_bars_support[negative_bars_support['body'] > 0]

    positive_bars_resist = pd.concat(list_positive_bars_resist)
    positive_bars_resist = positive_bars_resist[positive_bars_resist['body'] < 0]

    negative_bars_resist = pd.concat(list_negative_bars_resist)
    negative_bars_resist = negative_bars_resist[negative_bars_resist['body'] > 0]

    support_1 = pd.concat([positive_bars_support, negative_bars_support])
    resist_1 = pd.concat([positive_bars_resist, negative_bars_resist])

    # Для отладки работы
    # support_1['Source'] = support_1['Source'].apply(lambda x: ['Support_1'])
    # resist_1['Source'] = support_1['Source'].apply(lambda x: ['Resist_1'])

    # Поиск соседей для баров
    finding_neighbor_in_df(support_1, time_frame=time_frame)
    finding_neighbor_in_df(resist_1, time_frame=time_frame)

    # Добавление информации об уровнях
    add_info_df(support_1)
    add_info_df(resist_1)

    levels_1 = preparation_for_unification([support_1, resist_1], percent=percent)
    levels_1['Source'] = levels_1['Source'].apply(lambda x: ['Levels_2'])
    levels_1 = unification(levels_1, c=c)
    return levels_1


def search_by_relative_extremes(Data: pd.DataFrame,
                                percent: float = 2.0,
                                c: float = 0.5,
                                window: int = 7,
                                by: Literal['body', 'shadow', 'all'] = 'all',
                                time_frame: timedelta = timedelta(days=1), ) -> pd.DataFrame:
    """Функция поиска уровня по методу ближайших локальных экстремумов

    :param Data: Информация о ценах
    :param percent: Процент, при котором две цены считаются близкими
    :param c: Процент при котором обнаруженные уровни можно объединить
    :param window: Окно поиска относительного экстремума, рекомендуется брать не ниже 5
    :param by: Какие цены мы рассматриваем
    :param time_frame: Тайм фрейм
    :return: pd.DataFrame c columns = ['Level', 'mother_date', 'mother_prices', 'Source']
    """

    data = Data.copy()
    data = data.assign(body=lambda x: x['Open'] - x['Close'])
    data = data.assign(shadow=lambda x: abs(x['High'] - x['Low']))

    if by == 'shadow':
        id_min_shadow = find_local_id_min(data=data, window=window, by='shadow')
        id_max_shadow = find_local_id_max(data=data, window=window, by='shadow')
    elif by == 'body':
        id_min_body = find_local_id_min(data=data, window=window, by='body')
        id_max_body = find_local_id_max(data=data, window=window, by='body')
    else:
        id_min_shadow = find_local_id_min(data=data, window=window, by='shadow')
        id_min_body = find_local_id_min(data=data, window=window, by='body')
        id_max_shadow = find_local_id_max(data=data, window=window, by='shadow')
        id_max_body = find_local_id_max(data=data, window=window, by='body')

    copy_data = data.copy()

    if by == 'shadow':
        df_Low = copy_data[['id', 'Date', 'Low', 'body', 'shadow']].copy()
        df_Low.rename(columns={'Low': 'Prices'}, inplace=True)
        df_Low = df_Low[df_Low['id'].isin(id_min_shadow)][['Date', 'Prices', 'body', 'shadow']]

        df_High = copy_data[['id', 'Date', 'High', 'body', 'shadow']].copy()
        df_High.rename(columns={'High': 'Prices'}, inplace=True)
        df_High = df_High[df_High['id'].isin(id_max_shadow)][['Date', 'Prices', 'body', 'shadow']]
        # Соединяем в 2 DataFrame Все данные о цена

        levels_2 = pd.concat([
            df_Low,
            df_High
        ])
    elif by == 'body':
        copy_data['top_price'] = copy_data[['Open', 'Close']].max(axis=1)
        copy_data['lower_price'] = copy_data[['Open', 'Close']].min(axis=1)

        df_Top = copy_data[['id', 'Date', 'top_price', 'body', 'shadow']].copy()
        df_Top.rename(columns={'top_price': 'Prices'}, inplace=True)
        df_Top = df_Top[df_Top['id'].isin(id_max_body)][['Date', 'Prices', 'body', 'shadow']]

        df_Lower_body = copy_data[['id', 'Date', 'lower_price', 'body', 'shadow']].copy()
        df_Lower_body.rename(columns={'lower_price': 'Prices'}, inplace=True)
        df_Lower_body = df_Lower_body[df_Lower_body['id'].isin(id_min_body)][['Date', 'Prices', 'body', 'shadow']]

        levels_2 = pd.concat([
            df_Lower_body,
            df_Top
        ])
    else:
        copy_data['top_price'] = copy_data[['Open', 'Close']].max(axis=1)
        copy_data['lower_price'] = copy_data[['Open', 'Close']].min(axis=1)

        df_Top = copy_data[['id', 'Date', 'top_price', 'body', 'shadow']].copy()
        df_Top.rename(columns={'top_price': 'Prices'}, inplace=True)
        df_Top = df_Top[df_Top['id'].isin(id_max_body)][['Date', 'Prices', 'body', 'shadow']]

        df_Lower_body = copy_data[['id', 'Date', 'lower_price', 'body', 'shadow']].copy()
        df_Lower_body.rename(columns={'lower_price': 'Prices'}, inplace=True)
        df_Lower_body = df_Lower_body[df_Lower_body['id'].isin(id_min_body)][['Date', 'Prices', 'body', 'shadow']]

        df_Low = copy_data[['id', 'Date', 'Low', 'body', 'shadow']].copy()
        df_Low.rename(columns={'Low': 'Prices'}, inplace=True)
        df_Low = df_Low[df_Low['id'].isin(id_min_shadow)][['Date', 'Prices', 'body', 'shadow']]

        df_High = copy_data[['id', 'Date', 'High', 'body', 'shadow']].copy()
        df_High.rename(columns={'High': 'Prices'}, inplace=True)
        df_High = df_High[df_High['id'].isin(id_max_shadow)][['Date', 'Prices', 'body', 'shadow']]

        levels_2 = pd.concat([
            df_Lower_body,
            df_Low,
            df_Top,
            df_High
        ])

    # Поиск соседей для баров
    finding_neighbor_in_df(levels_2, time_frame=time_frame)

    # Добавление информации об уровнях
    add_info_df(levels_2)

    levels_2['Source'] = [['Levels_2']] * len(levels_2)

    # И фильтруем их в зависимости от разности образующих баров
    levels_2 = preparation_for_unification(list_df=[levels_2], percent=percent, by='shadow')
    levels_2 = unification(levels_2, c=c)
    return levels_2


def search_by_abnormal_bars(Data: pd.DataFrame,
                            percent: float = 2.0,
                            c: float = 0.5,
                            window: int = 5,
                            by: Literal['body', 'shadow'] = 'shadow',
                            N: int = 10,
                            p: float = 20.0,
                            time_frame: timedelta = timedelta(days=1), ) -> pd.DataFrame:
    """Функция поиска уровня по методу остановки аномально больших баров

    :param Data: Информация о ценах
    :param percent: Процент, при котором две цены считаются близкими
    :param c: Процент при котором обнаруженные уровни можно объединить
    :param window: Окно поиска относительного экстремума, рекомендуется брать не ниже 5
    :param by: По чему мы смотрим на аномальность (тень или тело)
    :param time_frame: Тайм фрейм
    :param N: Какой срез данных мы рассматриваем для поиска аномального бара
    :param p: Какой процент от общего количества мы принимаем аномально большими
    :return: pd.DataFrame c columns = ['Level', 'mother_date', 'mother_prices', 'Source']
    """
    data = Data.copy()

    id_anomaly_size = set(find_anomaly_max_bars(data=data, by=by, N=N, p=p)['id'])

    if by == 'body':
        id_min_body = find_local_id_min(data=data, window=window, by='body')
        id_max_body = find_local_id_max(data=data, window=window, by='body')
        id_min_body = set(id_min_body)
        id_max_body = set(id_max_body)
    else:
        id_max_shadow = find_local_id_max(data=data, window=window, by='shadow')
        id_min_shadow = find_local_id_min(data=data, window=window, by='shadow')
        id_min_shadow = set(id_min_shadow)
        id_max_shadow = set(id_max_shadow)

    copy_data = data.copy()
    copy_data = copy_data.assign(body=lambda x: x['Open'] - x['Close'])
    copy_data = copy_data.assign(shadow=lambda x: abs(x['High'] - x['Low']))

    if by == 'body':
        copy_data['top_price'] = copy_data[['Open', 'Close']].max(axis=1)
        copy_data['lower_price'] = copy_data[['Open', 'Close']].min(axis=1)

        df_Top = copy_data[['id', 'Date', 'top_price', 'body', 'shadow']].copy()
        df_Top.rename(columns={'top_price': 'Prices'}, inplace=True)
        df_Top = df_Top[df_Top['id'].isin(id_max_body & id_anomaly_size)][['Date', 'Prices', 'body', 'shadow']]

        df_Lower_body = copy_data[['id', 'Date', 'lower_price', 'body', 'shadow']].copy()
        df_Lower_body.rename(columns={'lower_price': 'Prices'}, inplace=True)
        df_Lower_body = df_Lower_body[df_Lower_body['id'].isin(id_min_body & id_anomaly_size)][
            ['Date', 'Prices', 'body', 'shadow']]
        levels_3 = pd.concat([
            df_Lower_body,
            df_Top
        ])

    else:
        df_Low = copy_data[['id', 'Date', 'Low', 'body', 'shadow']].copy()
        df_Low.rename(columns={'Low': 'Prices'}, inplace=True)
        df_Low = df_Low[df_Low['id'].isin(id_min_shadow & id_anomaly_size)][['Date', 'Prices', 'body', 'shadow']]

        df_High = copy_data[['id', 'Date', 'High', 'body', 'shadow']].copy()
        df_High.rename(columns={'High': 'Prices'}, inplace=True)
        df_High = df_High[df_High['id'].isin(id_max_shadow & id_anomaly_size)][['Date', 'Prices', 'body', 'shadow']]

        levels_3 = pd.concat([
            df_Low,
            df_High
        ])

    levels_3.rename(columns={'Date': 'mother_date',
                             'Prices': 'Level'}, inplace=True)
    levels_3['mother_prices'] = levels_3[['Level']].apply(lambda x: [x['Level']], axis=1)
    levels_3['mother_date'] = levels_3[['mother_date']].apply(lambda x: [x['mother_date']], axis=1)
    levels_3['Source'] = 'Level_3'
    levels_3['Source'] = levels_3[['Source']].apply(lambda x: ['Level_3'], axis=1)
    levels_3.drop(['body', 'shadow'], axis=1, inplace=True)
    levels_3.reset_index(drop=True, inplace=True)
    level_3 = unification(levels_3, c=c)

    return level_3


def find_levels(Data: pd.DataFrame,
                method_tuple: Tuple[Literal['Level_1', 'Level_2', 'Level_3']] = ('Level_2', 'Level_3'),
                window_for_method: Tuple[int] = (5, 5),
                by: Tuple[Literal['body', 'shadow', 'all']] = ('all', 'shadow'),
                percent: Tuple[float] = (2.0, 2.0),
                c: Tuple[float] = (0.5, 0.5),
                time_frame: Tuple[timedelta] = (timedelta(days=1), timedelta(days=1)),
                N: Tuple[int] = (10, 10),
                p: Tuple[float] = (20.0, 20.0),
                C: float = 1.0
                ) -> pd.DataFrame:
    """Функция поиска уровней
    Методы поиска:  Level_1 - поиск по ближайшим значениям
                    Level_2 - поиск по относительным экстремумам
                    Level_3 - поиск по аномально большим барах
    ! Длина переданных кортежей должна быть одинакова или они должны отсутствовать

    :param Data: Информация о ценах
    :param method_tuple: Выбор методов для поиска уровней
    :param window_for_method: Выбор скользящего окна (применим к Level_2 и Level_3)
    :param by: по какой части тела мы ищем уровни 'body', 'shadow', 'all' (!В Level_3 есть только 'body' и 'shadow')
    :param percent: Процент, при котором две цены считаются близкими (применим к Level_1 и Level_2)
    :param c: Процент при котором обнаруженные уровни можно объединить
    :param time_frame: Тайм фрейм
    :param N: Какой срез данных мы рассматриваем для поиска аномального бара (применим к Level_3)
    :param p: Какой процент от общего количества мы принимаем аномально большими (применим к Level_3)
    :param C: Процент при котором финальные уровни склеиваются
    :return: pd.DataFrame c columns = ['Level', 'mother_date', 'mother_prices', 'Source']
    """
    Levels = list()

    for i, method in enumerate(method_tuple):
        if method == 'Level_1':
            new_levels = search_for_nearest_prices(Data=Data,
                                                   percent=percent[i],
                                                   c=c[i],
                                                   by=by[i],
                                                   time_frame=time_frame[i],
                                                   )
        elif method == 'Level_2':
            new_levels = search_by_relative_extremes(Data=Data,
                                                     percent=percent[i],
                                                     c=c[i],
                                                     window=window_for_method[i],
                                                     by=by[i],
                                                     time_frame=time_frame[i],
                                                     )
        else:
            new_levels = search_by_abnormal_bars(Data=Data,
                                                 percent=percent[i],
                                                 c=c[i],
                                                 window=window_for_method[i],
                                                 by=by[i],
                                                 time_frame=time_frame[i],
                                                 N=N[i],
                                                 p=p[i]
                                                 )
        Levels.append(new_levels)

    Levels = pd.concat(Levels)
    Levels = unification(Levels, c=C)

    return Levels
