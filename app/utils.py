import json
from datetime import datetime
from typing import List, Tuple, Dict

import numpy
import numpy as np
from avg import avg_body, avg
from models import Base, LevelsTable


def check_position(func):
    def wrapper(*args, **kwargs):
        position = kwargs.get('position')
        if position != 'low' and position != 'height':
            raise Exception(f'position может принимать значения только low и height, а было дано значение {position}')
        opposition = list({'low', 'height'} - {position, })[0]
        result = func(*args, opposition=opposition, **kwargs)
        return result

    return wrapper


def initial_k_day(point1: tuple, point2: tuple):
    delta_day = datetime.strptime(point2[0], '%Y-%m-%d %H:%M:%S') - datetime.strptime(point1[0], '%Y-%m-%d %H:%M:%S')
    delta_day = delta_day.days
    delta_prise = point2[1] - point1[1]
    k = delta_prise / delta_day
    return k


def predictable_price(point1: tuple, point2: tuple, point3: tuple):
    k = initial_k_day(point1, point2)
    delta_day = datetime.strptime(point3[0], '%Y-%m-%d %H:%M:%S') - datetime.strptime(point1[0], '%Y-%m-%d %H:%M:%S')
    pred_price = point1[1] + round(k * delta_day.days, 2)
    predictable_point = (point3[0], pred_price)
    return predictable_point


def parallel_line(point1: tuple, point2: tuple, point3: tuple, point4: tuple):
    k = initial_k_day(point1, point2)
    print(point1, point2, point3, point4)
    delta_day = datetime.strptime(point4[0], '%Y-%m-%d %H:%M:%S') - datetime.strptime(point3[0], '%Y-%m-%d %H:%M:%S')
    pred_price = point3[1] + round(k * delta_day.days, 2)
    predictable_point = (point4[0], pred_price)
    return predictable_point


@check_position
def trend_check(current_series: dict, new_prise: float, position: str, **kwargs) -> bool:
    if not len(current_series[f'series_{position}']):
        if current_series['color'] == 'g' and new_prise > current_series[f'mother_{position}'][-1] or \
                current_series['color'] == 'r' and new_prise < current_series[f'mother_{position}'][-1]:
            return True
    elif current_series['color'] == 'g' and new_prise > current_series[f'series_{position}'][-1][-1][-1] or \
            current_series['color'] == 'r' and new_prise < current_series[f'series_{position}'][-1][-1][-1]:
        return True
    else:
        return False


@check_position
def add_elem_series(current_series: dict, date, prise_low, prise_height, opposition, position: str) -> None:
    prises = {'low': prise_low, 'height': prise_height}
    current_series[f'series_{position}'].append(
        [current_series[f'mother_{position}'], parallel_line(current_series[f'mother_{opposition}'],
                                                             current_series[f'series_{opposition}'][-1][-1],
                                                             current_series[f'mother_{position}'],
                                                             (str(date), prises[position]))
         ])
    current_series[f'color_{position}'].append(current_series[f'color'])


@check_position
def add_last_elem_series(current_series: dict, date, prise_low, prise_height, position: str, resistance_lines,
                         opposition, colors_resistance_lines) -> None:
    prises = {'low': prise_low, 'height': prise_height}
    if current_series[f'mother_{position}']:
        current_series[f'series_{position}'].append(
            [current_series[f'mother_{position}'], parallel_line(current_series[f'mother_{opposition}'],
                                                                 current_series[f'series_{opposition}'][-1][-1],
                                                                 current_series[f'mother_{position}'],
                                                                 (str(date), prises[position]))])
        current_series[f'color_{position}'].append('c')

    current_series[f'series_{opposition}'].append(
        [current_series[f'mother_{opposition}'], predictable_price(current_series[f'mother_{opposition}'],
                                                                   current_series[f'series_{opposition}'][-1][-1],
                                                                   (str(date), prises[opposition]))])
    current_series[f'color_{opposition}'].append('c')
    resistance_lines.extend(current_series[f'series_{position}'])
    colors_resistance_lines.extend(current_series[f'color_{position}'])
    resistance_lines.extend(current_series[f'series_{opposition}'])
    colors_resistance_lines.extend(current_series[f'color_{opposition}'])


@check_position
def create_trend(current_series, prise_low, prise_height, position: str, opposition, date):
    prises = {'low': prise_low, 'height': prise_height}
    if prises[position] > current_series[f'mother_{position}'][1]:
        current_series[f'Trend_{position}'] = 'ascending'
        current_series['color'] = 'g'
    else:
        current_series[f'Trend_{position}'] = 'pissing'
        current_series['color'] = 'r'
    current_series[f'color_{position}'].append(current_series['color'])
    current_series[f'series_{position}'].append([current_series[f'mother_{position}'], (str(date), prises[position])])


@check_position
def check_point_and_trends(current_series, prise, point, position: str, **kwargs):
    return point and (
            current_series[f'Trend_{position}'] == 'pissing' and \
            prise < current_series[f'series_{position}'][-1][-1][-1] or \
            current_series[f'Trend_{position}'] == 'ascending' and prise > current_series[f'series_{position}'][-1][-1][
                -1])


def check_point_and_add_trends(current_series, prise_low, prise_height, date, point_low, point_height):
    if check_point_and_trends(current_series=current_series, position='height', point=point_height, prise=prise_height):
        current_series['series_height'].append([current_series['mother_height'], (str(date), prise_height)])
        current_series['color_height'].append(current_series['color'])
        return True
    if check_point_and_trends(current_series=current_series, position='low', point=point_low, prise=prise_low):
        current_series['series_low'].append([current_series['mother_low'], (str(date), prise_low)])
        current_series['color_low'].append(current_series['color'])
        return True
    return False


def color_anomalies(open_prise: List[float], close_prise: List[float], N=30) -> List[str]:
    """Вычисление списка цветов для баров.
        Эта функция получает на вход список открытия и закрытия бар и период вычисления (при необходимости).
        На выход функция возвращает список строк, который характеризуют цвет в зависимости от размера бара.

        Args:
            open_prise (List[float]): Список цен открытия бар
            close_prise (List[float]): Список цен закрытия бар
            N (int): Период вычислений

        Returns:
            List[str]: Список цветов.

        Raises:
            ValueError: Пока неизвестны
        """
    color = []

    for i in range(len(close_prise)):
        body = abs(open_prise[i] - close_prise[i])
        statistics = avg_body(open_prise[:i], close_prise[:i], N=N)
        if statistics is None:
            color.append(None)
        else:
            min_body, max_body = statistics
            if body < min_body:
                color.append('r')
            elif body > max_body:
                color.append('g')
            else:
                color.append(None)

    return color


def avg_lines(prices: List[float], date: List[datetime], N: Dict[int, str]) -> Dict[str, list]:
    """Функция вычисления средней скользящей.
    Эта функция получает на вход список цен, даты и периоды вычисления (при необходимости).
    На выход функция возвращает список списков кортежей, содержащих дату и среднее значение.

    Args:
        prices (List[float]): Список цен
        date (List[datetime]): Даты
        N (Dict[int, str]): Словарь с данными о периодах, где key - номер периода (int), values - цвет периода (str). Пример {30: 'r', 20: 'y'}

    Returns:
        List[List[Tuple[str, float]]]: Список средних скользящих.

    Raises:
        ValueError: Пока неизвестны
        """

    result = []
    avg_list = []

    for period in N.keys():
        for i in range(len(prices)):
            avg_price = avg(prices[:i], N=period)
            if avg_price is None:
                continue
            else:
                if avg_price:
                    new_point_30 = (date[i].strftime('%Y-%m-%d'), avg_price)
                    avg_list.append(new_point_30)
        result.append(avg_list)
        avg_list = []

    return dict(alines=result, colors=list(N.values()))


def transfer_data_to_dict(data: List[Base]) -> dict:
    """Функция преобразование списка информации о барах в словарь.
    Эта функция на вход получает список экземпляров класса из бд и возвращает словарь, где: key - название атрибутов
    клаcса,
    value - список значений каждого экземпляра по данному атрибуту

    Args:
        data (List[Base]): Список экземпляров класса из бд

    Returns:
        dict: Словарь с преобразованной информацией. Пример {'id': [1, 2, ...], 'date': [datetime(2022-10-23),
        datetime(2023-04-15), ...], ...}
    """
    keys = [atr for atr in data[0].__dict__ if not atr.startswith('_')]
    result = {key: [] for key in keys}
    for elem in data:
        for key, value in result.items():
            value.append(elem.__dict__[key])
    return result


def print_data_dict(data: Dict[str, list]) -> None:
    """Функция печати данных в формате json
    Функция получает на вход словарь с данными и выводит на печать в удобном виде.
    При получении объектов не подлежащих сериализация, выводится их стороковое представление, кроме типа datetime. Для
    знакомства обработки объектов не подлежащих сериализация обратитесь к функции default_print.

    Args:
        data (Dict[str, list]): Словарь с данными

    Returns:
        None
    """
    print(json.dumps((data), indent=4, default=default_print))


def default_print(obj) -> str:
    """Функция представление объекта, при невозможности сериализация его в JSON.
    По-умолчанию все объекты
    будут представлены в строковом виде, кроме datetime.
    Тип данных datetime будет отображен в формате '%Y-%m-%d %H:%M:%S' """

    if isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    else:
        return str(obj)


def find_max_and_min(data: Dict[str, list]) -> Tuple[float, float]:
    """Функция нахождения максимального и минимального числа в данных
    Функция получает на вход словарь с данными и возвращает максимальное и минимальное число из представленных
    данных.

    Args:
        data (Dict[str, list]): Словарь с данными

    Returns:
        Tuple[float, float]: максимальное и минимальное число в данных

    """
    data_new = list()
    for value in data.values():
        if isinstance(value[0], float):
            data_new.extend(value)

    return max(data_new), min(data_new)


# NumpyArray = TypeVar('NumpyArray', bound=np.array)


def find_body(prices_open: List[float], prices_close, N=30):
    """Вычисление размеров бар .
        Эта функция получает на вход список открытия и закрытия бар и период вычисления (при необходимости).
        На выход получаем список размеров бар за N дней (по default = 30).

        Args:
            prices_open (List[float]): Список цен открытия бар
            prices_close (List[float]): Список цен закрытия бар
            N (int): Период вычислений

        Returns:
            NumpyArray[float]: Массив numpy с np.float

        """
    np.set_printoptions(precision=5, suppress=True)
    prices_open = np.array(prices_open[-1:-N:-1])
    prices_close = np.array(prices_close[-1:-N:-1])
    body = prices_open - prices_close
    return body


def find_quantile(ListLevel: LevelsTable, target: float):
    for level in ListLevel:
        if level.Level == target:
            confirmation = len(json.loads(level.Mother_date))
    ListConfirmations = list(map(lambda x: len(json.loads(x.Mother_date)), ListLevel))
    Q1 = numpy.quantile(ListConfirmations, 0.25)
    median = numpy.quantile(ListConfirmations, 0.5)
    Q3 = numpy.quantile(ListConfirmations, 0.75)
    if confirmation < Q1:
        return 'Q1', confirmation
    elif confirmation < median:
        return 'Q2', confirmation
    elif confirmation < Q3:
        return 'Q3', confirmation
    else:
        return 'Q4', confirmation
