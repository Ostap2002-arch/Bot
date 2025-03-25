from typing import List, Optional, Union
import numpy as np
from math import sqrt


def avg(prices: List[float], N=30) -> Optional[float]:
    """Вычисление средней скользящей.

    Эта функция получает на вход список float и период скользящей (при необходимости).
    На выход получаем среднее скользящее или None, eсли данных не достаточно для вычисления средней скользящей.

    Args:
        prices (List[float]): Список цен
        N (int): Период скользящей

    Returns:
        Optional[float]: Среднее скользящее

    Raises:
        ValueError: Пока неизвестны
    """
    np.set_printoptions(precision=5, suppress=True)
    if len(prices) < N:
        return None
    else:
        return float(np.array(prices[-1:-N:-1]).mean())


def avg_body(prices_open: List[float], prices_close, N=30) -> Union[None, tuple]:
    """Вычисление среднего размера бар.
    Эта функция получает на вход список открытия и закрытия бар и период вычисления (при необходимости).
    На выход получаем кортеж (минимального и максимального размера бара в мат. сатистике) или None,
    eсли данных не достаточно для вычисления среднего размера.

    Args:
        prices_open (List[float]): Список цен открытия бар
        prices_close (List[float]): Список цен закрытия бар
        N (int): Период вычислений

    Returns:
        Union[None, (float, float)]:: мин и макс размеры тела бар.

    Raises:
        ValueError: Пока неизвестны
    """
    if len(prices_close) < N:
        return None
    else:
        np.set_printoptions(precision=5, suppress=True)
        prices_open = np.array(prices_open[-1:-N:-1])
        prices_close = np.array(prices_close[-1:-N:-1])
        body = prices_open - prices_close
        body = np.abs(body)
        mean = body.mean()
        variance = sqrt(np.var(body))
        min_body, max_body = mean-variance, mean + variance
        MAX = mean + variance
        count = 0
        while min_body < 0 or max_body < 0:
            if min_body < 0:
                body = body[body < max_body]
            else:
                body = body[body > min_body]
            mean = body.mean()
            variance = sqrt(np.var(body))
            min_body, max_body = mean - variance, mean + variance
            count += 1

        return min_body, MAX










