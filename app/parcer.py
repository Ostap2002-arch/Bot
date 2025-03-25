# Записываем список монет в базу данных
import time

import tqdm

from app.mother_data import GetCoins, GetBars, GetLevels

time1 = time.time()
# При первом запуске
GetCoins(List_coins).loading_coins()
Coins = GetCoins.get_coins()

List_time_frame = [240] * len(Coins) + [720] * len(Coins)
for coin, time_frame in tqdm(zip(Coins*2, List_time_frame)):
    GetBars(coin=coin, time_frame=time_frame).get_data()
time2 = time.time()
print(f'Время очистки бд + запроса цен - {time2 - time1}c')

time1 = time.time()
for coin, time_frame in tqdm(zip(Coins*2, List_time_frame)):
    GetLevels(coin=coin, time_frame=time_frame).find_levels()
time2 = time.time()
print(f'Время поиска всех уровней - {time2-  time1}c')