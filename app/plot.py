import sys
from os.path import dirname, abspath

import pandas as pd

import mplfinance as mpf

sys.path.insert(0, dirname(abspath(__file__)))

from bars import Bars
from models import CoinsTable
from mother_data import GetLevels, GetPriorityLevels, GetHideLevels, GetUserLevels
from utils import transfer_data_to_dict


class Plot:
    dict_time_frame = {
        240: 186,
        720: 155}

    def __init__(self,
                 coin: CoinsTable,
                 time_frame: int | str = 240,
                 ):
        self.coin = coin
        self.time_frame = time_frame

    async def show_plot(self):
        Levels = await GetLevels(self.coin, self.time_frame).get_levels()
        Levels = list(pd.DataFrame(transfer_data_to_dict(Levels))['Level'])
        PriorityLevels = await GetPriorityLevels(self.coin, self.time_frame).get_levels()
        HideLevels = await GetHideLevels(self.coin, self.time_frame).get_levels()
        UserLevels = await GetUserLevels(self.coin, self.time_frame).get_levels()
        try:
            PriorityLevels = list(pd.DataFrame(transfer_data_to_dict(PriorityLevels))['Level'])
        except:
            PriorityLevels = []

        try:
            HideLevels = list(pd.DataFrame(transfer_data_to_dict(HideLevels))['Level'])
        except:
            HideLevels = []

        try:
            UserLevels = list(pd.DataFrame(transfer_data_to_dict(UserLevels))['Level'])
        except:
            UserLevels = []

        Levels = list(set(Levels) | set(PriorityLevels) | set(UserLevels) - set(HideLevels))
        Data = Bars(Coin=self.coin, time_frame=self.time_frame)
        color = ['r' if level in PriorityLevels else 'c' if level in UserLevels else 'b' for level in Levels]
        mpf.plot(await Data.get_data_for_mplfinance(), hlines=dict(hlines=Levels, colors=color, linestyle='-'), type='candle',
                 title=f'{self.coin.symbol}/{self.time_frame}')

    def show_plot_and_save(self):
        Levels = GetLevels(self.coin, self.time_frame).get_levels()
        Levels = list(pd.DataFrame(transfer_data_to_dict(Levels))['Level'])
        PriorityLevels = GetPriorityLevels(self.coin, self.time_frame).get_levels()
        HideLevels = GetHideLevels(self.coin, self.time_frame).get_levels()
        UserLevels = GetUserLevels(self.coin, self.time_frame).get_levels()
        try:
            PriorityLevels = list(pd.DataFrame(transfer_data_to_dict(PriorityLevels))['Level'])
        except:
            PriorityLevels = []

        try:
            HideLevels = list(pd.DataFrame(transfer_data_to_dict(HideLevels))['Level'])
        except:
            HideLevels = []

        try:
            UserLevels = list(pd.DataFrame(transfer_data_to_dict(UserLevels))['Level'])
        except:
            UserLevels = []

        Levels = list((set(Levels) | set(PriorityLevels) | set(UserLevels)) - set(HideLevels))
        Data = Bars(Coin=self.coin, time_frame=self.time_frame)
        color = ['r' if level in PriorityLevels else 'c' if level in UserLevels else 'b' for level in Levels]
        mpf.plot(Data.get_data_for_mplfinance(), hlines=dict(hlines=Levels, colors=color, linestyle='-'), type='candle',
                 savefig='static/chart.png', title=f'{self.coin.symbol}/{self.time_frame}')
