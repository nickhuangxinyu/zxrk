import pandas as pd
pd.set_option('display.max_columns', 1000)
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import sys
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import confusion_matrix
from ml_util import *
from sklearn.model_selection import cross_val_score

cols = ['UpdateTime', 'UpdateMillisec', 'InstrumentID', 'Volume', 'LastPrice', 'OpenInterest', 'PreSettlementPrice',
        'PreClosePrice', 'PreOpenInterest', 'OpenPrice', 'HighestPrice', 'LowestPrice', 'ClosePrice', 'SettlementPrice',
        'UpperLimitPrice', 'LowerLimitPrice', 'BidPrice1', 'AskPrice1', 'BidVolume1', 'AskVolume1', 'Turnover', 'TradingDay', 'LocalTime']

def MACD(mp, window_size):
    ema_long = mp.ewm(span=2*window_size, adjust=False).mean()
    ema_short = mp.ewm(span=window_size, adjust=False).mean()
    diff = ema_long - ema_short
    dea = diff.ewm(span=math.ceil(0.8*window_size), adjust=False).mean()
    return 2*(diff - dea)

contract_size = 300
time_set = [1,3,5,10,20,30,50, 100, 200]

class DataLoader:
  def __init__(self, path):
    self.LoadByPath(path)

  def LoadByPath(self, data_path):
    # data handling
    x_col = []
    y_col = 'yield_diff'
    df_list = []
    for f in os.listdir(data_path):
        if 'tick' not in f:
            continue
        df = pd.read_csv(data_path+f, header=0, names = cols, dtype={'UpdateTime':str, 'UpdateMillisec':str})
        df['mp'] = (df['BidPrice1'] + df['AskPrice1'])/2
        df['wp'] = (df['BidPrice1'] * df['BidVolume1'] + df['AskPrice1'] * df['AskVolume1']) / (df['BidVolume1'] + df['AskVolume1'])
        df['rwp'] = (df['BidPrice1'] * df['AskVolume1'] + df['AskPrice1'] * df['BidVolume1']) / (df['BidVolume1'] + df['AskVolume1'])
        df['atp'] = df['Turnover'].diff(1) / df['Volume'].diff(1) / contract_size
        df['wp-mp'] = df['wp'] - df['mp']
        df['rwp-mp'] = df['rwp'] - df['mp']
        df['atp-mp'] = df['atp'] - df['mp']
        df['r10'] = df['mp'].diff(-10) / df['mp']
        df = df[df['UpdateTime'] > '09']
        df['time_str'] = df['UpdateTime'] + df['UpdateMillisec']
        df_x = df[df['InstrumentID'] == 'IF2009']
        df_y = df[df['InstrumentID'] == 'IF2008']
        df = pd.merge(df_x, df_y, on=['time_str'], how='inner')
        df['mid_delta'] = df['mp_x'] - df['mp_y']
        for t in time_set:
            for s in ['x', 'y']:
                df['mid_diff_%s_%d' % (s, t)] = df['mp_%s' % (s)].diff(t)
                x_col.append('mid_diff_%s_%d' % (s, t))
                df['openinterest_diff_%s_%d' % (s, t)] = df['OpenInterest_%s' % (s)].diff(t)
                x_col.append('openinterest_diff_%s_%d' % (s, t))
                df['mid_delta_diff_%d' % (t)] = df['mid_delta'].diff(t)
                x_col.append('mid_delta_diff_%d' % (t))
                df['wp-mp_diff_%s_%d' % (s, t)] = df['wp-mp_%s' %(s)].diff(t)
                x_col.append('wp-mp_diff_%s_%d' % (s, t))
                df['atp-mp_diff_%s_%d' % (s, t)] = df['atp-mp_%s' %(s)].diff(t)
                x_col.append('atp-mp_diff_%s_%d' % (s, t))
                df['rwp-mp_diff_%s_%d' % (s, t)] = df['rwp-mp_%s' %(s)].diff(t)
                x_col.append('rwp-mp_diff_%s_%d' % (s, t))
            if t > 1:
                df['mid_delta_gap_%d'%(t)] = (df['mid_delta'] - df['mid_delta'].rolling(t).mean())/(df['mid_delta'].rolling(t).std()+1e-5)
                x_col.append('mid_delta_gap_%d'%(t))
                df['macd_mid_%s_%d' % (s, t)] = MACD(df['mp_%s'%(s)], t)
                x_col.append('macd_mid_%s_%d' % (s, t))
                
        df_list.append(df)
    
    x_col += ['wp-mp_x', 'atp-mp_x', 'rwp-mp_x']
    x_col += ['wp-mp_y', 'atp-mp_y', 'rwp-mp_y']
    
    x_col = list(set(x_col))
        
    df = reduce(lambda x,y: x.append(y), df_list)
    df[y_col] = df['r10_x'] - df['r10_y']
    
    x, y = df[x_col].values, df[y_col].values
    x = RobustScaler().fit_transform(x)
    self.x, self.y, self.df, self.x_col, self.y_col = x, y, df, x_col, y_col

  def GetTrainValidTest(self, valid_ratio = 0.2, test_ratio = 0.2):
    test_ratio = 0.2
    valid_ratio = 0.2
    x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=test_ratio, random_state=1)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.2, random_state=1)
    return x_train, x_valid, x_test, y_train, y_valid, y_test

  def Plot(self):
    self.PlotTargetDist()
    PlotFactorCorr(self.x, self.y)

  def PlotTargetDist(self):
    plt.title('y distribution')
    plt.hist(self.df[~np.isnan(self.df[self.y_col])][self.y_col])
    plt.grid()
    plt.show()

if __name__ == '__main__':
  dl = DataLoader('../data/')
  dl.GetTrainValidTest()
  dl.Plot()
