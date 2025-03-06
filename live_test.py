import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from datetime import time
from assistance_functions import DeepLearning
from tensorflow.python.keras.saving.save import load_model
from sklearn.metrics import classification_report
import tensorflow as tf
import pandas as pd
from joblib import dump, load


def loss_counter(ytrue, ypred):
    counter = 0
    max_counter = 0

    for pred_index,true in enumerate(ytrue):
        if ypred[pred_index] == 2: continue # if prediction was no trade no loss needs to be counted
        if ypred[pred_index] != true :
            counter+=1
            # print('Loss')
            max_counter = max(max_counter, counter)

        else:
            counter = 0
            # print('Win')
    return max_counter


gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=8044)] ) # Limit to 7.9 gigs approx.
model = load_model("C:/Users/muyu2/OneDrive/Documents/DeepLearning/models/increased_layers_8.7_7.7.h5")

def live_sim(month):
    pd.options.display.width = 0

    #instance by instance( ensure its working correctly and results can be replicated )
    price = pd.read_csv(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/{month}.csv', sep='\t')

    # price = pd.read_csv(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/jan_2025.csv', sep='\t')
    price.columns = ['date', 'time', 'open', 'high','low','close','tickvol', 'volume', 'spread']
    price['datetime'] = price['date'] + " "+ price['time']
    price['datetime'] = pd.to_datetime(price["datetime"], dayfirst=True, format='mixed')
    price.drop(['date', 'time','tickvol','spread'], axis=1, inplace=True)

    #Change column arrangement
    price = price[['datetime', 'open','high','low','close', 'volume']]

    price.drop('volume', axis= 1, inplace=True)
    technicals_price = price.copy()

    #Technical Indicators Creation
    technicals = price.drop('datetime', axis = 1)
    technicals['ema_12'] = ta.ema(technicals_price['close'],length=12)
    technicals['ema_21'] = ta.ema(technicals_price['close'],length=21)


    technicals['atr_7'] = ta.atr(technicals_price['high'],technicals_price['low'], technicals_price['close'], length=7)
    technicals['atr_14'] = ta.atr(technicals_price['high'],technicals_price['low'], technicals_price['close'], length=14)
    technicals.set_index(technicals_price['datetime'], inplace= True)

    temp_df = technicals[['atr_7','atr_14', 'ema_12', 'ema_21']]
    log_returns = np.log(temp_df / temp_df.shift(1))

    technicals.drop(['atr_7','atr_14', 'ema_12', 'ema_21'], axis=1, inplace=True)
    technicals = pd.concat([log_returns,technicals], axis=1)

    technicals.dropna(axis=0, inplace=True)

    kwargs = {
        'sl':0.0006 ,
        'tp':0.0006,
        'look_forward': 3 ,
        'sequence_length': 50,

    }

    y_true = []
    y_pred = []
    x1_values = []
    x2_values = []

    end_time = time(22, 0)
    start_time = time(10, 0)

    for index in range(51, len(technicals)):
        techs = technicals[ index - 51 : index + 20 ]#issue may arise form here

        techs = techs.copy()
        loaded_scaler = load('C:/Users/muyu2/OneDrive/Documents/DeepLearning/scaler_40_len.joblib')

        current_time = techs.index[51].time()
        techs[['atr_7', 'atr_14', 'ema_12', 'ema_21']] = loaded_scaler.transform(techs[['atr_7', 'atr_14', 'ema_12', 'ema_21']])


        if current_time >= end_time or current_time <= start_time:
            continue # avoid high spreads


        price_heikin = techs.reset_index()
        techs, label_true,heikin = DeepLearning.price_gen_live(**kwargs, prices=techs, ashi_price=DeepLearning.heiken_ashi(price_heikin))

        #Data formatting
        # ashi = np.concatenate(heikin).flatten()
        # ashi = ashi[~np.array([isinstance(x, pd.Timestamp) for x in ashi])]
        # x2 = ashi.reshape((1,21,7)).astype(np.float32)

        # correct shape of X1
        tech = np.concatenate(techs).flatten()
        x1 = tech.reshape(1,51,4).astype(np.float32)

        x1_values.append(x1)


        #Prediction and saving
        y_true.append(label_true)


    # x2 = np.concatenate(np.array(x2_values)).flatten()
    # x2_val = x2.reshape(len(x2_values), 21,7)

    x1 = np.concatenate(np.array(x1_values)).flatten()
    x1_val = x1.reshape(len(x1_values), 51,4)

    y_pred = model.predict(x1_val)
    Y_classes = np.argmax(y_pred, axis=1)

    new_preds = []
    buy_thresh, sell_thresh = 0.6 ,0.6
    for i in range(0,len(Y_classes)):
        max_index = Y_classes[i]
        if max_index == 1 and y_pred[i][1] >= sell_thresh : # for selling
            # print('sell')node
            new_preds.append(1)
        elif max_index == 0 and y_pred[i][0] >= buy_thresh:
            new_preds.append(0)
            # print('buy')
        else:
            new_preds.append(2)

    # canceling losses

    print(f"Month is {month}")
    print(classification_report(y_true, new_preds))
    print(f'Max losses in a row  :{loss_counter(y_true,new_preds)}')


month_list = ['jan', 'feb', 'march', 'april','may', 'june']# month_list = ['july','aug','sept','oct','nov', 'dec']

for month in month_list:
    live_sim(month)

