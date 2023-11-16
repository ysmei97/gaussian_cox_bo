import numpy as np
import pandas as pd
import csv
from scipy.stats import norm, poisson, uniform, expon
from datetime import datetime


def load_coal_mine_disaster_data():
    with open('../dataset/coal_mine_disasters.csv') as coal_csv:
        coal_data = csv.reader(coal_csv)
        date = []
        for row in coal_data:
            date.append(row[1:])
        arr_time = np.array(date).astype(float) - 1851
    return arr_time, len(arr_time)


def load_tornado_data():
    day_seconds = 86400.0
    with open('../dataset/2022_tornadoes.csv') as tornado_csv:
        tornado_df = pd.read_csv(tornado_csv)
        tornado_df = tornado_df[tornado_df['loss'] > 0]  # tornado with actual loss
        time = tornado_df['time'].astype('str')
        month = tornado_df['mo']
        day = tornado_df['dy']
        dec_time = time.str.split(':').apply(lambda x: (int(x[0]) * 3600 + int(x[1]) * 60 + int(x[2])) / day_seconds)
        arr_time = np.array(((month - 1) * 30 + day + dec_time).values)[:, None]
        location = tornado_df[['slon', 'slat']]
        location['intensity'] = 0
    return arr_time, len(arr_time), location


def load_crime_data():
    day_seconds = 86400.0
    with open('../dataset/crime_incidents_in_2022.csv') as crime_csv:
        crime_df = pd.read_csv(crime_csv)
        crime_df = crime_df[crime_df['METHOD'] == 'GUN']  # gun violence
        date = crime_df['REPORT_DAT'].astype('str')
        time = []
        location = crime_df[['LONGITUDE', 'LATITUDE']]
        for i in date.index:
            _date = datetime.strptime(date[i], '%Y/%m/%d %H:%M:%S+00')
            if _date.month in [5, 6]:  # select month
                if _date.day <= 31:  # select day
                    time.append((_date.month - 5) * 30 + (_date.day - 1) + (_date.hour * 3600 + _date.minute * 60 + _date.second) / day_seconds)
                    continue
            location = location.drop(i, axis=0)
        arr_time = np.sort(np.array(time))
        sorted_indices = np.argsort(np.array(time))
        location = location.reset_index(drop=True).reindex(sorted_indices).reset_index(drop=True)
        location['intensity'] = 0
    return arr_time, len(arr_time), location


def load_taxi_data():
    with open('../dataset/porto_taxi.csv') as taxi_csv:
        taxi_df = pd.read_csv(taxi_csv, engine='python', on_bad_lines='skip')
        taxi_df = taxi_df[taxi_df['CALL_TYPE'] == 'C']
        taxi_df['POLYLINE'] = taxi_df['POLYLINE'].str.replace('[', '')
        taxi_df['POLYLINE'] = taxi_df['POLYLINE'].str.replace(']', '')
        taxi_df["GEO_LEN"] = taxi_df["POLYLINE"].apply(lambda x: len(x))
        taxi_df['POLYLINE'] = taxi_df['POLYLINE'].apply(lambda x: x.split(','))
        taxi_df.drop(taxi_df[taxi_df["GEO_LEN"] == 0].index, axis=0, inplace=True)
        lon, lat = [], []
        for index, row in taxi_df.iterrows():
            lon.append(float((row['POLYLINE'])[0]))
            lat.append(float((row['POLYLINE'])[1]))
        taxi_df['START_LON'] = np.array(lon)
        taxi_df['START_LAT'] = np.array(lat)
        latitude = (taxi_df['START_LON'] >= -8.63) & (taxi_df['START_LON'] <= -8.60)
        longitude = (taxi_df['START_LAT'] >= 41.15) & (taxi_df['START_LAT'] <= 41.18)
        taxi_df = taxi_df[latitude & longitude]
        arr_time = (taxi_df['TIMESTAMP'] - 1372636800) / 3600
        location = taxi_df[['START_LON', 'START_LAT']].reset_index(drop=True)
        location['intensity'] = 0
    return arr_time.to_numpy(dtype='float'), len(arr_time), location


def load_synthetic_data(func:int, test:int):
    with open('../dataset/synthetic_funcs/func_' + str(func) + '.csv') as func_csv:
        func_data = pd.read_csv(func_csv)
        arr_time = func_data['test' + str(test)].dropna()
    return arr_time.to_numpy(dtype='float'), len(arr_time)

# Test dataloader
if __name__ == "__main__":
    # arr, arr_size = load_coal_mine_disaster_data()
    # arr, arr_size, loc = load_tornado_data()
    # arr, arr_size, loc = load_crime_data()
    arr, arr_size, loc = load_taxi_data()
    # arr, arr_size = load_synthetic_data('func_1', 6)
    print(arr_size, loc.head)
