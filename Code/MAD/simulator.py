import numpy as np

def sensor_simulation(series_length, features, ar, lag, constant, trend, error_amp, season_amp=0, season_period=1):
    e = np.zeros(features)
    y = np.zeros((series_length, features))
    for i in range (1, series_length):      
        for j in range(0, features):
            e[j] = error_amp*np.random.randn()
        season = season_amp * np.sin(2 * np.pi * i / season_period)
        y[i,:] = np.dot(ar, y[(i-lag),:]) + e + trend*i + constant + season
    return y

def point_anomaly(percentage, length, amplitude=1):
    anomaly = np.zeros(length)
    for i in range(1,length):
        if np.random.rand() < percentage:
            anomaly[i] = amplitude
    return anomaly

def trend_anomaly(percentage, length, anomaly_length, amplitude=1):
    anomaly = np.zeros(length)
    for i in range(1,length):
        if np.random.rand() < percentage:
            for j in range(0, anomaly_length):
                anomaly[i-j] = amplitude*(1-(j/anomaly_length))
                anomaly[i+j] = amplitude*(1-(j/anomaly_length))
    return anomaly
