import statsmodels.api as sm
import numpy as np

def build_parameter(features, ar, trend, std_dev):
    ''' Describe! '''
    parameters = []
    [parameters.append(trend[i]) for i in range(trend.shape[0])]
    [parameters.append(ar[j,k]) for j in range(ar.shape[0]) for k in range(ar.shape[0])]
    parameters.append(std_dev)
    return parameters

def sensor(features,ar,trend,std_dev, series_length, ar_order):
    ''' Describe! ''' 
    # Build parameter vector
    parameters = build_parameter(features, ar, trend, std_dev)

    # Create model
    endog = np.zeros((series_length,features))
    model = sm.tsa.VARMAX(endog, order=(ar_order, 0), trend='c')

    # Simulate time series
    sensor = model.simulate(parameters,series_length).reshape(series_length,features)
    return sensor

def add_season(simulation, feature, amplitude, period):
    ''' Describe! '''
    series_length = len(simulation)
    x = np.linspace(1,series_length, series_length)
    season = amplitude * np.sin(2 * np.pi * x / period)
    simulation[:,feature] += season
    return simulation