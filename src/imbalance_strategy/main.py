import numpy as np
import pandas as pd
import datetime
import os
import glob


import statsmodels.api as sm
from scipy.stats import weibull_min


# Constants

ASYMMETRY_MULTIPLIER = 0.1
FORECAST_STEPS = 2


class ImbalanceForecast:

    def __init__(self, forecast_model):
        self.forecast_model = forecast_model
        self.fitted_model = None

    def train_model(self, training_data: pd.DataFrame):

        zero_indexes = training_data[training_data == 0].index
        data = training_data.drop(zero_indexes)

        # Change index explicitly to avoid errors
        hour_index = data.index.to_period('H')
        data.index = hour_index

        if self.forecast_model == 'arima':

            self.fitted_model = sm.tsa.arima.ARIMA(data, order=(3, 0, 2)).fit()
        elif self.forecast_model == 'exponential_smoothing':
            self.fitted_model = sm.tsa.arima.ARIMA(data, order=(0, 1, 1)).fit()


    def forecast_imbalance(self, imbalance_timeseries):
        """
        Interface method for different forecasts to be used for imbalance QUANTITY estimation.
        :param imbalance_timeseries:
        :return:
        """

        # First, we remove all data until the last seen value
        zero_indexes = imbalance_timeseries[imbalance_timeseries == 0].index
        data = imbalance_timeseries.drop(zero_indexes)

        self.train_model(data)
        yhat = 0
        try:
            if self.forecast_model == 'arima':
                yhat = self.fitted_model.forecast(FORECAST_STEPS)

                yhat = yhat[-1]  # Take the last
            elif self.forecast_model == 'exponential_smoothing':
                yhat = self.fitted_model.forecast(FORECAST_STEPS)
            else:
                yhat = data.iloc[-1]
        except:
            print('Forecast model not fitted')

        return yhat


def calculate_optimal_quantile(imbalance_forecast, asymmetry_multiplier):
    """

    :param imbalance_forecast:
    :param asymmetry_multiplier:
    :return:
    """

    loss_function_value = imbalance_forecast * asymmetry_multiplier

    if imbalance_forecast > 0:
        optimal_quantile = loss_function_value / (loss_function_value + 1)
    elif imbalance_forecast == 0:
        optimal_quantile = 0.5
    else:
        optimal_quantile = 1 / (abs(loss_function_value) + 1)

    return optimal_quantile


def calculate_error_distribution():
    """
    Calculate the probability distribution of wind forecast error distribution.
    Weibull min value distribution
    :return:
    """
    c, loc, scale = (12.15088, -141.36674, 142.20086)
    frozen_dist = weibull_min(c, loc=loc, scale=scale)
    return frozen_dist


def select_last_available_forecast(full_forecast):
    """
    Select single value from dataframe, that gets manipulated
    :param full_forecast:
    :return:
    """
    datetime_2h_after = datetime.datetime.now().replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=2)
    last_value = full_forecast[datetime_2h_after.isoformat().split('.')[0]]
    return last_value


def get_last_file(folder):
    # From stackoverflow.com/questions/39327032/how-to-get-the-latest-file-in-a-folder

    files = glob.glob(folder)
    files.sort(key=os.path.getmtime)
    return files[-1]  # Take the newest


def run_model():
    """
    Main interface method for running the model
    :return:
    """
    folder_path = '//cisfs/Ealeksei/from_periotheus/imbalance_cost_optimization/*'
    filename = get_last_file(folder_path)
    # filename='./data/EG_imbalance_risk_management_model_input.csv'
    data = pd.read_csv(filename, index_col=0, parse_dates=True)

    # Save the imported data for future reference
    short_name = '_'.join(filename.split('_')[-2:])
    data.to_csv('./imports/' + 'imports_' + short_name)

    imbalance_data = data.iloc[:, 0]

    imbalance_forecast = ImbalanceForecast('arima')
    # 1. Conduct imbalance forecast
    forecasted_imbalance = imbalance_forecast.forecast_imbalance(imbalance_data)

    # 2. Calculate optimal quantile
    alpha = calculate_optimal_quantile(forecasted_imbalance, ASYMMETRY_MULTIPLIER)

    # 3. Enhance the wind forecast

    forecast_point_values = data.loc[:, 'meteologica_forecast']
    single_last_available = select_last_available_forecast(forecast_point_values)
    error_distribution_rvs = calculate_error_distribution()

    error_component = error_distribution_rvs.ppf(alpha)  # PPF - Inverse cumulative distribution
    optimal_nominated_value = single_last_available + error_component

    da_sales = data.loc[:, 'DA_sales']
    optimal_id_pos = optimal_nominated_value - da_sales

    # Compile the data
    datetime_2h_after = datetime.datetime.now().replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=2)
    data = {'optimal_nominated_value': optimal_nominated_value,
            'optimal_quantile': alpha,
            'imbalance_forecast_2h': forecasted_imbalance,
            'optimal_intraday_position': optimal_id_pos}
    output_data = pd.DataFrame(data=data, index=[datetime_2h_after])
    output_data.index.name = 'timestamp'

    return output_data


if __name__=='__main__':
    output_data = run_model()
    filename = 'EG_imbalance_optimization_output' + str(datetime.datetime.now().isoformat().split('.')[0]).replace(':','_') +  '.csv'
    output_data.to_csv('./exports/' + filename)
    output_data.to_csv('//cisfs/Ealeksei/to_periotheus/from_imbalance_optimization/' + filename)
