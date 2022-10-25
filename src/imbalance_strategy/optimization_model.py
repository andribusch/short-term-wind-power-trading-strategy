import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import linprog
import scipy.stats
from scipy.stats import norm

import matplotlib.pyplot as plt

from imbalance_estimator import ImbalanceEstimator


class OptimizationModel:

	def __init__(self):
		self.imbalance_estimator = None  # Imbalance forecast

		#
		self.imbalance_data = None
		# Data for optimization - mainly the wind forecasts
		self.portfolio_data = None

	def optimize_imbalance_position(self) -> float:
		"""
		Main working method of the Optimization Model

		We have to have information about
		1) The most recent imbalance state + forecasted scenarios
		2) All information about the portfolio (from Periotheus)

		This information comes from
		a) Charger - The function, that gathers data from sources
		b)

		Kõigepealt proovime talupoja mõistusega!!
		:return:
		"""
		IMBALANCE_POSITION = 30.5
		HISTORICAL_ERROR_STDERR = 13.75
		HISTORICA_ERROR_MEAN = 4.91


		print("meil on tuuletoodangu ennustus")
		print(self.portfolio_data)

		print("meil on imbalance ennustus")
		if self.imbalance_estimator.model:
			# Pass in new data, dont re-fit
			# self.imbalance_estimator.append_observation()
			pass
		else:
			self.imbalance_estimator.fit()

		imbalance_point_forecast = self.imbalance_estimator.forecast()
		print(imbalance_point_forecast)

		print('Peale igat forecasti, lisame tegeliku observationi treeningandmetele')
		new_observation = self.imbalance_data.iloc[-1:]
		self.imbalance_estimator.append_observation(new_observation)



		print('Arvutame optimaalse tuuletoodangu kvantiili')
		alpha = self.calculate_optimal_quantile(imbalance_point_forecast)

		print('Arvutame tuuleprognoosile tõenäosuspiirid')
		# Every hour, we deal with a different random variable
		# We have the mean, as the point estimate, and error distribution
		original_forecasted_wind_production = self.portfolio_data['2_hours_before_MW'].values[-1]

		# Normal distribution N(X| error_mu+forecast, error_stderr)
		x = HISTORICAL_ERROR_STDERR * np.random.randn(1, 1000) + original_forecasted_wind_production
		x = np.sort(x)

		y = 1. * np.arange(len(x[0])) / (len(x[0]) - 1)

		optimal_quantile_value = x[0][int(alpha * 1000)]

		return optimal_quantile_value

	def calculate_optimal_quantile(self, forecasted_imbalance, asymmetry_multiplier=0.1):
		"""
		model: Could be whatever estimator, to predict imbalance state. Could be a wrapper class, that has forecast() method
		asymmetry_multiplier: Hyperparameter, that increases the asymmetry. Higher value puts more confidence to prediction result.

		return: Alpha - the wind forecast quantile, that optimizes the profits, given imbalance market state estimation.
		"""
		loss_function_value = forecasted_imbalance * asymmetry_multiplier
		x = [-1, 0, 1]

		if forecasted_imbalance > 0:
			pi = [1, 0, loss_function_value]
		else:
			pi = [abs(loss_function_value), 0, 1]
		optimal_quantile = loss_function_value / (loss_function_value + 1) if forecasted_imbalance > 0 else 1 / (
					loss_function_value + 1)

		plt.plot(x, pi)
		plt.show()

		return optimal_quantile

	def get_portfolio_data(self, end_datetime: datetime) -> None:
		"""
		Charge the class instance with portfolio data from Periotheus
		Get all the data in order to perform optimization
		a) Portfolio data - Periotheus

		NB! This is for standard working. In simulation, a different method is used!
		:return: None
		"""
		# TODO: Currently from file, but can be changed with. Not very easy and requires thinking.
		# FIXME: Here, speed is also critical, to get as much time to execute trades in the market
		data = pd.read_excel('../../data/imbalance_strategy/EG_portfolio_march.xlsx')
		# Parse the datetime
		for idx, each in data.iloc[:, 0].iteritems():
			date = each.split('-')[0].strip()
			date = ''.join(date.split(','))
			date = datetime.strptime(date, '%d.%m.%Y %H:%M')
			data.iloc[idx, 0] = date
		data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0])
		data.index = data.iloc[:, 0]  # Change the col to index with datetimes, instead of integers!
		data.drop('Unnamed: 0', axis=1, inplace=True)
		self.portfolio_data = data[:end_datetime]

	def get_imbalance_data(self, end_datetime: datetime) -> None:
		"""
		Charge the class instance with imbalance data from BTD API
		b) Market data - BTD API

		NB! This is for standard working. In simulation, a different method is used!
		:return:
		"""
		# TODO: Currently from file, but can be changed with API call easily
		data = pd.read_csv('../../data/imbalance_strategy/imbalance_march.csv', index_col=1, parse_dates=True)
		data.drop(['Unnamed: 0', 'to'], inplace=True, axis=1)

		aggregated_df = change_timeseries_aggregation_step(data, self.AGGREGATION)
		aggregated_df['from'] = pd.to_datetime(aggregated_df['from'])

		self.imbalance_data = aggregated_df

		# Give the data to estimator, if it exists
		# TODO: Where is the best place, where to give data to imbalance estimator
		if self.imbalance_estimator:
			self.imbalance_estimator.set_training_data(self.imbalance_data)

	def set_imbalance_estimator(self, imbalance_estimator: ImbalanceEstimator) -> None:
		"""Set the forecasting method, for scenario generation"""
		self.imbalance_estimator = imbalance_estimator


def change_timeseries_aggregation_step(timeseries_df: pd.DataFrame, aggregation_step: int) -> pd.DataFrame:
	"""
	Take univariate timeseries with 1-min data and aggregate them with given aggregation step
	:param timeseries_df:
	:param aggregation_step:
	:return: aggregated  DataFrame
	"""
	n_count = timeseries_df.shape[0]
	n_count_aggregated = round(n_count / aggregation_step)
	new_entries = {'from': [], 'aggregate_system_imbalance': []}
	for i in range(0, n_count, aggregation_step):
		chunk = timeseries_df.iloc[i:(i + aggregation_step), :]
		aggregate = (chunk.iloc[:, 1].mean())

		new_entry_timestamp = str(chunk.index[0]).split('+')[0]
		new_entries['from'].append(new_entry_timestamp)
		new_entries['aggregate_system_imbalance'].append(aggregate)
	aggregated_df = pd.DataFrame(data=new_entries)
	return aggregated_df
