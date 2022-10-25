from abc import ABC, abstractmethod

import pandas as pd
import statsmodels.api as sm


class ImbalanceEstimator(ABC):
	"""The abstract wrapper Imbalance forecasting class."""

	def __init__(self):
		self.training_data = None
		self.forecast_horizon = 2

		self.model = None  # A different object in different implementations

	@abstractmethod
	def fit(self) -> None:
		"""Train the model, using training data attached to implementing class."""
		return

	@abstractmethod
	def forecast(self) -> pd.DataFrame:
		"""The method forecasts out of the sample"""
		return


class ARIMAEstimator(ImbalanceEstimator):
	"""
	Use uni-variate timeseries model ARIMA in order to generate scenarios.

	"""

	def fit(self) -> None:
		# Training data has to be already transformed into correct format
		# Analysis of the suitable rank is done in separate analysis notebook
		model = sm.tsa.arima.ARIMA(self.training_data['Baltics'], order=(3, 0, 1))
		fitted = model.fit()
		self.model = fitted
		return

	# TODO: The returned DataFrame is not 100% sure yet!
	def forecast(self) -> pd.DataFrame:
		prediction_results = self.model.get_forecast(self.forecast_horizon)
		point_forecast = prediction_results.predicted_mean.iloc[-1]
		conf_int = prediction_results.conf_int(alpha=0.05)  # This returns DataFrame already
		return point_forecast

	def set_training_data(self, training_data):
		self.training_data = training_data

	def append_observation(self, observation: pd.DataFrame) -> None:
		"""
		To reduce running time in backtest, append the observation to training data
		:param observation: 1 observation, with proper structure (datetime index, that matches training)
		:return:
		"""
		self.model = self.model.append(observation)

class TransformerEstimator(ImbalanceEstimator):
	"""Use transformer neural networks architecture, in order to generate scenarios"""

	def fit(self) -> None:
		print("fitted Transformer model")
		return

	def forecast(self):
		return [5, 6, 8]


if __name__ == "__main__":
	arima = ARIMAEstimator()
	transformer = TransformerEstimator()

	arima.fit()
	transformer.fit()
