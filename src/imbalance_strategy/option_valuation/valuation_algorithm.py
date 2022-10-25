"""
Base class for valuation algorithms. Define common functionalities.
Inspiration for strategy pattern design: https://refactoring.guru/design-patterns/strategy/python/example
"""
import datetime
from abc import ABC, abstractmethod


class ValuationAlgorithm(ABC):
    """
    This is the Strategy.
    All base classes take input data as a parameter. Data is in standard form.
    input_data_availability_end - This is the index, where the "availability" ends. Simulation sees input data until
    that index.
    """
    @abstractmethod
    def get_prediction_for_tomorrow(self):
        """
        A method for using the strategy for regular prediction OUTSIDE SIMULATION
        :return:
        """
        pass

    @abstractmethod
    def valuate(self) -> list:
        pass

    @abstractmethod
    def forecast_volume(self) -> list:
        pass

    @abstractmethod
    def forecast_volume_probability(self) -> list:
        pass

    @abstractmethod
    def forecast_price_premium(self, tomorrow_date: datetime.date) -> list:
        pass

    @abstractmethod
    def forecast_premium_probability(self) -> list:
        pass
