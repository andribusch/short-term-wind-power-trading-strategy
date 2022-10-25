from datetime import datetime, timedelta
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from api_connections import get_spot_forecast
from statistics import mean, median, StatisticsError

from valuation_algorithm import ValuationAlgorithm


class WeekdayDifferingValuation(ValuationAlgorithm):
    """
    Find the averages for different types of parameters
    """
    data_from_simulation = None
    data_from_charger = None

    def __init__(self):
        self.activations_data = None
        # self.activation_fi_se = None
        self.bidding_data = None
        self.mFRR_prices = None
        self.spot_prices = None

        self.forecasted_premium = None
        self.out_filtered_weekdays = [5, 6]

    def prepare_data_for_algorithm_simulation(self):
        """
        The data from "service" (simulation or daily run) is general. Prepare it for suitable form for algorithm.
        :return:
        """
        self.activations_data = self.data_from_simulation[['Datetime', 'market_regulation_total']]
        # self.activation_fi_se = self.data_from_simulation[['Datetime', 'Market regulation (Finland, Sweden)']]
        self.bidding_data = self.data_from_simulation[['Datetime', 'bid_volumes_baltics']]
        self.mFRR_prices = self.data_from_simulation[['Datetime', 'mFRR_UP_balancing_price']]
        self.spot_prices = self.data_from_simulation[['Datetime', 'SPOTEE']]

    def prepare_data_daily_forecast(self, filename):
        data = pd.read_excel(filename)
        data.fillna(0, inplace=True)
        data['Datetime'] = pd.to_datetime(data['Datetime'], format='%Y%m%d %H:%M')
        self.data_from_charger = data

    def get_prediction_for_tomorrow(self):
        """
        Interface method for regular course of action (Outside simulation)
        :return: Dataframe of full output
        """
        tomorrow_date = datetime.today().date() + timedelta(days=1)

        # If tomorrow WEEKEND
        if tomorrow_date.weekday() in [5, 6]:
            self.out_filtered_weekdays = [0, 1, 2, 3, 4]

        self.bidding_data = self.data_from_charger[['Datetime', 'bid_volumes_baltics']]
        self.activations_data = self.data_from_charger[['Datetime', 'market_regulation_total']]
        self.mFRR_prices = self.data_from_charger[['Datetime', 'mFRR_UP_balancing_price']]
        self.spot_prices = self.data_from_charger[['Datetime', 'SPOTEE']]

        forecasted_volumes = self.forecast_volume()
        forecasted_premiums = self.forecast_price_premium(tomorrow_date)
        forecasted_premium_probabilities = self.forecast_premium_probability()
        forecasted_volume_probabilities = self.forecast_volume_probability()

        output_df = pd.DataFrame(data={'forecasted_price_premiums': forecasted_premiums,
                                       'forecasted_balancing_volumes': forecasted_volumes,
                                       'forecasted_price_premiums_probabilities': forecasted_premium_probabilities,
                                       'forecasted_balancing_volume_probabilities': forecasted_volume_probabilities},
                                 columns=['forecasted_price_premiums', 'forecasted_balancing_volumes',
                                          'forecasted_price_premiums_probabilities',
                                          'forecasted_balancing_volume_probabilities'])
        # output_df.to_excel('output.xlsx')
        return output_df

    def valuate(self):
        """
        The decision-making logic based on predictions about premiums and volumes

        This is boolean value option - We put all the forecasted volume to the market, if certain conditions are met.

        1) Forecasted volume has to be over 5MW
        2) Forecasted premium has to be at least 20eur (that means 20% with 100 euro avg. price)
        3) Probability of volume has to be over 50%
        4) Probability of premium has to be over 50%
        :return:
        """
        # Take the last day
        tomorrow_date = self.data_from_simulation.tail(1)['Datetime'].iloc[0].date() + timedelta(days=1)
        if tomorrow_date.weekday() in [5, 6]:
            self.out_filtered_weekdays = [0, 1, 2, 3, 4]

        MIN_VOLUME = 0
        MIN_VOLUME_PROBALBILITY = 0
        MIN_PREMIUM = 0
        MIN_PREMIUM_PROBABILITY = 0.5

        forecasted_volumes = self.forecast_volume()
        volume_probabilities = self.forecast_volume_probability()
        forecasted_premiums = self.forecast_price_premium(tomorrow_date)
        premium_probabilities = self.forecast_premium_probability()

        option_valuations = []

        # Calculate the decision
        for hour in range(24):
            if tomorrow_date < self.data_from_simulation.head(1)['Datetime'].iloc[0].date() + timedelta(days=7):
                # Don't make positive predictions in first 7 days - not enough data!
                option_valuations.append(0)
            elif forecasted_volumes[hour] >= MIN_VOLUME \
                    and volume_probabilities[hour] >= MIN_VOLUME_PROBALBILITY \
                    and forecasted_premiums[hour] >= MIN_PREMIUM \
                    and premium_probabilities[hour] >= MIN_PREMIUM_PROBABILITY:
                option_valuations.append(1)
            else:
                option_valuations.append(0)

        return option_valuations

    def forecast_volume(self):
        activations_volumes = []
        for hour in range(24):

            hour_filtered_df = self.activations_data[
                self.activations_data.apply(lambda x: x['Datetime'].hour == hour, axis=1) != 0]

            weekday_filtered_df = hour_filtered_df[hour_filtered_df.apply(
                lambda row: row['Datetime'].weekday() not in self.out_filtered_weekdays, axis=1) != 0]

            filtered_df = weekday_filtered_df[weekday_filtered_df.apply(lambda x: x['market_regulation_total'] != 0, axis=1) != 0]
            try:
                activations_volumes.append(mean(filtered_df['market_regulation_total']))
            except StatisticsError:
                activations_volumes.append(0)
        return activations_volumes

    def forecast_volume_probability(self) -> list:
        """
        Kui suur on tõenäosus, et sellel kellaajal aktiveeriti nii palju MWh mFRR?
        count(Tunnid, kus Aktiveeritud kogus > 5MW) / count(kõik tunnid)
        :return: list of 24 probabilitites
        """
        # TODO: Take into account the standard deviation of results.

        volume_activation_probabilities = []

        for hour in range(24):
            # Take this hour
            hour_filtered_df_hour = self.activations_data[self.activations_data.apply(
                lambda row: row['Datetime'].hour == hour, axis=1) != 0]
            # Weekdays
            weekday_filtered_df = hour_filtered_df_hour[hour_filtered_df_hour.apply(
                lambda row: row['Datetime'].weekday() not in self.out_filtered_weekdays, axis=1) != 0]
            # Take only, where activated
            filtered_df = weekday_filtered_df[
                weekday_filtered_df.apply(lambda x: x['market_regulation_total'] != 0, axis=1) != 0]

            soodsad = len(filtered_df.index)
            koik = len(weekday_filtered_df.index)

            try:
                volume_activation_probabilities.append(soodsad / koik)
            except ZeroDivisionError:
                volume_activation_probabilities.append(0)

        return volume_activation_probabilities

    def forecast_price_premium(self, tomorrow_date: datetime.date):
        """
        Take the average price premium
        :return: Premiums for 24 hours into the future
        """
        if self.data_from_charger is None:
            input_data = self.data_from_simulation
        else:
            input_data = self.data_from_charger

        premiums = []
        for hour in range(24):
            # Group by hours

            filtered_all_data_hour = input_data[
                input_data.apply(lambda row: row['Datetime'].hour == hour, axis=1) != 0]

            filtered_all_data_weekday = filtered_all_data_hour[filtered_all_data_hour.apply(
                lambda row: row['Datetime'].weekday() not in self.out_filtered_weekdays, axis=1) != 0]
            # Where premium is present, find avg amount
            filtered_df = filtered_all_data_weekday[filtered_all_data_weekday.apply(
                lambda row: (row['mFRR_UP_balancing_price'] - row['SPOTEE']) > 0, axis=1) != 0]

            try:
                premiums.append(mean(filtered_df['mFRR_UP_balancing_price'] - filtered_df['SPOTEE']))
            except StatisticsError:
                premiums.append(0)

        # This is needed in the probability calculation
        self.forecasted_premium = premiums
        return premiums

    def forecast_premium_probability(self) -> list:
        """
        Given the forecasted value calculated before, what is the (frequentistic) probability, that the premium is that or bigger?
        1) Filter all data to only this hour
        2) Soodsad võimalused filter - When was the premium bigger?
        3) Soodsad / kõik
        :return: Probability of given forecast
        """
        # aggregate_activation_probability = self.calculate_activation_probability(input_data_availability_end)
        if self.data_from_charger is None:
            input_data = self.data_from_simulation
        else:
            input_data = self.data_from_charger

        probabilities = []
        for hour in range(24):
            forecasted_value = self.forecasted_premium[hour]
            # Take this hour from all data! (Need mFRR price and spot price both)
            filtered_all_data_hour = input_data[
                input_data.apply(lambda x: x['Datetime'].hour == hour, axis=1) != 0]

            filtered_all_data_weekday = filtered_all_data_hour[filtered_all_data_hour.apply(
                lambda row: row['Datetime'].weekday() not in self.out_filtered_weekdays, axis=1) != 0]

            koik = len(filtered_all_data_weekday.index)

            # Soodsad: Premium is same or bigger
            filtered_df = filtered_all_data_weekday[
                filtered_all_data_weekday.apply(lambda row: (row['mFRR_UP_balancing_price'] - row['SPOTEE']) >= forecasted_value,
                                             axis=1) != 0]
            soodsad = len(filtered_df.index)

            try:
                probabilities.append(soodsad / koik)
            except ZeroDivisionError:
                probabilities.append(0)
        return probabilities

    # ------------- Business logic methods. For this strategy only ------------------------------


if __name__ == '__main__':

    # TODO: Change the amounts of data given to forecast tomorrow
    valuation = WeekdayDifferingValuation()
    valuation.prepare_data_daily_forecast('../../data/simulation/historical_data_short.xlsx')
    values_for_tomorrow = valuation.get_prediction_for_tomorrow().round(2)
    values_for_tomorrow.to_excel('mFRR_outputs_weekday_differed_' + str(datetime.today().date() + timedelta(days=1)) + '.xlsx')
