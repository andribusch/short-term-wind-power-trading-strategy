# Improving Short Term Power Markets Trading Strategy for Wind Power Producer 

Informatics bachelor thesis project.

- **analysis** - All the research behind models, that has been done. Separated into optimization and forecast research
- **src** - The main code of the backtesting and the real model. NB! The folders `simulation` and `option_valuation` serve the function of background, as it was the initial research, that was discontinued and new ideas built upon these insights. 

## Excecutive Summary

The main goal of the present bachelor's thesis is to develop a short term power market trading strategy, based on statistical and machine learning methods and implement it to an automated decision model. The work explores the previously mentioned business problem and the validity of different alternative solutions, depending on the available data and specific business requirements. **The most important components of the technical implementation are the optimization algorithm and a time series forecasting model**.

The most representative parts of the repository are [Trading strategy formulation](analysis/optimization/Imbalance strategy results analysis.ipynb), [Statistical Timeseries Forecast modelling](analysis/forecast/System imbalance Forecasts development_official.ipynb), and [Backtesting results](analysis/optimization/Results analysis_official.ipynb)

The practical contribution of this work is the automated decision system that helps electricity traders (including algorithmic trading algorithms) make better informed decisions and reduce computational overhead in fast-paced, short-term trading.  


The designed solutions were trained on data from 2019 to 2020 and backtested on real market data between period 01/2021 to 02/2022, analyzing the effect of the model on real economic variables. As a result of this model, the monthly average profit was shown to increase by 12-18 \% depending on the chosen risk aversiveness profile. The cost of not managing the risks was reduced by up to 480 \%. The most accurate system imbalance forecast results were obtained, using recurrent neural network model, that reached mean absolute error of 34.93, and the accuracy of directional imbalance 82.29\%.

