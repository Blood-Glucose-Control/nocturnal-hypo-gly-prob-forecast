# Exclusions - Tracking

For tracking the unimplemented assigned models in the following issues: [#83](https://github.com/Blood-Glucose-Control/nocturnal-hypo-gly-prob-forecast/issues/83), [#84](https://github.com/Blood-Glucose-Control/nocturnal-hypo-gly-prob-forecast/issues/84), [#85](https://github.com/Blood-Glucose-Control/nocturnal-hypo-gly-prob-forecast/issues/85), [#86](https://github.com/Blood-Glucose-Control/nocturnal-hypo-gly-prob-forecast/issues/86), [#87](https://github.com/Blood-Glucose-Control/nocturnal-hypo-gly-prob-forecast/issues/87).

Essentially, if you decided not to implement the model, document it here so that we know why it didn't work. Tag with 'redundant' or 'too many bugs', 'cannot do quantile/interval' etc.

## 0 ARMA

StatsModelsARIMA - not needed AutoARIMA works fine.
StatsForecastAutoARIMA - not needed AutoARIMA works fine.

## 1 Exponential Smoothing

## 2 ARCH

## 3 Structural
UnobservedComponents - doesn't seem worth the extra effort at this point, most of the hyperparms are for things we don't need (trend, seasonality, levels)

DynamicFactor - the main advantage of these models are that they can 'model simultaneously and consistently data sets in which the number of series exceeds the number of time series observations'. This is not the case with our problem, so I am not including this model.

TBATS/BATS - no seasonality in our data, not included

Prophets - no seasonality or holidays in our data, not included.


## 4 Deep Learning TS
