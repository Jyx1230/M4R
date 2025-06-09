# M4R

This thesis develops a flexible spatial model to capture how extreme rainfall and wind speed
co-occur in hurricanes, addressing a key shortcoming of standard CAR and MCAR models which
assume a single global spatial dependence parameter across the entire domain. We build a cus-
tom bivariate framework in which both total precipitation and wind speed follow non-Gaussian
distributions, linked by an edge-based CAR prior that allows the spatial correlation structure to
vary locally. To account for the known weakening of this coupling over land, the model is then
extended by adding a simple linear predictor on the per-edge correlation, driven by the local
land–sea fraction. Model comparison relies on evaluation metrics and residual diagnostics. The
results demonstrate that fitting precipitation and wind jointly captures their interaction and
yields a better fit than the standard CAR models and including the land–sea fraction clearly
reveals that correlation drops sharply at landfall and recovers offshore.

It contains the following files:
- EDA_1 and EDA_2: select threshold, do sptail filtering, empirical analysis and plot
- CAR with Constant Correlation.R, CAR with no smoothing.R, MCAR with cholesky.R: candidate models used
  to compare with our custom bivariate model
- bivariate model with log normal and weibull.R, bivariate model with log student t and lognormal.R, bivariate
  model with log student-t and generalized gamma.R: out custom bivariate model with different prior pairs
- bivariate model with linear predictor.R: bivaraite model with extension of linear predictor driven by the local
  land–sea fraction
