
# RCTs to Regression

Treatment indicator: $D_i \in \left\{0,1\right\}$

> *example:* eligibility for expanded Medicaid

Outcome: $Y_i$

> *example:* number of doctor visits in past 6 months

Potential outcomes $Y_i(0),Y_i(1)$

Individual-level treatment effect \$*{i}=Y*{i}( 1) -Y\_{i}( 0) \$ (can
never know this).

Unbiased estimate of average treatment effect: $$
\hat{\delta}=\bar{Y}_{1}-\bar{Y}_{0},
$$ or OLS coefficient on $D_{i}$ from this regression: $$
Y_{i}=\alpha +\delta D_{i}+\varepsilon _{i}.
$$

Let’s run it!

``` r
library(tidyverse)
library(fixest)
library(glue)
```

``` r
# read in data
oregonhie = read_csv('https://github.com/Mixtape-Sessions/Machine-Learning/blob/main/Labs/data/oregon_hie_table5.csv?raw=true')
```

    ## Rows: 23741 Columns: 27
    ## ── Column specification ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## dbl (27): household_id, treatment, weight, rx_any, rx_num, doc_any, doc_num, er_any, er_num, hosp_any, hosp_num, ddddraw_sur_2, ddddraw_sur_3, ddddraw_sur_...
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
regvars = oregonhie |>
  select(doc_num, treatment, weight, starts_with("ddd")) |>
  drop_na(everything())

(reg = feols(
  doc_num ~ treatment, 
  data = regvars, 
  weights = ~weight
))
```

    ## OLS estimation, Dep. Var.: doc_num
    ## Observations: 23,441 
    ## Standard-errors: IID 
    ##             Estimate Std. Error  t value   Pr(>|t|)    
    ## (Intercept) 1.914219   0.029470 64.95541  < 2.2e-16 ***
    ## treatment   0.268199   0.041657  6.43833 1.2313e-10 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## RMSE: 3.53409   Adj. R2: 0.001723

``` r
effect = coef(reg)['treatment']

cat(
  glue("Estimated effect of Medicaid eligibility on number of doctor visits (bivariate): { sprintf('%.2f', effect) }")
)
```

    ## Estimated effect of Medicaid eligibility on number of doctor visits (bivariate): 0.27

## Aluminum standard: Regression control

The bivariate regression above leans heavily on random assignment of
treatment: $$
D_{i}\perp\!\!\!\!\perp \left( Y_{i}\left( 0\right) ,Y_{i}\left( 1\right) \right) .
$$ Sometimes, even in an RCT, treatment is assigned randomly only
conditional on some set of covariates $X_i$. \>*example:* in the Oregon
HIE, eligibility for Medicaid was granted via lottery, but households
with more members could have more lottery entries. So the lottery
outcome is random only conditional on household size.

So what happens if we don’t have random assignment? In terms of our
regression model above, it means $\varepsilon_i$ may be correlated with
$D_i$. For example, perhaps household size, $X_i$, which increases the
probability of treatment, is also associated with more doctor visits. If
$X_i$ is omitted from the model, it is part of the error term: $$
\varepsilon_i=\beta X_i +\eta_i.
$$ We’ll assume for now that everything else related to doctor visits
($\eta_i$) is unrelated to treatment. What does our bivariate regression
coefficient deliver in this case? $$
\hat{\delta}^{OLS}\underset{p}{\rightarrow}\frac{Cov\left(Y_i,D_i\right)}{Var\left(D_i\right)}=\delta+\gamma\frac{Cov\left(X_i,D_i\right)}{Var\left(D_i\right)}
$$ Simple regression gives us what we want ($\delta$) plus an **omitted
variables bias** term. The form of this term tells us what kinds of
$X_i$ variables we should take care to control for in our regressions.

According to the OVB formula, what kinds of variables should be be sure
to control for in regressions?

Careful investigators will find a set of regressors $X_i$ for which they
are willing to assume treatment is as good as randomly assigned: $$
D_i\perp\!\!\!\!\perp\left( Y_{i}\left( 0\right) ,Y_{i}\left( 1\right) \right) |X_{i}\text{.}
$$ This combined with a linear model for the conditional expectation of
\$% Y\_{i}( 0) \$ and \$Y\_{i}( 1) \$ given $X_{i}$ means we can
estimate the average treatment via OLS on the following regression
equation: $$
Y_{i}=\delta D_{i}+X_{i}^{\prime }\beta +\varepsilon _{i}.
$$

``` r
# Add the household size indicators to our regressor set and run regression:
(reg_cov = feols(
  doc_num ~ treatment + ..("dd"), 
  data = regvars, 
  weights = ~weight
))
```

    ## OLS estimation, Dep. Var.: doc_num
    ## Observations: 23,441 
    ## Standard-errors: IID 
    ##               Estimate Std. Error   t value   Pr(>|t|)    
    ## (Intercept)   1.907095   0.081847 23.300650  < 2.2e-16 ***
    ## treatment     0.314084   0.042962  7.310823 2.7403e-13 ***
    ## ddddraw_sur_2 0.097841   0.108469  0.902022 3.6705e-01    
    ## ddddraw_sur_3 0.091622   0.110122  0.831998 4.0542e-01    
    ## ddddraw_sur_4 0.115474   0.103457  1.116152 2.6437e-01    
    ## ddddraw_sur_5 0.289466   0.103688  2.791688 5.2476e-03 ** 
    ## ddddraw_sur_6 0.111492   0.096427  1.156230 2.4760e-01    
    ## ddddraw_sur_7 0.054379   0.093583  0.581078 5.6119e-01    
    ## ... 10 coefficients remaining (display them with summary() or use argument n)
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## RMSE: 3.52638   Adj. R2: 0.005394

``` r
effect_cov = coef(reg_cov)['treatment']

cat(
  glue("Estimated effect of Medicaid eligibility on number of doctor visits (with controls): { sprintf('%.2f', effect_cov) }")
)
```

    ## Estimated effect of Medicaid eligibility on number of doctor visits (with controls): 0.31

How did the estimate of the effect of Medicaid eligility change? What
does that tell us about the relationship between the included regressors
and the outcome and treatment?

## Connection to ML

Where does machine learning fit into this? It might be tempting to treat
this regression as a prediction exercise where we are predicting $Y_{i}$
given $D_{i}$ and $X_{i}$. Don’t give in to this temptation. We are not
after a prediction for $Y_{i}$, we are after a coefficient on $D_{i}$.
Modern machine learning algorithms are finely tuned for producing
predictions, but along the way they compromise coefficients. So how can
we deploy machine learning in the service of estimating the causal
coefficient \$\$?

To see where ML fits in, first remember that an equivalent way to
estimate \$% \$ is the following three-step procedure:

1.  Regress $Y_{i}$ on $X_{i}$ and compute the residuals,
    $\tilde{Y}_{i}=Y_{i}-\hat{Y}_{i}^{OLS}$, where
    $\hat{Y}_{i}^{OLS}=X_{i}^{\prime}\left( X^{\prime }X\right) ^{-1}X^{\prime }Y$

2.  Regress $D_{i}$ on $X_{i}$ and compute the residuals,
    $\tilde{D}_{i}=D_{i}-\hat{D}_{i}^{OLS}$, where
    $\hat{D}_{i}^{OLS}=X_{i}^{\prime}\left( X^{\prime }X\right) ^{-1}X^{\prime }D$

3.  Regress $\tilde{Y}_{i}$ on $\tilde{D}_{i}$.

Let’s try it!

``` r
yreg = feols(
  doc_num ~ ..("dd"),
  data = regvars, weights = ~weight
)

# Calculate residuals
regvars$ytilde = residuals(yreg)

# regress treatment on covariates
dreg = feols(
  treatment ~ ..("dd"),
  data = regvars, weights = ~weight
)
# Calculate residuals
regvars$dtilde = residuals(dreg)


reg = feols(
  ytilde ~ dtilde,
  data = regvars, weights = ~weight
)

effect_fwl = coef(reg)['dtilde']

cat(
  glue("Estimated effect of Medicaid eligibility on number of doctor visits (partialled out): { sprintf('%.2f', effect_fwl) }")
)
```

    ## Estimated effect of Medicaid eligibility on number of doctor visits (partialled out): 0.31

ML enters the picture by providing an alternate way to generate
$\hat{Y}_i$ and $\hat{D}_i$ when OLS is not the best tool for the job.
The first two steps are really just prediction exercises, and in
principle any supervised machine learning algorithm can step in here.
Back to the whiteboard!
