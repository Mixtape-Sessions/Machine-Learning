
# Where ML Fits into Causal Inference (review)

The traditional go-to tool for causal inference is multiple regression:
$$
Y_i = \delta D_i + X_i'\beta+\varepsilon_i,
$$ where $D_i$ is the “treatment” or causal variable whose effects we
are interested in, and $X_i$ is a vector of controls, conditional on
which we are willing to assume $D_i$ is as good as randomly assigned.

> *example:* Suppose we are interested in the magnitude of racial
> discrimination in the labor market. One way to conceptualize this is
> the difference in earnings between two workers who are identical in
> productivity, but differ in their race, or, the “effect” of race. Then
> $D_i$ would be an indicator for, say, a Black worker. $Y_i$ would be
> earnings, and $X_i$ would be characteristics that capture determinants
> of productivity, including educational attainment, cognitive ability,
> and other background characteristics.

Where does machine learning fit into causal inference? It might be
tempting to treat this regression as a prediction exercise where we are
predicting $Y_{i}$ given $D_{i}$ and $X_{i}$. Don’t give in to this
temptation. We are not after a prediction for $Y_{i}$, we are after a
coefficient on $D_{i}$. Modern machine learning algorithms are finely
tuned for producing predictions, but along the way they compromise
coefficients. So how can we deploy machine learning in the service of
estimating the causal coefficient \$\$?

To see where ML fits in, first remember that an equivalent way to
estimate \$% \$ is the following three-step procedure:

1.  Regress $Y_{i}$ on $X_{i}$ and compute the residuals,
    $\tilde{Y}% _{i}=Y_{i}-\hat{Y}_{i}^{OLS}$, where
    $\hat{Y}_{i}^{OLS}=X_{i}^{\prime }\left( X^{\prime }X\right) ^{-1}X^{\prime }Y$

2.  Regress $D_{i}$ on $X_{i}$ and compute the residuals,
    $\tilde{D}% _{i}=D_{i}-\hat{D}_{i}^{OLS}$, where
    $\hat{D}_{i}^{OLS}=X_{i}^{\prime }\left( X^{\prime }X\right) ^{-1}X^{\prime }D$

3.  Regress $\tilde{Y}_{i}$ on $\tilde{D}_{i}$.

Steps 1 and 2 are prediction exercises–ML’s wheelhouse. When OLS isn’t
the right tool for the job, we can replace OLS in those steps with
machine learning:

1.  Predict $Y_{i}$ based on $X_{i}$ using ML and compute the residuals,
    $\tilde{Y}% _{i}=Y_{i}-\hat{Y}_{i}^{ML}$, where $\hat{Y}_{i}^{ML}$
    is the prediction from an ML algorithm

2.  Predict $D_{i}$ based on $X_{i}$ using ML and compute the residuals,
    $\tilde{D}% _{i}=D_{i}-\hat{D}_{i}^{ML}$, where $\hat{D}_{i}^{ML}$
    is the prediction from an ML algorithm

3.  Regress $\tilde{Y}_{i}$ on $\tilde{D}_{i}$.

This is the basis for the two major methods we’ll look at today: The
first is “Post-Double Selection Lasso” (Belloni, Chernozhukov, Hansen).
The second is “Double-Debiased Machine Learning” (Chernozhukov,
Chetverikov, Demirer, Duflo, Hansen, Newey, Robins)

# Post Double Selection Lasso (PDS Lasso)


Try it yourself first

``` r
library(tidyverse)
library(fixest)
library(rsample)
library(glmnet)
library(randomForest)
```

``` r
nlsy = read_csv('https://github.com/Mixtape-Sessions/Machine-Learning/blob/main/Labs/data/nlsy97.csv?raw=true')
```

    ## Rows: 1266 Columns: 994
    ## ── Column specification ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## dbl (994): lnw_2016, educ, black, hispanic, other, exp, afqt, mom_educ, dad_educ, yhea_100_1997, yhea_2000_1997, yhea_2100_1997, yhea_2200_1997, ysaq_284_1997, ysaq_285_19...
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
head(nlsy)
```

    ## # A tibble: 6 × 994
    ##   lnw_2…¹  educ black hispa…² other   exp  afqt mom_e…³ dad_e…⁴ yhea_…⁵ yhea_…⁶ yhea_…⁷ yhea_…⁸ ysaq_…⁹ ysaq_…˟ ysaq_…˟ ysaq_…˟ ysaq_…˟ ysaq_…˟ ysaq_…˟ ysaq_…˟ ysaq_…˟ ysaq_…˟
    ##     <dbl> <dbl> <dbl>   <dbl> <dbl> <dbl> <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>
    ## 1    4.08    16     0       0     0    11  7.07      12      12       3       5       0      98       3      12       0      -4      -4      -4      -4      -4      -4      -4
    ## 2    3.29     9     0       0     0    19  4.75       9      10       2       5      10     140       4       9       0      -4      -4      -4      -4      -4      -4      -4
    ## 3    2.83     9     0       1     0    22  1.20      12       9       3       5       7     185       4      13       1      14       4      50      -4      -4      -4      -4
    ## 4    4.31    16     0       0     0    13  8.93      16      18       2       5      10     140       3      11       0      -4      -4      -4      -4      -4      -4      -4
    ## 5    5.99    16     0       1     0    15  2.26      16      16       1       5      10     145       4      14       1      12       4       1      -4      -4      -4      -4
    ## 6    4.71    16     0       0     0    14  8.95      18      20       1       5       7     137       4      13       1      13       3       4      -4      -4      -4      -4
    ## # … with 971 more variables: ysaq_354_1997 <dbl>, ysaq_355_1997 <dbl>, ysaq_356_1997 <dbl>, ysaq_357_1997 <dbl>, ysaq_373_1997 <dbl>, ysaq_374_1997 <dbl>,
    ## #   yinc_14700_1997 <dbl>, yinc_14800_1997 <dbl>, youth_bothbio_01_1997 <dbl>, youth_nonr1dead_01_1997 <dbl>, youth_nonr1inhh_01_1997 <dbl>, youth_nonr1sex_01_1997 <dbl>,
    ## #   youth_nonr2inhh_01_1997 <dbl>, youth_nonr2sex_01_1997 <dbl>, youth_parent_01_1997 <dbl>, youth_parentguar_01_1997 <dbl>, youth_parentsex_01_1997 <dbl>, p4_001_1997 <dbl>,
    ## #   p4_002_1997 <dbl>, p4_003_1997 <dbl>, p4_028_1997 <dbl>, p4_029_1997 <dbl>, p5_101_1997 <dbl>, p5_102_1997 <dbl>, p6_002_1997 <dbl>, p6_003_1997 <dbl>, p6_004_1997 <dbl>,
    ## #   p6_005_1997 <dbl>, pc8_090_1997 <dbl>, pc8_092_1997 <dbl>, pc9_001_1997 <dbl>, pc9_002_1997 <dbl>, pc9_003_1997 <dbl>, pc9_004_1997 <dbl>, pc9_014_1997 <dbl>,
    ## #   pc9_023_1997 <dbl>, pc9_032_1997 <dbl>, pc12_025_1997 <dbl>, pc12_026_1997 <dbl>, pc12_027_1997 <dbl>, pc12_028_1997 <dbl>, paryouth_nonr1dead_1997 <dbl>,
    ## #   paryouth_nonr1inhh_1997 <dbl>, paryouth_nonr1sex_1997 <dbl>, paryouth_nonr2dead_1997 <dbl>, paryouth_nonr2inhh_1997 <dbl>, paryouth_nonr2sex_1997 <dbl>, …

## Define outcome, regressor of interest

- y: `lnw_2016`
- d: `black`

## Simple Regression with no Controls

Regress y on d and print out coefficient

``` r
feols(
  lnw_2016 ~ i(black),
  data = nlsy
)
```

    ## OLS estimation, Dep. Var.: lnw_2016
    ## Observations: 1,266 
    ## Standard-errors: IID 
    ##              Estimate Std. Error   t value   Pr(>|t|)    
    ## (Intercept)  3.179211   0.026626 119.40286  < 2.2e-16 ***
    ## black::1    -0.381721   0.062468  -6.11067 1.3191e-09 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## RMSE: 0.85633   Adj. R2: 0.027925

### …

Is this the effect we’re looking for?

Let’s try a regression where we control for a few things: education
(linearly), experience (linearly), and cognitive ability (afqt,
linearly).

``` r
feols(
  lnw_2016 ~ i(black) + educ + exp + afqt,
  data = nlsy
)
```

    ## OLS estimation, Dep. Var.: lnw_2016
    ## Observations: 1,266 
    ## Standard-errors: IID 
    ##              Estimate Std. Error  t value   Pr(>|t|)    
    ## (Intercept)  1.153807   0.499366  2.31054 2.1019e-02 *  
    ## black::1    -0.261690   0.063867 -4.09740 4.4449e-05 ***
    ## educ         0.089260   0.019526  4.57147 5.3190e-06 ***
    ## exp          0.036226   0.016727  2.16580 3.0514e-02 *  
    ## afqt         0.037111   0.010226  3.62896 2.9595e-04 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## RMSE: 0.824041   Adj. R2: 0.097708

### … 

How does it compare to the simple regression?

But who is to say the controls we included are sufficient? We have a
whole host (hundred!) of other potential controls, not to mention that
perhaps the controls we did put in enter linearly. This is a job for ML!

To prep, let’s define a matrix X with all of our potential controls:

``` r
potential_controls = setdiff(colnames(nlsy), c("lnw_2016", "black"))
```

## Post Double Selection Lasso

### Step 1: Lasso the outcome on X

Don’t forget to standard Xs, or choose the normalize=True option

``` r
X = as.matrix(nlsy[, potential_controls])
y = nlsy[["lnw_2016"]]

# Run cross-validation for y
lasso_y <- cv.glmnet(x=X, y=y)

lasso_y_coefs = coef(lasso_y, lasso_y$lambda.1se)
lasso_y_coefs = as.matrix(lasso_y_coefs)

keep_y = rownames(lasso_y_coefs)[lasso_y_coefs != 0]
# Don't need intercept
keep_y = setdiff(keep_y, "(Intercept)")
```

### Step 2: Lasso the treatment on d

``` r
# Run cross-validation for d
d = nlsy[["black"]]
lasso_d <- cv.glmnet(x=X, y=d)

lasso_d_coefs = coef(lasso_d, lasso_d$lambda.1se)
lasso_d_coefs = as.matrix(lasso_d_coefs)

keep_d = rownames(lasso_d_coefs)[lasso_d_coefs != 0]
# Don't need intercept
keep_d = setdiff(keep_d, "(Intercept)")
```

### Step 3: Form the union of controls

``` r
keep = union(keep_y, keep_d)
```

### Concatenate treatment with union of controls and regress y on that and print out estimate

``` r
# Need `` surrounding variables since some variables start with underscore
formula = paste(
  "lnw_2016 ~ black + ", 
  paste0("`", keep, "`", collapse = " + ")
)
formula = as.formula(formula)

(fullreg = feols(formula, data = nlsy))
```

    ## Variables '`_BGpp4_029__1`' and '`_BGpfp_yfmr_4`' have been removed because of collinearity (see $collin.var).

    ## OLS estimation, Dep. Var.: lnw_2016
    ## Observations: 1,266 
    ## Standard-errors: IID 
    ##                             Estimate Std. Error   t value   Pr(>|t|)    
    ## (Intercept)                 2.341612   0.225027 10.405919  < 2.2e-16 ***
    ## black                      -0.141191   0.086111 -1.639644 1.0134e-01    
    ## educ                        0.054290   0.010662  5.091736 4.1163e-07 ***
    ## hispanic                    0.068641   0.079383  0.864671 3.8739e-01    
    ## afqt                        0.038466   0.011318  3.398817 6.9899e-04 ***
    ## yhea_2200_1997              0.000995   0.000701  1.419437 1.5603e-01    
    ## p4_001_1997                -0.067009   0.026496 -2.528978 1.1567e-02 *  
    ## cv_bio_mom_age_child1_1997 -0.002313   0.004404 -0.525223 5.9953e-01    
    ## ... 58 coefficients remaining (display them with summary() or use argument n)
    ## ... 2 variables were removed because of collinearity (`_BGpp4_029__1` and `_BGpfp_yfmr_4`)
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## RMSE: 0.802714   Adj. R2: 0.100286

## Double-Debiased Machine Learning

For simplicity, we will first do it without sample splitting

### Step 1: Ridge outcome on Xs, get residuals

``` r
# Run cross-validation for y
ridge_y <- cv.glmnet(x=X, y=y, alpha = 0)

y_hat = predict(ridge_y, ridge_y$lambda.1se, newx = X)
nlsy$y_resid = nlsy$lnw_2016 - as.numeric(y_hat)
```

### Step 2: Ridge treatment on Xs, get residuals

``` r
# Run cross-validation for y
ridge_d <- cv.glmnet(x=X, y=d, alpha = 0)

d_hat = predict(ridge_d, ridge_d$lambda.1se, newx = X)
nlsy$d_resid = nlsy$black - as.numeric(d_hat)
```

### Step 3: Regress y resids on d resids and print out estimate

``` r
feols(y_resid ~ d_resid, nlsy)
```

    ## OLS estimation, Dep. Var.: y_resid
    ## Observations: 1,266 
    ## Standard-errors: IID 
    ##                  Estimate Std. Error      t value   Pr(>|t|)    
    ## (Intercept) -1.330000e-15   0.024103 -5.54000e-14 1.00000000    
    ## d_resid     -3.168053e-01   0.084223 -3.76149e+00 0.00017663 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## RMSE: 0.856933   Adj. R2: 0.010287

### The real thing: with sample splitting

``` r
set.seed(5)
N_folds = 5
# 5 folds with equal 
nlsy$fold_id = sample(1:N_folds, size = nrow(nlsy), replace = T)

nlsy$y_resid = 0
nlsy$d_resid = 0

# Loop through each fold, use other 4 folds to estimate
for(i in 1:5) {
  in_training = (nlsy$fold_id != i)
  in_test = (nlsy$fold_id == i)

  # Ridge regression for y using training
  ridge_y = cv.glmnet(
    x=X[in_training,], y=y[in_training], alpha = 0
  )
  # Calculate residuals for testing
  nlsy[in_test, "y_resid"] =
    y[in_test] - predict(ridge_y, newx = X[in_test, ])

  # Ridge regression for d using training
  ridge_d = cv.glmnet(
    x=X[in_training,], y=d[in_training], alpha = 0
  )
  # Calculate residuals for testing
  nlsy[in_test, "d_resid"] = 
    d[in_test] - predict(ridge_d, newx = X[in_test, ])
}

# k-fold cross-validation ensures standard errors are fine
feols(
  y_resid ~ d_resid, data = nlsy, vcov = "hc1"
)
```

    ## OLS estimation, Dep. Var.: y_resid
    ## Observations: 1,266 
    ## Standard-errors: Heteroskedasticity-robust 
    ##              Estimate Std. Error   t value   Pr(>|t|)    
    ## (Intercept) -0.000992   0.024262 -0.040884 0.96739479    
    ## d_resid     -0.302294   0.077506 -3.900259 0.00010114 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## RMSE: 0.862505   Adj. R2: 0.013436

## Now do DML using Random Forest!

``` r
set.seed(5)
N_folds = 5
# 5 folds with equal 
nlsy$fold_id = sample(1:N_folds, size = nrow(nlsy), replace = T)

nlsy$y_resid = 0
nlsy$d_resid = 0

# Loop through each fold, use other 4 folds to estimate
for(i in 1:5) {
  in_training = (nlsy$fold_id != i)
  in_test = (nlsy$fold_id == i)

  # Ridge regression for y using training
  ridge_y = randomForest(
    x = X[in_training,], y = y[in_training]
  )
  # Calculate residuals for testing
  nlsy[in_test, "y_resid"] =
    y[in_test] - predict(ridge_y, newdata = X[in_test, ])

  # Ridge regression for d using training
  ridge_d = randomForest(
    x = X[in_training,], y = d[in_training]
  )
  # Calculate residuals for testing
  nlsy[in_test, "d_resid"] = 
    d[in_test] - predict(ridge_d, newdata = X[in_test, ])
}
```

    ## Warning in randomForest.default(x = X[in_training, ], y = d[in_training]): The response has five or fewer unique values. Are you sure you want to do regression?

    ## Warning in randomForest.default(x = X[in_training, ], y = d[in_training]): The response has five or fewer unique values. Are you sure you want to do regression?

    ## Warning in randomForest.default(x = X[in_training, ], y = d[in_training]): The response has five or fewer unique values. Are you sure you want to do regression?

    ## Warning in randomForest.default(x = X[in_training, ], y = d[in_training]): The response has five or fewer unique values. Are you sure you want to do regression?

    ## Warning in randomForest.default(x = X[in_training, ], y = d[in_training]): The response has five or fewer unique values. Are you sure you want to do regression?

``` r
# k-fold cross-validation ensures standard errors are fine
feols(
  y_resid ~ d_resid, data = nlsy, vcov = "hc1"
)
```

    ## OLS estimation, Dep. Var.: y_resid
    ## Observations: 1,266 
    ## Standard-errors: Heteroskedasticity-robust 
    ##              Estimate Std. Error   t value Pr(>|t|)    
    ## (Intercept) -0.007765   0.023164 -0.335205  0.73753    
    ## d_resid     -0.136571   0.074804 -1.825707  0.06813 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## RMSE: 0.823375   Adj. R2: 0.002127
