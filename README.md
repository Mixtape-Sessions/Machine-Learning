![Mixtape Sessions Banner](https://raw.githubusercontent.com/Mixtape-Sessions/Machine-Learning/main/img/banner.png)


## About

Machine Learning's wheelhouse is out-of-sample prediction, but these powerful methods can be deployed in service of causal inference. This two-session workshop will introduce the basics of machine learning prediction methods, including lasso and random forests and how they feature in causal inference methods like double machine learning (DML) and post-double selection lasso (PDS lasso). The course covers the conceptual and theoretical basis for the methods and also gets into the nuts and bolts of implementation in python and Stata using real-world data.


## Schedule

### Day 1

1. What’s your question? (prediction vs. causality)

2. Standard tools of causal inference
   - gold standard: RCT
   - Multiple Regression

3. ML prediction tools
   - prediction objective
   - bias-variance tradeoff
   - lasso
   - random forest

### Day 2

1. Where does ML prediction fit within causal inference?
   - flexibly adjust for covariates
   - estimate heterogeneous treatment effects

2. Post-Double Selection Lasso
   - Theory
   - Implementation

3. Double Machine Learning
   - Theory
   - Implementation


## Readings

The following is a set of introductory readings for machine learning and causal inference and is in a good potential reading order

[Kleinberg, Ludwig, Mullainathan, and Obermeyer (2015)](https://github.com/Mixtape-Sessions/Machine-Learning/raw/main/Readings/Kleinberg_Ludwig_Mullainathan_Obermeyer_2015.pdf)

[Varian (2014)](https://github.com/Mixtape-Sessions/Machine-Learning/raw/main/Readings/Varian_2014.pdf)

[Mullainathan and Spiess (2017)](https://github.com/Mixtape-Sessions/Machine-Learning/raw/main/Readings/Mullainathan_Spiess_2017.pdf)

[Athey and Imbens 2019)](https://github.com/Mixtape-Sessions/Machine-Learning/raw/main/Readings/Athey_Imbens_2019.pdf)

[Belloni, Chernozhukov, and Hansen (2014)](https://github.com/Mixtape-Sessions/Machine-Learning/raw/main/Readings/Belloni_Chernozhukov_Hansen_2014.pdf)


## Slides

[Day 1](https://github.com/Mixtape-Sessions/Machine-Learning/raw/main/Slides/Day-1.pdf)

[Day 2](https://github.com/Mixtape-Sessions/Machine-Learning/raw/main/Slides/Day-2.pdf)

## Coding Labs

1. [RCT to Regression](https://github.com/Mixtape-Sessions/Machine-Learning/blob/main/Labs/python/RCT%20to%20Regression.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mixtape-Sessions/Machine-Learning/blob/main/Labs/python/RCT%20to%20Regression.ipynb)

2. [Predication](https://github.com/Mixtape-Sessions/Machine-Learning/blob/main/Labs/python/Prediction.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mixtape-Sessions/Machine-Learning/blob/main/Labs/python/Prediction.ipynb)

3. [Causal via Predication](https://github.com/Mixtape-Sessions/Machine-Learning/blob/main/Labs/python/Causal%20via%20Prediction.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mixtape-Sessions/Machine-Learning/blob/main/Labs/python/Causal%20via%20Prediction.ipynb)
