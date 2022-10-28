capture log close
local output causalml
log using `output'.log, replace text
/*******************************************
PROGRAM: causalml.do
PROGRAMMER: Brigham Frandsen
PURPOSE: illustrate common ML prediction tools and their use in causal inference
strategies.
DESCRIPTION: The empirical example investigate earnings race gaps in the NLSY
DATE: October 28, 2022
*********************************************/

*load the NLSY97 data
import delimited using "https://github.com/Mixtape-Sessions/Machine-Learning/blob/main/Labs/data/nlsy97.csv?raw=true",clear

* initialize matrix to store results
mat results = J(2,6,.)
mat colnames results = OLSsimple OLSmultiple PDScv PDSplugin PDSauto DML
mat rownames results = estimate se

* simple regression, no controls
reg lnw_2016 black,robust
* save results
mat results[1,1]=(_b[black],_se[black])'

* multiple regression, basic controls
reg lnw_2016 black educ exp afqt
* save results
mat results[1,2]=(_b[black],_se[black])'

* now define comprehensive set of controls including background characteristics
* all variables except earnings and black:
unab allvars : *
local yd lnw_2016 black
local X : list allvars - yd
local numXs : list sizeof X
display "Number of Xs in dictionary: `numXs'"

* Let's do PDS Lasso with CV-chosen penalty (manually, so we can see each part)

* lasso the outcome on X
lasso linear lnw_2016 `X'
local Xy `e(allvars_sel)'
local ky = e(k_nonzero_sel)
display "num covs kept by outcome lasso: `ky'"
	
* lasso treatment on X
lasso linear black `X'
local Xd `e(allvars_sel)'
local kd = e(k_nonzero_sel)
display "num covs kept by treatment lasso: `kd'"

* regress outcome on treatment and union of controls
reg lnw_2016 black `Xy' `Xd',robust	
* save results
mat results[1,3]=(_b[black],_se[black])'

* Now same, but with plugin penalty
* lasso the outcome on X
lasso linear lnw_2016 `X',selection(plugin,het)
local Xy `e(allvars_sel)'
local ky = e(k_nonzero_sel)
display "num covs kept by outcome lasso: `ky'"
	
* lasso treatment on X
lasso linear black `X',selection(plugin,het)
local Xd `e(allvars_sel)'
local kd = e(k_nonzero_sel)
display "num covs kept by treatment lasso: `kd'"

* regress outcome on treatment and union of controls
reg lnw_2016 black `Xy' `Xd',robust	
* save results
mat results[1,4]=(_b[black],_se[black])'

* Now using Ahrens, Hansen, and Schaffer's pdslasso
pdslasso lnw_2016 black (`X'),robust
* A little easier, right?
* save results
mat results[1,5]=(_b[black],_se[black])'

* Now Double/De-biased machine learning (DML) based on a random forest

* randomly divide observations into 5 folds
gen rorder=runiform()
sort rorder
gen fold = mod(_n,5)

* initialize variables to mark which are training (fit) and test (estimation)
gen fitsample=.
gen estsample=.
* initialize variables to hold the ML resids
gen uhat=.
gen vhat=.

* loop through the folds		
forvalues fold = 0/4 {
	replace fitsample = fold!=`fold'
	replace estsample = fold==`fold'
	
	* fit outcome on training set
	rforest lnw_2016 `X' if fitsample==1,type(reg)
	* generate prediciton in test set
	predict yhat if estsample==1
	* populate residual variable for this fold
	replace uhat = lnw_2016-yhat if estsample==1
	
	* fit treatment on training set
	rforest black `X' if fitsample==1,type(reg)
	* generate prediction in test set
	predict dhat if estsample==1
	* populate residual variable for this fold
	replace vhat = black-dhat if estsample==1
	* clean up
	drop yhat dhat
}

* now to second-step regression
display "DDML results"
reg uhat vhat,robust
* save results
mat results[1,6]=(_b[vhat],_se[vhat])'

* display results
mat li results,noheader format(%10.3f)
log close