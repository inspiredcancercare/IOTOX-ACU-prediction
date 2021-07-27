# Machine learning algorithms predicting acute care use for patients within 90-day of immune checkpoint inhibitors
## eMethod:Detailed data preparation, model optimization, and examination
### data preparation
We applied the following data preparation techniques to prepare the data before modeling according to thorough observations on the pre-coronavirus disease (COVID) training sample. 
<ul>
  <li>Numeric predictors</li>
		<dd>- Log or Yeo-Johnson transformation to enhance normality</dd>
		<dd>- Rescaling the predictors to make all predictors having 0 mean and standard deviation 1</dd>
		<dd>- Missing data imputation using median of each predictor </dd>
  <li>Categorical predictors</li>
		<dd>- Lumping classes occurring in fewer than 10% of the training sample to an “Other” category for each predictor to reduce data complexity</dd>
		<dd>- One-hot encoding to convert each predictor into a binary term for each class of the original data</dd>
		<dd>- Missing data imputation using mode of each predictor </dd>
  <li>Ordinal predictors</li>
	<li>All predictors</li>
</ul>

We carried our all data preparation steps using the [R recipes package version 0.1.16](https://cran.r-project.org/web/packages/recipes/recipes.pdf) with the following code after setting up the environment with necessary R pachages using our [setEnvironment.R]().

```
#-----load package------------
packs <- c("tidyverse", "caret", "e1071", "recipes","tidymodels","kernlab","themis",
           "pROC","recipes", "ranger", "xgboost","nnet", "DALEX")

InstIfNec<-function (pack) {
  if (!do.call(require,as.list(pack))) {
    do.call(install.packages,as.list(pack)) }
  do.call(require,as.list(pack)) }
lapply(packs, InstIfNec)
```
