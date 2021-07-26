# Machine learning algorithms predicting acute care use for patients within 90-day of immune checkpoint inhibitors
## eMethod:Detailed data preparation, model optimization, and examination
### data preparation
We applied the following data preparation techniques to prepare the data before modeling according to thorough observations on the pre-coronavirus disease (COVID) training sample. 
<ul>
  <li>Numeric predictors</li>
		<dd>- Log or Yeo-Johnson transformation to enhance normality</dd>
		<dd>-	Rescaling the predictors to make all predictors having 0 mean and standard deviation 1</dd>
		<dd>-	Missing data imputation using median of each predictor </dd>
  <li>Categorical predictors</li>
  <li>Ordinal predictors</li>
	<li>All predictors</li>
</ul>
