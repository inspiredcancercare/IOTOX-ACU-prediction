# Machine learning algorithms predicting acute care use for patients within 90-day of immune checkpoint inhibitors
Table of contents
=================
<!--ts-->
 * [eMethods](#emethods)
   * [Data preparation](#data-preparation)
   * [Model training and optimization](#model-training-and-optimization)
   * [Model calibration and risk threshold determination](#model-calibration-and-risk-threshold-determination)
   * [Model ] 
 * Candidate predictors used
 * Figures for variable importance analysis
 * Figures for Shapley additive explanation analysis
<!--te-->

eMethods
==============================================================
Data preparation
----------------
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

We carried our all data preparation steps using the [R recipes package version 0.1.16](https://cran.r-project.org/web/packages/recipes/recipes.pdf) with the following code after setting up the environment with necessary R pachages using our [setEnvironment.R](https://github.com/inspiredcancercare/IOTOXACU/blob/ebd8db31e69fc480140ba78161109068a5273abe/setEnvironment.R).

```
my_seed<-2021

outcome ="has_90_day_readmit"

predictors<-names(df_train)
formula<-as.formula(paste(outcome, paste(predictors, collapse = "+"), sep="~"))
print("formula:")
print(formula)
log_trans_variable<- c("l_AST", "l_ALT",  "l_Alk_Phos", "l_BUN", "l_Basophil_Abs",
                       "l_Basophil_pct", "l_Chloride", "l_Creatinine", "l_Eosinophil_pct",
                       "l_Eosinophil_Abs", "l_Platelet", "l_Glucose_Lvl", "l_IGRE_pct",
                       "l_LDH", "l_Lymphocyte_Abs", "l_Lymphocyte_pct", 
                       "l_Monocyte_Abs", "l_Monocyte_pct", "l_Neutrophil_Abs",
                       "l_Neutrophil_pct", "l_RBC","l_RDW_CV", "l_RDW_SD", 
                       "l_eGFR_AA", "l_creatinine_clear", "l_Bili_Direct", 
                       "l_Bili_Indirect",  "l_Bili_Total", "l_IG_Abs",
                       "l_INR", "l_INRBC", "l_PT", "l_PTT", "l_TSH","l_T4_Free", 
                       "l_WBC","v_spo2","v_resp")

ordinal_varaible<-c("v_pain","ecog_latest", "ADI_NATRANK")

##### split data into training and testing samples ####
set.seed(my_seed)
split<-initial_split(db, prop = .8) #where db is the pre-COVID sample
df_train<-training(split)
df_test<-testing(split)

##### set a recipe for data prepraration ####
df_recipe<-recipe(formula, df_train) %>%
  step_medianimpute(all_numeric(), -recipes::all_outcomes()) %>%
  step_modeimpute(all_nominal(), -recipes::all_outcomes()) %>%
  step_log(!!log_trans_variable, offset = 1) %>%
  step_YeoJohnson(l_Anion_Gap) %>%
  step_other(marital_status, smoking, threshold = 0.1, other = "other") %>% 
  step_integer(!!ordinal_varaible) %>%
  step_normalize(all_numeric(),-!!ordinal_varaible,-recipes::all_outcomes()) %>%
  step_zv(all_predictors(),-recipes::all_outcomes()) %>%
  step_nzv(all_predictors(),-recipes::all_outcomes()) %>%
  step_dummy(all_nominal(),-recipes::all_outcomes(), one_hot=FALSE, id="dummy") %>%
  step_corr(recipes::all_predictors(),-recipes::all_outcomes())
  
#### Create resample ####
set.seed(my_seed)
cvfolds<-vfold_cv(df_train, v = 10, strata = outcome)
write_rds_file(cvfolds,"cvfolds.RDS")
```
 [back to top](#table-of-contents)
 
 Model training and optimization
 -------------------------------
 We trained four ML algorithms, including logistic regression with elastic net penalty (LRENP), random forest (RF), extreme gradient boosting trees (XGBT), and single hidden layer neural network (SHLNN), using the per-COVID training sample. We used the following steps to determine the best values of hyperparameters for each algorithm alongside 10-fold cross validation process. 
1. Trained and evaluated 30 inital models using random search approach from presdefined search spaces (eTable 1)
2. Fitted a Gaussian process (GP) model using inital models' hyperparameter values as predictors and model performance (AUROC) as outocmes
3. Projected a potentially optimal hyperparatemer values using the GP model
4. Created a new model with the hyperparameter values   
5. Compared the new model with the averaging AUROC of the inital models.
6. Repeated step 3-5 until 40 iterations or no model improvement in 20 consecutive models were reach

We used [R tidymodels package version 0.1.0](https://www.tidymodels.org/) to develop all models including the multivariate and univariate ECOG logistic regression. R code for all six models is avaialbe in [modeling.R](https://github.com/inspiredcancercare/IOTOXACU/blob/5451babcba443dd4e87b6e3a3b20889a641b6f91/modeling.R).

 eTable 1. Hyperparameters for each algorithm, corresponding search spaces, and optimal values
 <table>
    <thead>
        <tr>
            <th> ML algorithm </th>
            <th> Hyperparameter </th>
            <th> Search spaces </th>
	    <th> Value selected </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            	<td rowspan=2>LRENP</td>
	    	<td> penalty</td>
		<td> 10^-16 ~ 10^0 </td>
		<td> 0.00000000159 </td>
            </tr>
        <tr>
		<td> mixture </td>
		<td> 0 - 1 </td>
		<td> 0.00437 </td>
            </tr>
	<tr>
	    	<td rowspan=3>RF</td>
		<td> mtry </td>
		<td> 1 - 81 </td>
		<td> 73 </td>
	    </tr>
	<tr>
		<td> trees </td>
		<td> 500 - 4500 </td>
		<td> 1138 </td>
	    </tr>
	<tr>
		<td> min_n </td>
		<td> 2 - 700 </td>
		<td> 490 </td>
	    </tr>
        <tr>
		<td rowspan=7>XGBT</td>
		<td> mtry </td>
		<td> 1 - 81 </td>
		<td> 27 </td>
        </tr>
	<tr>
		<td> trees </td>
		<td> 1 - 4000 </td>
		<td> 472 </td>
	    </tr>
	<tr>
		<td> min_n </td>
		<td> 2 - 75 </td>
		<td> 4 </td>
	    </tr>
	<tr>
		<td> tree_depth </td>
		<td> 1 - 150 </td>
		<td> 13 </td>
	    </tr>
	 <tr>
		<td> learn_rate </td>
		<td> 10^-17 - 10^-1 </td>
		<td> 0.000596 </td>
	    </tr>
	 <tr>
		<td> loss_reduction </td>
		<td> 10^-17 - 10^1.5 </td>
		<td> 0.00000195 </td>
	    </tr> 
	<tr>
		<td> sample_size </td>
		<td> 0.1 - 1 </td>
		<td> 0.916 </td>
	    </tr> 
        <tr>
		<td rowspan=3>SHLNN</td>
		<td> hidden_units </td>
		<td> 1 - 20 </td>
		<td> 1 </td>
        </tr>
	<tr>
		<td> penalty </td>
		<td> 10^-10 - 10^2 </td>
		<td> 0.764 </td>
	    </tr>
	 <tr>
		 <td> epochs </td>
		 <td> 1 - 1500 </td>
		 <td> 926 </td>
	    </tr>
    	<tr>
		<td colspan=4> <b>Abbreviations:</b> LRENP: logistic regression with elastic net penalty; RF: random forest; XGBT: extreme gradient boosting trees; SHLNN: single hidden layer neural network.</td>
	    </tr>
	</tbody>
</table>

[Back to top](#table-of-contents)

Model calibration and risk threshold determination
--------------------------------------------------
After creating ML models, we use Platt scaling method to create a calibration model using the pre-COVID training sample following steps below for each model:
1. Applied the model to make predictions on the sample and extrated the predicted probabilities generated by the model
2. Fitted a logistic regression model to correlate the probabilites and observed outcomes (true classes) 
3. Used the logistic regression model to adjust the model outputs on the pre-COVID testing sample
 
To determine the risk threshold, we caculated sensitivity and specificity over risk thresholds from 0.001 to 1 using calibrated optputs on the pre-COVID testing sample for each algorithm. We selected the threshold which maximize both sensitivity and specificity of each model. 

We provide our code for model calibration and risk threshold determination in [reclassified.R]()

eFigure 1. Sensitivity and specificity over acute care use risk thresholds from 0.001 to 1 for each algorithm

***[TBA]***

[Back to top](#table-of-contents)

Candidate predictors
=================================================

 <table>
    <thead>
        <tr>
            <th> Category </th>
            <th> Name </th>
            <th> Type </th>
	    <th> Value </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            	<td rowspan=6> Basic information </td>
	    	<td> Age </td>
		<td> Numeric </td>
		<td> TBA </td>
            </tr>
		<tr>
			<td> Gender </td>
			<td> Nominal </td>
			<td> Male, Female </td>
		    </tr>
		<tr>
			<td> Marital status </td>
			<td> Nomial </td>
			<td> Legally separated, Married, Other, Significant other, Single, Widowed, Divorced </td>
		    </tr>
		<tr>
			<td> Smoking status </td>
			<td> Nomial </td>
			<td> Current Every Day Smoker, Current Some Day Smoker, Former Smoker, Heavy Tobacco Smoker, Light Tobacco Smoker, Never Smoker, Passive Smoke Exposure - Never Smoker, Smoker, Current Status Unknown   </td>
		    </tr>		
		<tr>
			<td> ECOG </td>
			<td> Ordinal </td>
			<td> 0, 1, 2, 3, 4 </td>
		    </tr>
	   	<tr>
			<td> ADI national </td>
			<td> Ordinal </td>
			<td> 0 - 99, GQ, GQ-PH, KVM, PH </td>
		    </tr>
	<tr>
	    	<td rowspan=8>Treatment</td>
		<td> RT 30 days before </td>
		<td> Nominal</td>
		<td> True, False </td>
	    </tr>
		<tr>
			<td> Atezolizumab </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
		<tr>
			<td> Avelumab </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	   	<tr>
			<td> Cemiplimab-rwlc </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Durvalumab </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	   	<tr>
			<td> Ipilimumab </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Nivolumab </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Pembrolizumab </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
        <tr>
		<td rowspan=34>Cormobidity</td>
		<td> Anxiety </td>
		<td> Nominal </td>
		<td> True, False </td>
        </tr>
		<tr>
			<td> Anxiety </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Asthma </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Atherosclerosis </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Atrial Fibrillaton </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Cardiac arrhythmia </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Cerebrovascular accident </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Cerebrovascular disease </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Chronic kidney disease </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Chronic obstructive pulmonary disease </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Chronic pulmonary disease </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Congestive heart failure </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Coronary artery disease </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Deep vein thrombosis </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Dementia </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Depression </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Diabetes mellitus </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> ESRD </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Gerd </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Hemiplegia paraplegia </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Hypertension </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Hypothyroidism </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Mild liver disease </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Moderate or severe liver disease </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Myocardial infarction </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Neuropathy </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Obesity </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Obstructive sleep apnea </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Peripheral vascular disease </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Psychosis </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Pulmonary hypertension </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Rheumatoid arthritis </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Leukopenia </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Thrombocytopenia </td>
			<td> Nominal </td>
			<td> True, False </td>
		    </tr>	    	   
        <tr>
		<td rowspan=9>Vital sign</td>
		<td> Systolic blood pressure  </td>
		<td> Numeric </td>
		<td> TBA </td>
        </tr>
		<tr>
			<td> Diastolic blood pressure </td>
			<td> Numeric </td>
			<td> TBA  </td>
		    </tr>
	    	<tr>
			<td> Mean arterial pressure </td>
			<td> Numeric </td>
			<td> TBA </td>
		    </tr>
	    	<tr>
			<td> Body temperature </td>
			<td> Numeric </td>
			<td> TBA </td>
		    </tr>
	    	<tr>
			<td> Pulse </td>
			<td> Numeric </td>
			<td> TBA </td>
		    </tr>
	    	<tr>
			<td> Respiration </td>
			<td> Numeric </td>
			<td> TBA </td>
		    </tr>
	    	<tr>
			<td> SPO2 </td>
			<td> Numeric </td>
			<td> TBA </td>
		    </tr>
	    	<tr>
			<td> BMI </td>
			<td> Numeric </td>
			<td> TBA </td>
		    </tr>
	    	<tr>
			<td> Pain </td>
			<td> Ordinal </td>
			<td> 0 - 10 </td>
		    </tr>
	</tbody>
</table>
