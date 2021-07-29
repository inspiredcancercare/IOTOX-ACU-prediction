# Pragmatic Machine learning algorithms offer robust and interpretable predictions of 90-day unplanned acute care use for cancer immunotherapy patients
Table of contents
=================
<!--ts-->
 * [eMethods](#emethods)
   * [Data preparation](#data-preparation)
   * [Model training and optimization](#model-training-and-optimization)
   * [Model calibration and risk threshold determination](#model-calibration-and-risk-threshold-determination)
   * [Model exmination and explanation](#model-exmination-and-explanation)
 * [Candidate predictors](#candidate-predictors)
 * [Model calibration plots](#model-calibration-plots)
 * [Variable importance plots](#variable-importance-plots)
 * [Shapely additive explanation plots](#shapely-additive-explanation-plots)
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

We carried out all data preparation steps using the [R recipes package version 0.1.16](https://cran.r-project.org/web/packages/recipes/recipes.pdf) with the following code after setting up the environment with necessary R packages using our [setEnvironment.R](https://github.com/inspiredcancercare/IOTOXACU/blob/ebd8db31e69fc480140ba78161109068a5273abe/setEnvironment.R).

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
 We trained four ML algorithms, including logistic regression with elastic net penalty (LRENP), random forest (RF), extreme gradient boosting trees (XGBT), and single hidden layer neural network (SHLNN), using the per-COVID training sample. We used the following steps to determine the best values of hyperparameters for each algorithm alongside a 10-fold cross-validation process. 
1. Trained and evaluated 30 initial models using a random search approach from predefined search spaces (eTable 1)
2. Fitted a Gaussian process (GP) model using initial models' hyperparameter values as predictors and model performance (AUROC) as outcomes
3. Projected potentially optimal hyperparameter values using the GP model
4. Created a new model with the hyperparameter values   
5. Compared the new model with the averaging AUROC of the initial models.
6. Repeated step 3-5 until 40 iterations or no model improvement in 20 consecutive models were reached

We used [R tidymodels package version 0.1.0](https://www.tidymodels.org/) to develop all models including the multivariate and univariate ECOG logistic regression. R code for all six models is available in [modeling.R](https://github.com/inspiredcancercare/IOTOXACU/blob/5451babcba443dd4e87b6e3a3b20889a641b6f91/modeling.R).

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
1. Applied the model to make predictions on the sample and extracted the predicted probabilities generated by the model
2. Fitted a logistic regression model to correlate the probabilities and observed outcomes (true classes) 
3. Used the logistic regression model to adjust the model outputs on the pre-COVID testing sample
 
To determine the risk threshold, we calculated sensitivity and specificity over risk thresholds from 0.001 to 1 using calibrated outputs on the pre-COVID testing sample for each algorithm. We selected the threshold which maximizes both sensitivity and specificity of each model. 

We provide our code for model calibration and risk threshold determination in [reclassified.R]()

eFigure 1. Sensitivity and specificity over acute care use risk thresholds from 0.001 to 1 for each algorithm

![image](https://user-images.githubusercontent.com/79476527/127394784-93f7fa17-5dcb-4ecc-92be-b4f3c1ed1e6b.png)


[Back to top](#table-of-contents)

Model exmination and explanation
--------------------------------
To examine model performance, we calculated performance metrics including area under the receiver operating charactistic curve (AUROC), accuracy, sensitivity, specificity, positive predictive value, negative predictive value, and confusion matrix for each model after calibration wth corresponding risk threshold. We caculated the metrics on the post-COVID testing and peri-COVID samples. We used McNamer's test to statistically compare our ML algorithms to two logistic regression-based models in terms of AUROC on both samples. We carried out the McNamer's test with 2000 stratified bootstrapping replications and an alpha level of 0.05. The code for performance examination and statistical comparison is available in [performance.R]()

In addition to the statistical examinations, we used two model agnostic approaches, variable importance analysis and Shapley additive explanation, to provide additional insights into model behaviors and information potentially enabling individualized preventive intervention provision. We used [R DELAX package version 1.2.1](https://github.com/ModelOriented/DALEX) to reveal how our models utilized data to generate predictions for the peri-COVID sample. 

For variable importance, we used the permutation approach to determine the importance of a predictor to an algorithm by measuring the prediction error fluctuations when randomly shuffling the predictor’s value. In this study, we use one minus AUROC as a function to measure the prediction error. A predictor with a larger value of prediction error is more important to an algorithm. We used the R code in [explanation.R]() to conduct the analysis. 

Another approach we used to explain our algorithms is an example-based explanation revealing model behavior on a particular instance of the peri-COVID sample. Specifically, we calculated Shapely values to determine the contributions of predictor values for a randomly selected instance in the sample to the model output. We focused on the predictors which are most important (top 10) for each algorithm. R code for the analysis is available in [explanation.R](). 

[Back to top](#table-of-contents)

Candidate predictors
=================================================

 <table>
    <thead>
        <tr>
            <th> Category </th>
            <th> Name </th>
	    <th> Value </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            	<td rowspan=6> Basic information </td>
	    	<td> Age </td>
		<td> Numeric </td>
            </tr>
		<tr>
			<td> Gender </td>
			<td> Male, Female </td>
		    </tr>
		<tr>
			<td> Marital status </td>
			<td> Legally separated, Married, Other, Significant other, Single, Widowed, Divorced </td>
		    </tr>
		<tr>
			<td> Smoking status </td>
			<td> Current Every Day Smoker, Current Some Day Smoker, Former Smoker, Heavy Tobacco Smoker, Light Tobacco Smoker, Never Smoker, Passive Smoke Exposure - Never Smoker, Smoker, Current Status Unknown   </td>
		    </tr>		
		<tr>
			<td> ECOG </td>
			<td> 0, 1, 2, 3, 4 </td>
		    </tr>
	   	<tr>
			<td> ADI national </td>
			<td> 0 - 99, GQ, GQ-PH, KVM, PH </td>
		    </tr>
	<tr>
	    	<td rowspan=8>Treatment</td>
		<td> RT 30 days before </td>
		<td> True, False </td>
	    </tr>
		<tr>
			<td> Atezolizumab </td>
			<td> True, False </td>
		    </tr>
		<tr>
			<td> Avelumab </td>
			<td> True, False </td>
		    </tr>
	   	<tr>
			<td> Cemiplimab-rwlc </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Durvalumab </td>
			<td> True, False </td>
		    </tr>
	   	<tr>
			<td> Ipilimumab </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Nivolumab </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Pembrolizumab </td>
			<td> True, False </td>
		    </tr>
        <tr>
		<td rowspan=35>Cormobidity</td>
		<td> Anxiety </td>
		<td> True, False </td>
        </tr>
		<tr>
			<td> Anxiety </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Asthma </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Atherosclerosis </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Atrial Fibrillaton </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Cardiac arrhythmia </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Cerebrovascular accident </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Cerebrovascular disease </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Chronic kidney disease </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Chronic obstructive pulmonary disease </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Chronic pulmonary disease </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Congestive heart failure </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Coronary artery disease </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Deep vein thrombosis </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Dementia </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Depression </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Diabetes mellitus </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> ESRD </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Gerd </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Hemiplegia paraplegia </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Hypertension </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Hypothyroidism </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Mild liver disease </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Moderate or severe liver disease </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Myocardial infarction </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Neuropathy </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Obesity </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Obstructive sleep apnea </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Peripheral vascular disease </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Psychosis </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Pulmonary hypertension </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Rheumatoid arthritis </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Leukopenia </td>
			<td> True, False </td>
		    </tr>
	    	<tr>
			<td> Thrombocytopenia </td>
			<td> True, False </td>
		    </tr>
		<tr>
			<td> low platelets or wbc </td>
			<td> True, False  </td>
		     </tr>
        <tr>
		<td rowspan=9>Vital sign</td>
		<td> Systolic blood pressure  </td>
		<td> Numeric </td>
        </tr>
		<tr>
			<td> Diastolic blood pressure </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Mean arterial pressure </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Body temperature </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Pulse </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Respiration </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> SPO2 </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> BMI </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Pain </td>
			<td> 0 - 10 </td>
		    </tr>
	     <tr>
		<td rowspan=53>Laboratory</td>
		<td> ALT  </td>
		<td> Numeric </td>
        </tr>
		<tr>
			<td> AST </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Albumin level </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Alkaline phosphatase </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> BUN </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Basophil abs </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Basophil pct </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> CO2 </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Calcium </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Chloride </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Creatinine </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Eosinophil abs </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Eosinophil pct </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Glucose level </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> HCT </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> HGB </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> IGRE pct </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> LDH </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Lymphocyte abs </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Lymphocyte pct </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> MCH </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> MCHC </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> MCV </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> MPV </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Monocyte abs </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Monocyte pct </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Neutrophil abs </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Neutrophil pct </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Platelet count </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Potassium level </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> RBC </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> RDW CV </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> RDW SD </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Sodium level </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Total protein </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> WBC </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> eGFR AA </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Creatinine clear </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Anion gap </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Bili direct </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Bili indirect </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Bili total </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> IG abs </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> INR </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> INRBC </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Magnesium </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> PT </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> PTT </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Phosphorus </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> T4 free </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> TSH </td>
			<td> Numeric </td>
		    </tr>
	    	<tr>
			<td> Uric Acid </td>
			<td> Numeric </td>
		    </tr>
	</tbody>
</table>

[Back to top](#table-of-contents)

Model calibration plots
-----------------------

We present calibration plots for all calibrated algorithms we examined on the peri-COVID sample in this section. The plot for the random forest algorithm is included in our main paper.

eFigure 2. Calibration plots for the multivariate logistice regression algorithm.

![image](https://user-images.githubusercontent.com/79476527/127537411-e0cd32ed-d0c1-4797-8f3f-2f54218f35de.png)

**Note**: The multivariate logistic regression model is well calibrated for patients with low to moderate risk of ACU. The model are likely to underestimat the risk for patients with high ACU risk.

eFigure 3. Calibration plots for the ECOG logistice regression algorithm.

![image](https://user-images.githubusercontent.com/79476527/127537556-40014b79-ea3a-4696-a84e-2db64535d159.png)



eFigure 4. Calibration plots for the logistice regression with elastic net panelty algorithm.

![image](https://user-images.githubusercontent.com/79476527/127537090-e9ea0400-87f7-4baa-836c-e956d6d0a9fb.png)


eFigure 5. Calibration plots for the extreme gradient boosting trees algorithm.

![image](https://user-images.githubusercontent.com/79476527/127537728-2fd9d45b-ebf8-4854-a8a6-93e67d7ac32d.png)


eFigure 6. Calibration plots for the single hidden layer neural network algorithm.

![image](https://user-images.githubusercontent.com/79476527/127537882-80710349-7e51-4eef-b448-91607d9aa3da.png)


[Back to top](#table-of-contents)

Variable importance plots
--------------------------

We present the result for the random forest algorithm in our main paper and the rest ML algorithms here.

eFigure 2. Important variables for logistic regression with elastic net penalty.

![image](https://user-images.githubusercontent.com/79476527/127511420-37ea9423-9f3e-480f-855c-0d7450e850a2.png)

eFigure 3. Important variables for extreme gradient boosting trees.

![image](https://user-images.githubusercontent.com/79476527/127511133-5fc6a51d-57dd-47ae-a1af-d547c30031b0.png)

eFigrue 4. Important variables for single hidden layer neural network.

![image](https://user-images.githubusercontent.com/79476527/127511237-14eba026-e4f3-41c6-b1c2-305bd774ef20.png)

[Back to top](#table-of-contents)


Shapely additive explanation plots
--------------------------

The result for the random forest algorithm is included in the main test, and here are the results for other algorithms.

eFigure 5. Contributions of predictor values to the prediction of the logistic regression with elastic net penalty algorithm for the randomly selected case.

![image](https://user-images.githubusercontent.com/79476527/127511846-5fd65032-b88c-4733-95f1-c799ec785950.png)

eFigrue 6. Contributions of predictor values to the prediction of the extreme gradient boosting trees algorithm for the randomly selected case.

![image](https://user-images.githubusercontent.com/79476527/127511884-a6f7d4a0-6396-4f58-aa91-1ca1120e6086.png)

eFigure 7. Contributions of predictor values to the prediction of the single hidden layer neural network algorithm for the randomly selected case.

![image](https://user-images.githubusercontent.com/79476527/127511867-1d527761-89fe-48a4-b198-f3e48133d8d2.png)


[Back to top](#table-of-contents)
