	Online supplementary materials and code repository
# Machine learning algorithms using routinely collected clinical data offer robust and interpretable predictions of 90-day unplanned acute care use for cancer immunotherapy
	
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
  <li>All predictors</li>
	<dd>- Removing predictors that contain only a single value or have most traning sample with the same value</dd>
	<dd>- Removing predictors that have large absolute correlations (>=0.90) with other predictors </dd>
</ul>

We carried out all data preparation steps using the [R recipes package version 0.1.16](https://cran.r-project.org/web/packages/recipes/recipes.pdf) with the following code after setting up the environment with necessary R packages using our [setEnvironment.R](https://github.com/inspiredcancercare/IOTOXACU/blob/ebd8db31e69fc480140ba78161109068a5273abe/setEnvironment.R).

```
my_seed<-2021

outcome ="has_90_day_readmit"

predictors<-names(df_train)
formula<-as.formula(paste(outcome, paste(predictors, collapse = "+"), sep="~"))
print("formula:")
print(formula)
log_trans_variable<- c("lab_AST","lab_ALT","lab_ALK_Phos","lab_BUN","lab_Basophilab_Abs", "lab_Basophilab_pct",
	"lab_Chloride", "lab_Creatinine", "lab_Eosinophilab_pct","lab_Eosinophilab_Abs","lab_Platelet", 
	"lab_Glucose_Lvl", "lab_IGRE_pct",  "lab_LDH", "lab_Lymphocyte_Abs", "lab_Lymphocyte_pct", "lab_Monocyte_Abs",
	"lab_Neutrophilab_Abs",  "lab_Neutrophilab_pct","lab_RBC","lab_RDW_CV", "lab_RDW_SD", "lab_eGFR_AA", 
	"lab_creatinine_clear",   "lab_Bili_Direct", "lab_Bili_Indirect", "lab_Bili_Total","lab_IG_Abs","lab_INR",
	"lab_INRBC", "lab_PT","lab_TSH", "lab_WBC","vit_pulse","vit_BMI","vit_spo2","vit_resp")

ordinal_varaible<-c("vit_pain","ecog_latest", "ADI_NATRANK")

##### split data into training and testing samples ####
set.seed(my_seed)
split<-initial_split(db, prop = .8) #where db is the pre-COVID sample
df_train<-training(split)
df_test<-testing(split)

##### set a recipe for data prepraration ####
df_recipe<-recipe(formula, df_train) %>%
  step_medianimpute(all_numeric(), -recipes::all_outcomes()) %>%
  step_modeimpute(all_nominal(), -recipes::all_outcomes()) %>%
  step_log(!!!syms(log_trans_variable), offset = 1) %>%
  step_YeoJohnson(lab_Anion_Gap) %>%
  step_other(marital, smoke, threshold = 0.1, other = "other") %>% 
  step_integer(!!ordinal_varaible) %>%
  step_normalize(all_numeric(),-!!ordinal_varaible,-recipes::all_outcomes()) %>%
  step_zv(all_predictors(),-recipes::all_outcomes()) %>%
  step_nzv(all_predictors(),-recipes::all_outcomes()) %>%
  step_dummy(all_nominal(),-recipes::all_outcomes(), one_hot=FALSE, id="dummy") %>%
  step_corr(recipes::all_predictors(),-recipes::all_outcomes())
  
#### Create resample ####
set.seed(my_seed)
cvfolds<-vfold_cv(df_train, v = 10, strata = outcome)

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

We used [R tidymodels package version 0.1.0](https://www.tidymodels.org/) to develop all models including the multivariate and univariate logistic regression using Eastern Cooperative Oncology Group (ECOG) as the only predictor. No model optimization was performed for the the multivariate and univariate ECOG logistic regression R code for all six models is available in [modeling.R](https://github.com/inspiredcancercare/IOTOXACU/blob/5451babcba443dd4e87b6e3a3b20889a641b6f91/modeling.R).

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
		<td> 0.0766 </td>
            </tr>
        <tr>
		<td> mixture </td>
		<td> 0 - 1 </td>
		<td> 0.0624 </td>
            </tr>
	<tr>
	    	<td rowspan=3>RF</td>
		<td> mtry </td>
		<td> 1 - 83 </td>
		<td> 1 </td>
	    </tr>
	<tr>
		<td> trees </td>
		<td> 1 - 4500 </td>
		<td> 1500 </td>
	    </tr>
	<tr>
		<td> min_n </td>
		<td> 2 - 700 </td>
		<td> 291 </td>
	    </tr>
        <tr>
		<td rowspan=7>XGBT</td>
		<td> mtry </td>
		<td> 1 - 83 </td>
		<td> 8 </td>
        </tr>
	<tr>
		<td> trees </td>
		<td> 1 - 4000 </td>
		<td> 1003 </td>
	    </tr>
	<tr>
		<td> min_n </td>
		<td> 1 - 75 </td>
		<td> 11 </td>
	    </tr>
	<tr>
		<td> tree_depth </td>
		<td> 1 - 150 </td>
		<td> 14 </td>
	    </tr>
	 <tr>
		<td> learn_rate </td>
		<td> 10^-17 - 10^-1 </td>
		<td> 0.000000014 </td>
	    </tr>
	 <tr>
		<td> loss_reduction </td>
		<td> 10^-17 - 10^1.5 </td>
		<td> 0.0000000000154 </td>
	    </tr> 
	<tr>
		<td> sample_size </td>
		<td> 0.1 - 1 </td>
		<td> 0.763 </td>
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

We provide our code for model calibration and risk threshold determination in [reclassified.R](https://github.com/inspiredcancercare/IOTOXACU/blob/main/R%20code/reclassification.R)


<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/154921420-2c32f8e6-217f-4ef0-bd20-00f65d905a00.png">
</p>

eFigure 1. Sensitivity and specificity over acute care use risk thresholds from 0.001 to 1 for each algorithm

[Back to top](#table-of-contents)

Model exmination and explanation
--------------------------------
To examine model performance, we calculated performance metrics including area under the receiver operating charactistic curve (AUROC), accuracy, sensitivity, specificity, positive predictive value, negative predictive value, and confusion matrix for each model after calibration wth corresponding risk threshold. We caculated the metrics on the post-COVID testing and peri-COVID samples. We used McNamer's test to statistically compare our ML algorithms to two logistic regression-based models in terms of AUROC on both samples. We carried out the McNamer's test with 2000 stratified bootstrapping replications and an alpha level of 0.05. The code for performance examination and statistical comparison is available in [performance.R](https://github.com/inspiredcancercare/IOTOXACU/blob/be49e816355b7b78b1748eaa497aa85e00fe8c0a/R%20code/performance.R)

In addition to the statistical examinations, we used two model agnostic approaches, variable importance analysis and Shapley additive explanation, to provide additional insights into model behaviors and information potentially enabling individualized preventive intervention provision. We used [R DELAX package version 1.2.1](https://github.com/ModelOriented/DALEX) to reveal how our models utilized data to generate predictions for the peri-COVID sample. 

For variable importance, we used the permutation approach to determine the importance of a predictor to an algorithm by measuring the prediction error fluctuations when randomly shuffling the predictor’s value. In this study, we use one minus AUROC as a function to measure the prediction error. A predictor with a larger value of prediction error is more important to an algorithm. We used the R code in [explanation.R](https://github.com/inspiredcancercare/IOTOXACU/blob/be49e816355b7b78b1748eaa497aa85e00fe8c0a/R%20code/explanation.R) to conduct the analysis. 

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
	    	<td rowspan= 6 >Treatment</td>
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
	</tbody>
</table>


[Back to top](#table-of-contents)

Model calibration plots
-----------------------

We present calibration plots for all calibrated algorithms we examined on the peri-COVID sample in this section. The plot for the LRENP algorithm is included in our main paper.


<p align="center">
  <img width="460" height="400" src="https://user-images.githubusercontent.com/38151091/151495499-6a0fb6cb-dd86-4bf9-9026-c97dd0aefeb2.png">
</p>

eFigure 2. Calibration plot for the ECOG logistic regression algorithm.

**Note:** The lack of alignment between the solid and dash lines in most areas indicates that the model is not well calibrated.

<p align="center">

  <img width="460" height="400" src="https://user-images.githubusercontent.com/38151091/151475184-4cdf832a-b6ad-4805-b588-5e854410568a.png">
</p>

eFigure 3. Calibration plot for the ranfom forest algorithm.

**Note:** The calibration analysis result for the random forest is suboptimal. The model tends to overestimate the risk for patients with low risk (20%) and underestimate the rsik for patients with moderate to high risk of ACU (>=50%).

<p align="center">
  <img width="460" height="400" src="https://user-images.githubusercontent.com/38151091/151476847-c03ee632-09e7-4c88-b0ef-eb51afec0269.png">
</p>

eFigure 4. Calibration plot for the extreme gradient boosting tree algorithm.

**Note:** The calibration of the extreme gradient boosting trees algorithm is relatively well-balibrated for patients with low to moderate risk of ACU (<= 50%). The model is unstable for patients in high risk area (>50%).

<p align="center">
  <img width="460" height="400" src="https://user-images.githubusercontent.com/38151091/151476988-1777145e-a82e-4778-82a4-175e1ccede0a.png">
</p>

eFigure 5. Calibration plot for the single hidden layer neural network algorithm.

**Note:** The calibration plot suggests that the model generates reliable outputs only for patients with a 25% to 60 % of AUC risk and tends to underestimate the risk for other patients.


[Back to top](#table-of-contents)

Variable importance plots
--------------------------

We present the result for the LRENP algorithm in our main paper and the rest ML algorithms here.

<p align="center">
  <img width="460" height="400" src="https://user-images.githubusercontent.com/38151091/151477070-b7e2ffea-d128-4b41-8a41-0c1b309fd8e1.png">
</p>

eFigure 6. Important variables for the random forest algorithm.

<p align="center">
  <img width="460" height="400" src="https://user-images.githubusercontent.com/38151091/151477159-d51cdd9e-5b90-4607-8748-f0c8bea38855.png">
</p>

eFigure 7. Important variables for the extreme gradient boosting trees algorithm.

<p align="center">
  <img width="460" height="400" src="https://user-images.githubusercontent.com/38151091/151477347-0608af07-5c35-4bb7-a32a-c063b5ba17cb.png">
</p>

eFigrue 8. Important variables for the single hidden layer neural network algorithm.


[Back to top](#table-of-contents)



Shapely additive explanation plots
--------------------------

The result for the LRENP algorithm is included in the main test, and here are the results for other algorithms.

<p align="center">
  <img width="460" height="400" src="https://user-images.githubusercontent.com/38151091/151478003-b6261d25-aa85-446d-9c1b-e5c9be3c86d5.png">
</p>

eFigure 9. Contributions of predictor values to the prediction of the random forest algorithm for the randomly selected case.

<p align="center">
  <img width="460" height="400" src="https://user-images.githubusercontent.com/38151091/151477936-1214a44f-0ff5-4548-9e14-e93dd63cb9e9.png">
</p>

eFigrue 10. Contributions of predictor values to the prediction of the extreme gradient boosting trees algorithm for the randomly selected case.

<p align="center">
  <img width="460" height="400"  src="https://user-images.githubusercontent.com/38151091/151478076-1a389c00-f0bc-412f-b100-046a6250a6bc.png">
</p>

eFigure 11. Contributions of predictor values to the prediction of the single hidden layer neural network  algorithm for the randomly selected case.



[Back to top](#table-of-contents)
