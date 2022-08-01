	Online supplementary materials and code repository
# Machine learning algorithms using routinely collected clinical data offer robust and interpretable predictions of 90-day unplanned acute care use for cancer immunotherapy patients.
	
Table of contents
=================
<!--ts-->
 * [eMethods](#emethods)
   * [Data preparation](#data-preparation)
   * [Model training and optimization](#model-training-and-optimization)
   * [Model calibration and risk threshold determination](#model-calibration-and-risk-threshold-determination)
   * [Model exmination and explanation](#model-exmination-and-explanation)
 * [Candidate predictors](#candidate-predictors)
 * [Sample characteristics](#sample-characteristics)
 * [Model performance on pre COVID testing sample](#model-performance-on-pre-covid-testing-sample)
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
            	<td rowspan=7>XGBT</td>
	    	<td> mtry</td>
		<td> 1 - 298 </td>
		<td> 21 </td>
            </tr>
        <tr>
		<td> trees </td>
		<td> 1 - 2000 </td>
		<td> 1799 </td>
            </tr>
        <tr>
		<td> min_n </td>
		<td> 2 - 40 </td>
		<td> 34 </td>
            </tr>
        <tr>
		<td> tree_depth </td>
		<td> 1 - 15 </td>
		<td> 12 </td>
            </tr>
        <tr>
		<td> learn_rate </td>
		<td>10^-10 – 10^-1</td>
		<td>0.00558 </td>
            </tr>
        <tr>
		<td> loss_reduction </td>
		<td> 10^-10 – 10^1.5 </td>
		<td> 0.00123 </td>
            </tr>
        <tr>
		<td> sample_size </td>
		<td> 0.1 - 1 </td>
		<td> 1036 </td>
            </tr>
	<tr>
	    	<td rowspan=3>RF</td>
		<td> mtry </td>
		<td> 1 - 298 </td>
		<td> 16 </td>
	    </tr>
	<tr>
		<td> trees </td>
		<td> 1 - 2500 </td>
		<td> 1036 </td>
	    </tr>
	<tr>
		<td> min_n </td>
		<td> 1 - 40 </td>
		<td> 7 </td>
	    </tr>
        <tr>
		<td rowspan=2>SVM</td>
		<td> cost </td>
		<td> 10^-15 – 10^-1 </td>
		<td> 0.000516 </td>
        </tr>
<tr>
		<td> rbf_sigma </td>
		<td> 0.0013 – 0.00234 </td>
		<td> 0.00221 </td>
	    </tr>
        <tr>
		<td rowspan=2>LRENP</td>
		<td> penalty </td>
		<td> 10^-10 – 10^0 </td>
		<td> 0.18 </td>
        </tr>
<tr>
		<td> mixture </td>
		<td> 0 - 1 </td>
		<td> 0.0516 </td>
	    </tr>
        <tr>
		<td rowspan=3>SHLNN</td>
		<td> hidden_units </td>
		<td> 1 - 50 </td>
		<td> 1 </td>
        </tr>
	<tr>
		<td> penalty </td>
		<td> 10^-10 - 10^1 </td>
		<td> 9.16 </td>
	    </tr>
	 <tr>
		 <td> epochs </td>
		 <td> 1 - 2000 </td>
		 <td> 1793 </td>
	    </tr>
        <tr>
		<td rowspan=1>KNN</td>
		<td> neighbors </td>
		<td> 1 - 500 </td>
		<td> 500 </td>
        </tr>
        <tr>
		<td rowspan=2>MARS</td>
		<td> num_terms </td>
		<td> 1 - 298</td>
		<td> 18 </td>
        </tr>
<tr>
		 <td> prod_degree </td>
		 <td> 1 - 3 </td>
		 <td> 3 </td>
	    </tr>
<tr>
		<td rowspan=3>DT</td>
		<td> cost_complexity </td>
		<td> 10^-10 – 10^-1</td>
		<td> 00.000242 </td>
        	</tr>
<tr>
		 <td> tree_depth </td>
		 <td> 1 - 15 </td>
		 <td> 2 </td>
	    </tr>
<tr>
		 <td> min_n </td>
		 <td> 2 - 40 </td>
		 <td> 7 </td>
	    </tr>
    	<tr>
		<td colspan=4> <b>Abbreviations:</b> ; DT: decision tree; KNN: k-nearest neighbors; LR: logistic regression; LRENP: logistic regression with elastic net penalty; MARS: multivariate adaptive regression splines; RF: random forest; SHLNN: single hidden layer neural network; SVM: support vector machine; XGBT: extreme gradient boosting trees.</td>
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
    <th>Category</th>
    <th>Name</th>
    <th>Value</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="7">Basic information</td>
    <td>Age</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Gender</td>
    <td>Male, Female</td>
  </tr>
  <tr>
    <td>Marital status</td>
    <td>Legally separated, Married,&nbsp;&nbsp;&nbsp;Other, Significant other, Single, Widowed, Divorced</td>
  </tr>
  <tr>
    <td>Smoking status</td>
    <td>Current Every Day Smoker,&nbsp;&nbsp;&nbsp;Current Some Day Smoker, Former Smoker, Heavy Tobacco Smoker, Light Tobacco&nbsp;&nbsp;&nbsp;Smoker, Never Smoker, Passive Smoke Exposure - Never Smoker, Smoker, Current&nbsp;&nbsp;&nbsp;Status Unknown</td>
  </tr>
  <tr>
    <td>Alcohol use</td>
    <td>Yes, Not current, Never, No</td>
  </tr>
  <tr>
    <td>ECOG</td>
    <td>0, 1, 2, 3, 4</td>
  </tr>
  <tr>
    <td>Area deprivation index -&nbsp;&nbsp;&nbsp;national</td>
    <td>0 - 99, GQ, GQ-PH, KVM, PH</td>
  </tr>
  <tr>
    <td rowspan="2">Treatment</td>
    <td>RT 30 days before</td>
    <td>True, False</td>
  </tr>
  <tr>
    <td>Immune checkpoint inhibitor agent</td>
    <td>Avelumab, Durvalumab, Nivolumab, Pembrolizumab, Cemiplimab-rwlc,&nbsp;&nbsp;&nbsp;Atezolizumab</td>
  </tr>
  <tr>
    <td rowspan="3">Cancer information</td>
    <td>Site</td>
    <td>Nominal </td>
  </tr>
  <tr>
    <td>Histology</td>
    <td>Nominal </td>
  </tr>
  <tr>
    <td>Metastasis status</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td rowspan="41">Medical history</td>
    <td>Anxiety</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Asthma</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Atrioventricular septal defect</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Atrial fibrillaton</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Bronchiolitis obliterans organizing pneumonia</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Cardiac arrhythmia</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Cerebrovascular accident</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Cerebrovascular disease</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Chronic kidney disease</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Chronic obstructive pulmonary disease</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Chronic pulmonary disease</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Congestive heart failure</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Coronary artery disease</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Deep vein thrombosis</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Dementia</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Depression</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Diabetes insipidus</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Diabetes mellitus</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>ESRD</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>ESRD SF</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Gerd</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Hemiplegia Paraplegia</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Human immunodeficiency virus</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Hypertension</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Hypogammaglobulinemia</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Hypothyroidism</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Immune related adverse events pneumonitis</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Cryptogenic organizing pneumonia</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Liver disease</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Mild liver disease</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Moderate or severe liver disease</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Myocardial infarction</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Neuropathy</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Obesity</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Obstructive sleep apnea</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Peripheral vascular disease</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Pneumonia</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Psychosis</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Pulmonary embolism</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Pulmonary hypertension</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td>Rheumatoid arthritis</td>
    <td>Yes, No</td>
  </tr>
  <tr>
    <td rowspan="65">Laboratory</td>
    <td>ALT last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>AST last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Albumin Lvl last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Alk Phos last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Anion Gap last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>BUN last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Basophil Abs last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Basophil pct last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Bili Direct last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Bili Indirect last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Bili Total last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>CO2 last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Calcium Lvl last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Chloride last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Creatinine last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Eosinophil Abs last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Eosinophil pct last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Glucose Level last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Hct last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Hgb last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>IGRE pct last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>IG Abs last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>INRBC last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>LDH last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Lymphocyte Abs last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Lymphocyte pct last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>MCHC last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>MCH last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>MCV last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>MPV last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Magnesium last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Monocyte Abs last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Monocyte pct last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Neutrophil Abs last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Neutrophil pct last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Phosphorus last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Platelet count last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Potassium Lvl last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>RBC last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>RDW CV last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>RDW SD last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Sodium Lvl last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>T4 Free last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>TSH last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Total Protein last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>WBC last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Creatinine clear last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>eGFR AA last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Cortisol last, mean,SD</td>
    <td>Removed due to high missing rate</td>
  </tr>
  <tr>
    <td>Free T3 last, mean,SD</td>
    <td>Removed due to high missing rate</td>
  </tr>
  <tr>
    <td>INR last, mean, SD</td>
    <td>Removed due to high missing rate</td>
  </tr>
  <tr>
    <td>Lipase Lvl last, mean,SD</td>
    <td>Removed due to high missing rate</td>
  </tr>
  <tr>
    <td>POC Crea last, mean, SD</td>
    <td>Removed due to high missing rate</td>
  </tr>
  <tr>
    <td>POC Glucose last, mean, SD</td>
    <td>Removed due to high missing rate</td>
  </tr>
  <tr>
    <td>POC creatinine clear last, mean, SD</td>
    <td>Removed due to high missing rate</td>
  </tr>
  <tr>
    <td>T3 Total last, mean, SD</td>
    <td>Removed due to high missing rate</td>
  </tr>
  <tr>
    <td>Total Cells last, mean, SD</td>
    <td>Removed due to high missing rate</td>
  </tr>
  <tr>
    <td>UA Protein last, mean SD</td>
    <td>Removed due to high missing rate</td>
  </tr>
  <tr>
    <td>UA RBC last, mean, SD</td>
    <td>Removed due to high missing rate</td>
  </tr>
  <tr>
    <td>UA Spec Grav last, mean, SD</td>
    <td>Removed due to high missing rate</td>
  </tr>
  <tr>
    <td>UA WBC last, mean, SD</td>
    <td>Removed due to high missing rate</td>
  </tr>
  <tr>
    <td>UA pH last, mean, SD</td>
    <td>Removed due to high missing rate</td>
  </tr>
  <tr>
    <td>Uric Acid last, mean, SD</td>
    <td>Removed due to high missing rate</td>
  </tr>
  <tr>
    <td>aPTT last, , mean, SD</td>
    <td>Removed due to high missing rate</td>
  </tr>
  <tr>
    <td>POC Glucose last, mean, SD</td>
    <td>Removed due to high missing rate</td>
  </tr>
  <tr>
    <td rowspan="7">Vital Sign</td>
    <td>Pain last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Pulse last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Respiration last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Spo2 last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Body temperature last, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Diastolic blood pressurelast, mean, SD</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>Systolic blood pressure last, mean, SD</td>
    <td>Numeric</td>
  </tr>
</tbody>
</table>

[Back to top](#table-of-contents)


Sample characteristics
-----------------------
eTable 1: Patient characteristic description and comparison by sample 

| Variables |  | Pre-COVID    sample<br>    <br>(N = 4,010) |  | Peri-COVID    sample<br>    <br>(N = 3,950) |  | P-value |
|---|---|---|---|---|---|---|
|  |  | Mean or n | SD or % | Mean or n | SD or % |  |
| Age |  | 64.7 | 12.6 | 64.7 | 13.1 | .96 |
| Gender | Male | 2,539 | 63.3% | 2,369 | 60.0% | <.01 |
|  | Female | 1,471 | 36.7% | 1,581 | 40.0% |  |
| Race | White | 3,242 | 80.8% | 3,149 | 79.7% | .06 |
|  | Black | 271 | 6.8% | 323 | 8.2% |  |
|  | Other | 467 | 11.6% | 449 | 11.4% |  |
|  | Unknown | 30 | 0.7% | 29 | 0.7% |  |
| Marital status | Married | 2,977 | 74.2% | 2,802 | 70.9% | <.01 |
|  | Single | 980 | 24.4% | 1,074 | 27.2% |  |
|  | Other | 40 | 1.0% | 69 | 1.7% |  |
|  | Unknown | 13 | 0.3% | 5 | 0.1% |  |
| Smoking status | Smoker | 347 | 8.7% | 298 | 7.5% | <.01 |
|  | Former smoker | 1,990 | 49.6% | 1,830 | 46.3% |  |
|  | Never smoker | 1,631 | 40.7% | 1,800 | 45.6% |  |
|  | Unknown | 42 | 1.0% | 19 | 0.5% |  |
| Alcohol use | Yes | 1,448 | 36.1% | 1,30 | 32.9% | <.01 |
|  | Former user | 289 | 7.2% | 1,388 | 35.1% |  |
|  | No | 2,104 | 52.5% | 1,155 | 29.2% |  |
|  | Unknown | 169 | 4.2% | 106 | 2.7% |  |
| ECOG | 0 | 794 | 19.8% | 436 | 11.0% | .15 |
|  | 1 | 1,255 | 31.3% | 757 | 19.2% |  |
|  | >=2 | 699 | 17.4% | 364 | 9.2% |  |
|  | Unknown | 1,262 | 31.5% | 2,393 | 60.6% |  |
| Cancer site | Bronchus   and lung | 1,107 | 27.6% | 806 | 20.4% | <.01 |
|  | Skin | 787 | 19.6% | 615 | 15.6% |  |
|  | Kidney | 397 | 9.9% | 403 | 10.2% |  |
|  | Bladder | 158 | 3.9% | 126 | 3.2% |  |
|  | Prostate   gland | 145 | 3.6% | 139 | 3.5% |  |
|  | Breast | 73 | 1.8% | 190 | 4.8% |  |
|  | Colon | 73 | 1.8% | 94 | 2.4% |  |
|  | Stomach | 61 | 1.5% | 82 | 2.1% |  |
|  | Oropharynx | 56 | 1.4% | 49 | 1.2% |  |
|  | Thyroid   gland | 51 | 1.3% | 45 | 1.1% |  |
|  | Other | 761 | 19.0% | 1,006 | 25.5% |  |
|  | Unknown | 341 | 8.5% | 395 | 10.0% |  |
| Cancer   histology | Adenocarcinoma | 1,404 | 35.0% | 1,424 | 36.1% |  |
|  | Nevi and Melanoma | 652 | 16.3% | 476 | 12.1% |  |
|  | Squamous   or transitional cell Carcinoma | 625 | 15.6% | 561 | 14.2% |  |
|  | Other   specific Carcinoma | 344 | 8.6% | 296 | 7.5% |  |
|  | Sarcoma   or soft tissue tumor | 47 | 1.2% | 75 | 1.9% |  |
|  | Basal   cell Carcinoma | 29 | 0.7% | 29 | 0.7% |  |
|  | Other | 568 | 14.2% | 694 | 17.6% |  |
|  | Unknown | 341 | 8.5% | 395 | 10.0% |  |
| Metastasis status | Yes | 2,819 | 70.3% | 2,679 | 67.8% | .15 |
|  | No | 850 | 21.2% | 876 | 22.2% |  |
|  | Unknown | 341 | 8.5% | 395 | 10.0% |  |
| 30 days prior   RT | Yes | 268 | 6.7% | 270 | 6.8% | .81 |
|  | No | 3,742 | 93.3% | 3,680 | 93.2% |  |
| ICI agent | Atezolizumab | 253 | 6.3% | 332 | 8.4 | <.01 |
|  | Avelumab | 42 | 1.0% | 53 | 1.3% |  |
|  | Cemiplimab-rwlc | 31 | 0.8% | 70 | 1.8% |  |
|  | Durvalumab | 115 | 2.9% | 149 | 3.8% |  |
|  | Ipilimumab | 483 | 12.0% | 344 | 8.7% |  |
|  | Nivolumab | 1,305 | 32.5% | 1,028 | 26.0% |  |
|  | Pembrolizumab | 1,781 | 44.4% | 1,974 | 50.0% |  |
| 90-day   admission | Yes | 1,436 | 35.8% | 1,080 | 27.3% | <.01 |
|  | No | 2,574 | 64.2% | 2,870 | 72.7% |  |

***Abbrevations:*** COVID: coronavirus disease; ECOG: Eastern cooperative oncology group; ICI: immune checkpoint inhibitors; RT: radiation therapy.
[Back to top](#table-of-contents)


Model performance on pre COVID testing sample
----------------------
eTable 2: Model performance on pre-COVID testing sample (N = 802) 
|     <br>Metric     |     <br>XGBT     |     <br>RF     |     <br>SVM     |     <br>LRENP     |     <br>SHLNN     |     <br>KNN     |     <br>MARS     |     <br>DT     |     <br>LR     |     <br>ECOG     |
|---|---|---|---|---|---|---|---|---|---|---|
|    <br>AUROC<br>   <br>(95% CI)    |    <br>.67<br>   (.63, .71)    |    <br>.65<br>   (.61, .69)    |    <br>.66<br>   (.63, .70)    |    <br>.65<br>   (.61, .69)    |    <br>.65<br>   (.61, .69)    |    <br>.64<br>   (.60, .68)    |    <br>.64<br>   (.60, .68)    |    <br>.57<br>   (.54, .61)    |    <br>.64<br>   (.60, .68)    |    <br>.60<br>   (.55, .64)    |
|    <br>Accuracy <br>   (95% CI)    |    <br>.62<br>   (.59, .65)    |    <br>.63<br>   (.60, .67)    |    <br>.63<br>   (.59, .66)    |    <br>.63<br>   (.59, .66)    |    <br>.61<br>   (.58, .65)    |    <br>.61<br>   (.57, .64)    |    <br>.62<br>   (.58, .65)    |    <br>.59<br>   (.56, .63)    |    <br>.61<br>   (.58, .65)    |    <br>.51<br>   (.47, .55)    |
|    <br>Sensitivity    |    <br>.62    |    <br>.61    |    <br>.63    |    <br>.63    |    <br>.61    |    <br>.61    |    <br>.62    |    <br>.49    |    <br>.62    |    <br>.79    |
|    <br>Specificity    |    <br>.62    |    <br>.65    |    <br>.63    |    <br>.63    |    <br>.61    |    <br>.61    |    <br>.62    |    <br>.65    |    <br>.61    |    <br>.35    |
|    <br>PPV    |    <br>.47    |    <br>.49    |    <br>.48    |    <br>.48    |    <br>.47    |    <br>.46    |    <br>.47    |    <br>.44    |    <br>.47    |    <br>.41    |
|    <br>NPV    |    <br>.75    |    <br>.75    |    <br>.75    |    <br>.75    |    <br>.74    |    <br>.74    |    <br>.74    |    <br>.70    |    <br>.74    |    <br>.75    |
|    <br>True Positive    |    <br>177    |    <br>174    |    <br>180    |    <br>179    |    <br>175    |    <br>174    |    <br>176    |    <br>140    |    <br>176    |    <br>157    |
|    <br>False Positive    |    <br>196    |    <br>183    |    <br>192    |    <br>192    |    <br>199    |    <br>201    |    <br>198    |    <br>181    |    <br>201    |    <br>228    |
|    <br>True Negative    |    <br>320    |    <br>333    |    <br>324    |    <br>324    |    <br>317    |    <br>315    |    <br>318    |    <br>335    |    <br>315    |    <br>122    |
|    <br>False Negative    |    <br>109    |    <br>112    |    <br>106    |    <br>107    |    <br>111    |    <br>112    |    <br>110    |    <br>146    |    <br>110    |    <br>41    |
|    <br>Threshold    |    <br>.206    |    <br>.100    |    <br>.299    |    <br>.329    |    <br>.301    |    <br>.336    |    <br>.326    |    <br>.282    |    <br>.311    |    <br>.321    |

**Abbreviations:** AUROC: area under the receiver operating characteristic curve; DT: decision tree; ECOG: Eastern Cooperative Oncology Group; FN: false negative; FP: false positive; KNN: k-nearest neighbors; LR: logistic regression; LRENP: logistic regression with elastic net penalty; MARS: multivariate adaptive regression splines; NPV: negative predictive value; PPV: positive predictive value; RF: random forest; SHLNN: single hidden layer neural network; SVM: support vector machine; TN: true negative; TP: true positive; XGBT: extreme gradient boosting trees.

[Back to top](#table-of-contents)

Model calibration plots
-----------------------

We present calibration plots for all calibrated algorithms we examined on the peri-COVID sample in this section. The plot for the LRENP algorithm is included in our main paper.


<p align="center">
  <img width="460" height="400" src="https://user-images.githubusercontent.com/38151091/161775038-4909e440-c6fe-460a-857b-e109acbca61f.png">
</p>

eFigure 2. Calibration plot for the ECOG logistic regression algorithm.

**Note:** The lack of alignment between the solid and dash lines in most areas indicates that the model is not well calibrated.

<p align="center">

  <img width="460" height="400" src="https://user-images.githubusercontent.com/38151091/161775504-1fca1460-2ec0-4568-a7e1-81fe9988647f.png">
</p>


eFigure 3. Calibration plot for the ranfom forest algorithm.

**Note:** The calibration analysis result for the random forest is suboptimal. The model tends to underestimate the rsik for patients with moderate to high risk of ACU (>=30%).

<p align="center">
  <img width="460" height="400" src="https://user-images.githubusercontent.com/38151091/161775899-2f3a8adf-0b36-4fde-98b7-ca08358f0884.png">
</p>

eFigure 4. Calibration plot for the extreme gradient boosting tree algorithm.

**Note:** The calibration of the extreme gradient boosting trees algorithm is relatively well-balibrated for patients with low to moderate risk of ACU (<= 60%). The model tends to underestimate the risk for patients in a high risk area (>60%).

<p align="center">
  <img width="460" height="400" src="https://user-images.githubusercontent.com/38151091/161776133-39564734-4b3a-4f99-b9b3-ce8bff812b7f.png">
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
