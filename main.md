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
log_trans_variable<- c("lab_AST_last","lab_ALT_last”,"lab_ALK_Phos_last”,"lab_BUN_last”,"lab_Basophilab_Abs_last",”lab_Basophilab_pct_last”,
"lab_Chloride_last”,”lab_Creatinine_last”,”lab_Eosinophilab_pct_last”,"lab_Eosinophilab_Abs_last”,"lab_Platelet_last”,”lab_Glucose_Lvl_last”,
”lab_IGRE_pct_last", “lab_LDH_last",”lab_Lymphocyte_Abs_last”,”lab_Lymphocyte_pct_last”,”lab_Monocyte_Abs_last",”lab_Neutrophilab_Abs_last”, “lab_Neutrophilab_pct_last”,"lab_RBC_last”,"lab_RDW_CV_last”,”lab_RDW_SD_last”,”lab_eGFR_AA_last”,”lab_creatinine_clear_last”,
”lab_Bili_Direct_last”,”lab_Bili_Indirect_last",”lab_Bili_Total_last”,"lab_IG_Abs_last”,"lab_INR_last”,"lab_INRBC_last”,"lab_TSH_last",
”lab_WBC_last”,"vit_p_last”,"vit_spo2_last","vit_r_last","lab_AST_mean","lab_ALT_mean”,"lab_ALK_Phos_mean”,"lab_BUN_mean”,"lab_Basophilab_Abs_mean",
”lab_Basophilab_pct_mean”,"lab_Chloride_mean”,”lab_Creatinine_mean",”lab_Eosinophilab_pct_mean”,”lab_Eosinophilab_Abs_mean”,"lab_Platelet_mean",
”lab_Glucose_Lvl_mean”,”lab_IGRE_pct_mean”, “lab_LDH_mean”, ”lab_Lymphocyte_Abs_mean", ”lab_Lymphocyte_pct_mean”, ”lab_Monocyte_Abs_mean”,
"lab_Neutrophilab_Abs_mean”,“lab_Neutrophilab_pct_mean”,"lab_RBC_mean”,"lab_RDW_CV_mean",”lab_RDW_SD_mean”,”lab_eGFR_AA_mean”,”lab_creatinine_clear_mean",
”lab_Bili_Direct_mean”,”lab_Bili_Indirect_mean”,”lab_Bili_Total_mean”,"lab_IG_Abs_mean","lab_INR_mean”,"lab_INRBC_mean”,"lab_TSH_mean”,”lab_WBC_mean",
”vit_p_mean","vit_spo2_mean","vit_r_mean”,”lab_AST_sd","lab_ALT_sd”,"lab_ALK_Phos_sd”,"lab_BUN_sd”,"lab_Basophilab_Abs_sd”,”lab_Basophilab_pct_sd”,
"lab_Chloride_sd”,”lab_Creatinine_sd”,”lab_Eosinophilab_pct_sd”,"lab_Eosinophilab_Abs_sd”,"lab_Platelet_sd”,”lab_Glucose_Lvl_sd”,”lab_IGRE_pct_sd”, “lab_LDH_sd”,”lab_Lymphocyte_Abs_sd”,”lab_Lymphocyte_pct_sd”,”lab_Monocyte_Abs_sd", "lab_Neutrophilab_Abs_sd”, “lab_Neutrophilab_pct_sd”, 
"lab_RBC_sd”,"lab_RDW_CV_sd”, ”lab_RDW_SD_sd”,”lab_eGFR_AA_sd”,”lab_creatinine_clear_sd",  “lab_Bili_Direct_sd”,”lab_Bili_Indirect_sd”,
”lab_Bili_Total_sd”,"lab_IG_Abs_sd”,"lab_INR_sd”,"lab_INRBC_sd”, "lab_TSH_sd”, ”lab_WBC_sd", "vit_p_sd", 
"vit_spo2_sd", "vit_r_sd”)

ordinal_varaible<-c("vit_pain","pps_ecog_last", "ADI_NATRANK")

##### split data into training and testing samples ####
set.seed(my_seed)
split<-initial_split(db, prop = .8) #where db is the pre-COVID sample
df_train<-training(split)
df_test<-testing(split)

##### set a recipe for data prepraration ####
df_recipe<-recipe(formula, df_train) %>%
   step_medianimpute(recipes::all_numeric(), -recipes::all_outcomes(), id = "median_imputation") %>%
   step_modeimpute(recipes::all_nominal(), -recipes::all_outcomes(), id = "mode_imputation") %>%
   step_YeoJohnson(!!!log_trans_variables, -recipes::all_outcomes(), id = "yeoJohnson_transformation") %>%
   step_normalize(recipes::all_numeric(), -recipes::all_outcomes(), id = "scale_normalization") %>%
   step_integer(!!!ordinal_varaibles, id = "ordinalscore") %>%
   step_other(gender, marital_status, smoking, alcohol,disease_site,disease_histology, threshold = 0.1, other = "other", id = "lumping") %>%
   step_zv(recipes::all_predictors(),-recipes::all_outcomes(), id = "remove zero variance") %>%
   step_nzv(recipes::all_predictors(),-recipes::all_outcomes(), id = "remove near zero variance") %>%
   step_dummy(recipes::all_nominal(), -recipes::all_outcomes(), one_hot=FALSE, id="create_dummy_variables") %>%
   step_corr(recipes::all_numeric(), -recipes::all_outcomes(), id="remove_high_correlation")
  
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

eTable 2. Candidate predictors
<table>
<thead>
  <tr>
    <th class="tg-zr06">Category</th>
    <th class="tg-kcps">Name</th>
    <th class="tg-kcps">Value</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-f4yw" rowspan="7">Basic information</td>
    <td class="tg-cwad">Age</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-cwad">Gender</td>
    <td class="tg-cwad">Male, Female</td>
  </tr>
  <tr>
    <td class="tg-cwad">Marital status</td>
    <td class="tg-cwad">Married, Other, Single</td>
  </tr>
  <tr>
    <td class="tg-cwad">Smoking status</td>
    <td class="tg-cwad">Current smoker, Former smoker, Never smoker,</td>
  </tr>
  <tr>
    <td class="tg-cwad">Alcohol use</td>
    <td class="tg-cwad">Yes, Not current, No</td>
  </tr>
  <tr>
    <td class="tg-cwad">ECOG last, mean, sd</td>
    <td class="tg-cwad">0, 1, 2, 3, 4. Mean ECOG was not used due to high correlation</td>
  </tr>
  <tr>
    <td class="tg-cwad">Area deprivation index - national</td>
    <td class="tg-cwad">0 - 99, GQ, GQ-PH, KVM, PH</td>
  </tr>
  <tr>
    <td class="tg-f4yw" rowspan="2">Treatment</td>
    <td class="tg-cwad">RT 30 days before</td>
    <td class="tg-cwad">True, False</td>
  </tr>
  <tr>
    <td class="tg-kcps">Immune checkpoint inhibitor agent</td>
    <td class="tg-kcps">Avelumab, Durvalumab, Nivolumab, Pembrolizumab, Cemiplimab-rwlc, Atezolizumab</td>
  </tr>
  <tr>
    <td class="tg-f4yw" rowspan="3">Cancer information</td>
    <td class="tg-kcps">Site</td>
    <td class="tg-cwad">Nominal </td>
  </tr>
  <tr>
    <td class="tg-kcps">Histology</td>
    <td class="tg-cwad">Nominal </td>
  </tr>
  <tr>
    <td class="tg-kcps">Metastasis status</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-f4yw" rowspan="41">Medical histories</td>
    <td class="tg-kcps">Anxiety</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Asthma</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Atrioventricular septal defect</td>
    <td class="tg-cwad">Not used due to near zero variance</td>
  </tr>
  <tr>
    <td class="tg-kcps">Atrial fibrillaton</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Bronchiolitis obliterans organizing pneumonia</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Cardiac arrhythmia</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Cerebrovascular accident</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Cerebrovascular disease</td>
    <td class="tg-cwad">Not used due to near zero variance</td>
  </tr>
  <tr>
    <td class="tg-kcps">Chronic kidney disease</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Chronic obstructive pulmonary disease</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Chronic pulmonary disease not asthma</td>
    <td class="tg-cwad">Not used due to near zero variance</td>
  </tr>
  <tr>
    <td class="tg-kcps">Congestive heart failure</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Coronary artery disease</td>
    <td class="tg-cwad">Not used due to near zero variance</td>
  </tr>
  <tr>
    <td class="tg-kcps">Deep vein thrombosis</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Dementia</td>
    <td class="tg-cwad">Not used due to near zero variance</td>
  </tr>
  <tr>
    <td class="tg-kcps">Depression</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Diabetes insipidus</td>
    <td class="tg-cwad">Not used due to near zero variance</td>
  </tr>
  <tr>
    <td class="tg-kcps">Diabetes mellitus</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">ESRD</td>
    <td class="tg-cwad">Not used due to near zero variance</td>
  </tr>
  <tr>
    <td class="tg-kcps">ESRD SF</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Gerd</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Hemiplegia Paraplegia</td>
    <td class="tg-cwad">Not used due to near zero variance</td>
  </tr>
  <tr>
    <td class="tg-kcps">Human immunodeficiency virus</td>
    <td class="tg-cwad">Not used due to near zero variance</td>
  </tr>
  <tr>
    <td class="tg-kcps">Hypertension</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Hypogammaglobulinemia</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Hypothyroidism</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Immune related adverse events pneumonitis</td>
    <td class="tg-cwad">Not used due to near zero variance</td>
  </tr>
  <tr>
    <td class="tg-kcps">Cryptogenic organizing pneumonia</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Liver disease</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Mild liver disease</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Moderate or severe liver disease</td>
    <td class="tg-cwad">Not used due to high correlation</td>
  </tr>
  <tr>
    <td class="tg-kcps">Myocardial infarction</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Neuropathy</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Obesity</td>
    <td class="tg-cwad">Not used due to near zero variance</td>
  </tr>
  <tr>
    <td class="tg-kcps">Obstructive sleep apnea</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Peripheral vascular disease</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Pneumonia</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Psychosis</td>
    <td class="tg-cwad">Not used due to near zero variance</td>
  </tr>
  <tr>
    <td class="tg-kcps">Pulmonary embolism</td>
    <td class="tg-cwad">Yes, No</td>
  </tr>
  <tr>
    <td class="tg-kcps">Pulmonary hypertension</td>
    <td class="tg-cwad">Not used due to near zero variance</td>
  </tr>
  <tr>
    <td class="tg-kcps">Rheumatoid arthritis</td>
    <td class="tg-cwad">Not used due to near zero variance</td>
  </tr>
  <tr>
    <td class="tg-f4yw" rowspan="65">Laboratory</td>
    <td class="tg-kcps">ALT last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">AST last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Albumin Lvl last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Alk Phos last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Anion Gap last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">BUN last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Basophil Abs last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Basophil pct last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Bili Direct last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Bili Indirect last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Bili Total last, mean, SD</td>
    <td class="tg-cwad">Numeric, sd Bili Total was not used due to high correlation.</td>
  </tr>
  <tr>
    <td class="tg-kcps">CO2 last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Calcium Lvl last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Chloride last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Creatinine last, mean, SD</td>
    <td class="tg-cwad">Numeric, last Creatinine was not used due to high correlation.</td>
  </tr>
  <tr>
    <td class="tg-kcps">Eosinophil Abs last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Eosinophil pct last, mean, SD</td>
    <td class="tg-cwad">Numeric, last and mean Eosinophil pcts were not used due to high correlation.</td>
  </tr>
  <tr>
    <td class="tg-kcps">Glucose Level last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Hct last, mean, SD</td>
    <td class="tg-cwad">Numeric, sd Hct was not used due to high correlation.</td>
  </tr>
  <tr>
    <td class="tg-kcps">Hgb last, mean, SD</td>
    <td class="tg-cwad">Not used due to high correlation</td>
  </tr>
  <tr>
    <td class="tg-kcps">IGRE pct last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">IG Abs last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">INRBC last, mean, SD</td>
    <td class="tg-cwad">Numeric, last INRBC was not used due to near zero variance</td>
  </tr>
  <tr>
    <td class="tg-kcps">LDH last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Lymphocyte Abs last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Lymphocyte pct last, mean, SD</td>
    <td class="tg-cwad">Numeric, last Lymphocyte pct was not used due to high correlation.</td>
  </tr>
  <tr>
    <td class="tg-kcps">MCHC last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">MCH last, mean, SD</td>
    <td class="tg-cwad">Numeric, mean MCH was not used due to high correlation.</td>
  </tr>
  <tr>
    <td class="tg-kcps">MCV last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">MPV last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Magnesium last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Monocyte Abs last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Monocyte pct last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Neutrophil Abs last, mean, SD</td>
    <td class="tg-cwad">Numeric, last and mean Neutrophil Abs were not used due to high correlation.</td>
  </tr>
  <tr>
    <td class="tg-kcps">Neutrophil pct last, mean, SD</td>
    <td class="tg-cwad">Numeric, mean Neutrophil pct was not used due to high correlation.</td>
  </tr>
  <tr>
    <td class="tg-kcps">Phosphorus last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Platelet count last, mean, SD</td>
    <td class="tg-cwad">Numeric, mean Platelet count was not used due to high correlation.</td>
  </tr>
  <tr>
    <td class="tg-kcps">Potassium Lvl last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">RBC last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">RDW CV last, mean, SD</td>
    <td class="tg-cwad">Numeric, sd RDW CV was not used due to high correlation.</td>
  </tr>
  <tr>
    <td class="tg-kcps">RDW SD last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Sodium Lvl last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">T4 Free last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">TSH last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Total Protein last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">WBC last, mean, SD</td>
    <td class="tg-cwad">Numeric, sd WBC was not used due to high correlation.</td>
  </tr>
  <tr>
    <td class="tg-kcps">Creatinine clear last, mean, SD</td>
    <td class="tg-cwad">Numeric, mean and sd Creatinine clear were not used due to high correlation.</td>
  </tr>
  <tr>
    <td class="tg-kcps">eGFR AA last, mean, SD</td>
    <td class="tg-cwad">Numeric, last and mean eGFR AA were not used due to high correlation.</td>
  </tr>
  <tr>
    <td class="tg-kcps">Cortisol last, mean,SD</td>
    <td class="tg-cwad">Removed due to high missing rate</td>
  </tr>
  <tr>
    <td class="tg-kcps">Free T3 last, mean,SD</td>
    <td class="tg-cwad">Removed due to high missing rate</td>
  </tr>
  <tr>
    <td class="tg-kcps">INR last, mean, SD</td>
    <td class="tg-cwad">Removed due to high missing rate</td>
  </tr>
  <tr>
    <td class="tg-kcps">Lipase Lvl last, mean,SD</td>
    <td class="tg-cwad">Removed due to high missing rate</td>
  </tr>
  <tr>
    <td class="tg-kcps">POC Crea last, mean, SD</td>
    <td class="tg-cwad">Removed due to high missing rate</td>
  </tr>
  <tr>
    <td class="tg-kcps">POC Glucose last, mean, SD</td>
    <td class="tg-cwad">Removed due to high missing rate</td>
  </tr>
  <tr>
    <td class="tg-kcps">POC creatinine clear last, mean, SD</td>
    <td class="tg-cwad">Removed due to high missing rate</td>
  </tr>
  <tr>
    <td class="tg-kcps">T3 Total last, mean, SD</td>
    <td class="tg-cwad">Removed due to high missing rate</td>
  </tr>
  <tr>
    <td class="tg-kcps">Total Cells last, mean, SD</td>
    <td class="tg-cwad">Removed due to high missing rate</td>
  </tr>
  <tr>
    <td class="tg-kcps">UA Protein last, mean SD</td>
    <td class="tg-cwad">Removed due to high missing rate</td>
  </tr>
  <tr>
    <td class="tg-kcps">UA RBC last, mean, SD</td>
    <td class="tg-cwad">Removed due to high missing rate</td>
  </tr>
  <tr>
    <td class="tg-kcps">UA Spec Grav last, mean, SD</td>
    <td class="tg-cwad">Removed due to high missing rate</td>
  </tr>
  <tr>
    <td class="tg-kcps">UA WBC last, mean, SD</td>
    <td class="tg-cwad">Removed due to high missing rate</td>
  </tr>
  <tr>
    <td class="tg-kcps">UA pH last, mean, SD</td>
    <td class="tg-cwad">Removed due to high missing rate</td>
  </tr>
  <tr>
    <td class="tg-kcps">Uric Acid last, mean, SD</td>
    <td class="tg-cwad">Removed due to high missing rate</td>
  </tr>
  <tr>
    <td class="tg-kcps">aPTT last, , mean, SD</td>
    <td class="tg-cwad">Removed due to high missing rate</td>
  </tr>
  <tr>
    <td class="tg-kcps">POC Glucose last, mean, SD</td>
    <td class="tg-cwad">Removed due to high missing rate</td>
  </tr>
  <tr>
    <td class="tg-f4yw" rowspan="7">Vital Sign</td>
    <td class="tg-kcps">Pain last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Pulse last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Respiration last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Spo2 last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Body temperature last, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Diastolic blood pressurelast, mean, SD</td>
    <td class="tg-cwad">Numeric</td>
  </tr>
  <tr>
    <td class="tg-kcps">Systolic blood pressure last, mean, SD</td>
    <td class="tg-cwad">Numeric, mean SBP was not used due to high correlation.</td>
  </tr>
</tbody>
</table>


[Back to top](#table-of-contents)

Sample characteristics
-----------------------
eTable 3: Patient characteristic description and comparison by sample 

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
eTable 4: Model performance on pre-COVID testing sample (N = 802) 
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

We present calibration plots for all calibrated algorithms we examined on the peri-COVID sample in this section. The plot for the XGBT algorithm is included in our main paper.


<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/182243845-fd4859c6-fe97-4680-8e20-ddaa0a5148e5.png">
</p>

eFigure 1. Calibration plot for the ECOG logistic regression algorithm.

**Note:** The ECOG model assigned all patients with similar scores and was likely to overpredict in the area of predicted probabilities >=0.5.


<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/182244689-929a2167-c5dd-4fd0-9e6e-04ce3bfe4a3f.png">
</p>

eFigure 2. Calibration plot for the multivariate logistic regression algorithm.

**Note:** The calibration analysis result for model is relatively good, although the model tended to slightly overpredict in the rang of predicted probability <= 0.5.


<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/182251913-67d6367c-097b-4886-94eb-997b7e76804c.png">
</p>


eFigure 3. Calibration plot for the random forest algorithm.

**Note:** The random forest model is not calibrated as the red line flucated and dose not algin with the solid black line. 


<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/182248621-56760329-5565-43f2-b5fe-ea34d0986dca.png">
</p>

eFigure 4. Calibration plot for the support vector machine algorithm.

**Note:** The calibration plot suggests that the probabilities <=0.3 generated by the model was reliable, and predicted probabilities >=0.4 were likely to be the result of overestimating.


<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/182250954-a73e642b-1f5f-43d9-9ace-c395462877bd.png">
</p>

eFigure 5. Calibration plot for the logistic regression with elastic net panelty algorithm.

**Note:** The calibration plot for the logistic regression with elastic net panelty is similar to the plot for support vector machine. The red line is sightly closer to the diagonal sold black line indicate a sightly better calibration as compared to the support vector machine.

<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/182249063-743b503a-eaa1-4baf-b075-912319ba094f.png">
</p>

eFigure 6. Calibration plot for the single hidden layer neural network algorithm.

**Note:** The calibration plot for single hidden layer neural network is also similar to the plot for support vector machine but slightly better.


<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/182249493-e2dbb4db-87e5-4a93-a7ad-694c01474243.png">
</p>

eFigure 7. Calibration plot for the k nearest neighbors algorithm.

**Note:** The calibration plot suggests that the model is relatively well-calibrated and generates reliable predicted probabilities.


<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/182249866-0b817f7c-0066-40c0-a126-f7236cd70bb3.png">
</p>

eFigure 8. Calibration plot for the multivariate adaptive regression spline algorithm.

**Note:** The calibration plot suggests that the model generates reliable outputs for most patients, while the patients determined as at high AUC risk (>=0.6) are likely to be overestimated.


<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/182250294-9e0fa07f-6c27-4470-b8a4-aac3e514bef5.png">
</p>

eFigure 9. Calibration plot for the decision tree algorithm.

**Note:** The plot shows that the decision tree algorithm is not well-calibrated and consistently exaggerates the likelihood of patients being readmitted for acute care use.


[Back to top](#table-of-contents)

Variable importance plots
--------------------------

We present the result for the XGBT algorithm in our main paper and other ML algorithms here.
<br>
</br>
<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/182391122-8f597625-f430-43d7-92c3-05f4018114fe.png">
</p>
eFigure 10. Important variables for the random forest algorithm.

<br>
</br>
<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/182247034-b16e785e-6dbd-4ae6-b5a2-8ec9a9182934.png">
</p>
eFigure 11. Important variables for support vector machine algorithm.

<br>
</br>
<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/182247115-bbe64520-558d-4de2-90ce-abca45648ec9.png">
</p>
eFigrue 12. Important variables for the logistic regression with elastic net penalty algorithm.

<br>
</br>
<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/182247209-bb4cee6f-e4d3-4528-a7f3-cbb0f51be661.png">
</p>
eFigrue 13. Important variables for the single hidden layer neural network algorithm.

<br>
</br>
<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/182247307-e6d7612d-8295-4d25-b8bd-1c6444047fc1.png">
</p>
eFigrue 14. Important variables for the k nearest neighbors algorithm.

<br>
</br>
<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/182247380-7ce84ca6-c2d1-42d4-9dac-9c16223d5cff.png">
</p>
eFigrue 15. Important variables for the multivariate adaptive regression spline algorithm.

<br>
</br>
<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/182247472-16cc4d1c-f9d3-45fe-a1fd-b77798c5b809.png">
</p>
eFigrue 16. Important variables for the decision tree algorithm.
<br>
</br>

[Back to top](#table-of-contents)



Shapely additive explanation plots
--------------------------

The result for the XGBT algorithm is included in the main test, and here are the results for other algorithms.
<br></br>

<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/182392194-09c8a015-ee9c-49f4-8400-d3a792d90c51.png">
</p>
eFigure 17. Contributions of predictor values to the prediction of the random forest algorithm for the randomly selected case.
<br>
</br>

<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/182392324-04812c60-d44b-4ab8-a34e-8e442ad9e751.png">
</p>
eFigrue 18. Contributions of predictor values to the prediction of the support vector machine for the randomly selected case.

<br>
</br>
<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/182392736-333aa0b9-c9aa-4efc-9b10-066d2b6888d7.png">
</p>
eFigure 19. Contributions of predictor values to the prediction of the logistic regression with elastic net panelty algorithm for the randomly selected case.
<br>
</br>

<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/182392807-206b609a-e14e-439b-8610-84b3440fac23.png">
</p>
eFigure 20. Contributions of predictor values to the prediction of the single hidden layer neural network  algorithm for the randomly selected case.

<br>
</br>

<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/182393044-6fa810e9-61ea-44be-83ec-ff57b84e83df.png">
</p>
eFigure 21. Contributions of predictor values to the prediction of the k nearest neighbor algorithm for the randomly selected case.

<br>
</br>

<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/182393150-95fb58bc-7fdd-4f74-a7cf-954b3c483790.png">
</p>
eFigure 22. Contributions of predictor values to the prediction of the multivariate adaptive regression spline algorithm for the randomly selected case.

<br>
</br>

<p align="center">
  <img src="https://user-images.githubusercontent.com/38151091/182393320-8039d9aa-c48d-4d8b-be2f-dc1d63cb64e2.png">
</p>
eFigure 22. Contributions of predictor values to the prediction of the decision tree algorithm for the randomly selected case.


[Back to top](#table-of-contents)
