# Machine learning algorithms predicting acute care use for patients within 90-day of immune checkpoint inhibitors
Table of contents
=================
<!--ts-->
 * [eMethods](#emethods)
   * [Data preparation](#data-preparation)
   * [Model training and optimization](#model-training-and-optimization) 
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
