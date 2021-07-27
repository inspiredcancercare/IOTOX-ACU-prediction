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
 We trained four ML algorithms, including logistic regression with elastic net penalty (LRENP), random forest (RF), extreme gradient boosting trees (XGBT), and single hidden layer neural network (SHLNN), using the per-COVID training sample. We used the following steps to determine the best values of hyperparameters for each algorithm alongside 10-fold cross validation process. 
1. Trained and evaluated 30 inital models using random search approach from presdefined search spaces (eTable 1)
2. Fitted a Gaussian process (GP) model using inital models' hyperparameter values as predictors and model performance (AUROC) as outocmes
3. Projected a potentially optimal hyperparatemer values using the GP model
4. Created a new model with the hyperparameter values   
5. Compared the new model with the averaging AUROC of the inital models.
6. Repeated step 3-5 until 40 iterations or no model improvement in 20 consecutive models were reach

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

R code for the hyperparameter value selection and model training. We used [R tidymodels package version 0.1.0](https://www.tidymodels.org/)
```
#### Create resample####
set.seed(my_seed)
cvfolds<-vfold_cv(df_train, v = 10, strata = outcome)

#### Model tuning and training ####

#multivariate logistic regression
lr_model<-parsnip::logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

lr_workflow<-workflow() %>%
  add_recipe(recipe) %>%
  add_model(lr_model) %>%
  fit(df_train)

#Ecog univariate logistic regression
ecog_formula<- as.formula(paste(outcome, "ecog_latest", sep="~"))

ecog_rc <-  recipe(ecog_formula, df_train) %>%
  step_modeimpute(all_nominal(), -recipes::all_outcomes()) %>%
  step_integer(ecog_latest)

ecog_model<-parsnip::logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

ecog_workflow<-workflow() %>%
  add_recipe(ecog_rc) %>%
  add_model(ecog_model) %>%
  fit(df_train)

#logistic regression with elastic net penalty 
lrenp_model<-parsnip::logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

set.seed(my_seed)
lrenp_params<-model %>%
  parameters() %>%
  update(
    penalty = penalty(range(c(-16L,0))),
    mixture= mixture(range(c(0,1)))) %>%
  grid_max_entropy(size = 30)

lrenp_workflow<-workflow() %>%
  add_recipe(recipe) %>%
  add_model(lrenp_model)    

#inital random tuning
lrenp_inital_search <- model_tuning_grid(lrenp_workflow,cvfolds, lrenp_params)

#Bayesian tuning based on the results of the intial random tuning 
set.seed(my_seed)
lrenp_params<-model %>%
  parameters() %>%
  update(
    penalty = penalty(range(c(-16L,0))),
    mixture= mixture(range(c(0,1))))


lrenp_search <- model_tuning(lrenp_workflow, 
                             cvfolds, 
                             lrenp_params, 
                             lrenp_inital_search)       


print(autoplot(search, metric="roc_auc"))

print(search)

lrenp_best_model<-select_best(lrenp_search, metric = primary_metric)


set.seed(my_seed)
lrenp_finalworkflow<-lrenp_workflow %>%
  finalize_workflow(lrenp_best_model) %>%
  fit(df_train)

#Random forest
rf_model<-parsnip::rand_forest(mtry = tune(),trees = tune(), min_n = tune()) %>%
  set_engine("ranger", importance="impurity", seed = my_seed) %>%
  set_mode("classification")

set.seed(my_seed)
rf_params<-model %>%
  parameters() %>%
  update(
    mtry = mtry(range = c(1L,length(dplyr::select(juiced_df_train, -outcome)))),
    trees = trees(range(c(1L:4500L))),
    min_n = min_n(range(c(2L:700L)))) %>%
  grid_max_entropy(size = 30)

rf_workflow<-workflow() %>%
  add_recipe(recipe) %>%
  add_model(rf_model)    

#inital random tuning
rf_inital_search <- model_tuning_grid(rf_workflow,cvfolds, rf_params)

#Bayesian tuning based on the results of the intial random tuning 
set.seed(my_seed)
rf_params<-model %>%
  parameters() %>%
  update(
    mtry = mtry(range = c(1L,length(dplyr::select(juiced_df_train, -outcome)))),
    trees = trees(range(c(1L:4500L))),
    min_n = min_n(range(c(2L:700L))))


rf_search <- model_tuning(rf_workflow, 
                             cvfolds, 
                             rf_params, 
                             rf_inital_search)       


print(autoplot(search, metric="roc_auc"))

print(search)

rf_best_model<-select_best(rf_search, metric = primary_metric)


set.seed(my_seed)
rf_finalworkflow<-rf_workflow %>%
  finalize_workflow(rf_best_model) %>%
  fit(df_train)

#Extreme gradient boosting trees
xgb_model<-parsnip::boost_tree(trees = tune(), 
                               learn_rate = tune(),
                               tree_depth = tune(),
                               min_n = tune(),
                               loss_reduction = tune(),
                               sample_size = tune(),
                               mtry = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

set.seed(my_seed)
xgb_params<-model %>%
  parameters() %>%
  update(mtry = mtry(range = c(1L,length(dplyr::select(juiced_df_train, -outcome)))),
         trees = trees(range = c(1L, 4000L)),
         min_n = min_n(range = c(2L, 75L)),
         tree_depth = tree_depth(range = c(1L,150L)),
         learn_rate = learn_rate(range = c(-17L,-1L)),
         loss_reduction = loss_reduction(range = c(-17L, 1.5))) %>%
  grid_max_entropy(size = 30)

xgb_workflow<-workflow() %>%
  add_recipe(recipe) %>%
  add_model(xgb_model)    

#inital random tuning
xgb_inital_search <- model_tuning_grid(xgb_workflow,cvfolds, xgb_params)

#Bayesian tuning based on the results of the intial random tuning 
set.seed(my_seed)
xgb_params<-model %>%
  parameters() %>%
  update(mtry = mtry(range = c(1L,length(dplyr::select(juiced_df_train, -outcome)))),
         trees = trees(range = c(1L, 4000L)),
         min_n = min_n(range = c(2L, 75L)),
         tree_depth = tree_depth(range = c(1L,150L)),
         learn_rate = learn_rate(range = c(-17L,-1L)),
         loss_reduction = loss_reduction(range = c(-17L, 1.5)))

xgb_search <- model_tuning(xgb_workflow, 
                          cvfolds, 
                          xgb_params, 
                          xgb_inital_search)       


print(autoplot(search, metric="roc_auc"))

print(search)

xgb_best_model<-select_best(xgb_search, metric = primary_metric)

set.seed(my_seed)
xgb_finalworkflow<-xgb_workflow %>%
  finalize_workflow(xgb_best_model) %>%
  fit(df_train)


#single hidden layer neural network

shlnn_model<-parsnip::mlp(hidden_units = tune(),
                          penalty = tune(),
                          epochs = tune())%>%
  set_engine("nnet") %>%
  set_mode("classification")

set.seed(my_seed)
shlnn_params<-model %>%
  parameters() %>%
  update(hidden_units = hidden_units(range(c(1L,20L))),
         penalty = penalty(range(c(-10L,2L))),
         epochs = epochs(range(c(1L,1500L)))) %>%
  grid_max_entropy(size = 30)

shlnn_workflow<-workflow() %>%
  add_recipe(recipe) %>%
  add_model(shlnn_model)    

#inital random tuning
shlnn_inital_search <- model_tuning_grid(shlnn_workflow,cvfolds, shlnn_params)

#Bayesian tuning based on the results of the intial random tuning 
set.seed(my_seed)
shlnn_params<-model %>%
  parameters() %>%
  update(hidden_units = hidden_units(range(c(1L,20L))),
         penalty = penalty(range(c(-10L,2L))),
         epochs = epochs(range(c(1L,1500L)))) 

shlnn_search <- model_tuning(shlnn_workflow, 
                           cvfolds, 
                           shlnn_params, 
                           shlnn_inital_search)       

print(autoplot(search, metric="roc_auc"))

print(search)

shlnn_best_model<-select_best(shlnn_search, metric = primary_metric)

set.seed(my_seed)
shlnn_finalworkflow<-shlnn_workflow %>%
  finalize_workflow(shlnn_best_model) %>%
  fit(df_train)
```
[Back to top](#table-of-contents)
 
