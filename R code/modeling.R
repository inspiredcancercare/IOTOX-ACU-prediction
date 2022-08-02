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
lrenp_params<-lrenp_model %>%
  parameters() %>%
  grid_max_entropy(size = 30)

lrenp_workflow<-workflow() %>%
  add_recipe(recipe) %>%
  add_model(lrenp_model)    

#inital random tuning
lrenp_inital_search <- model_tuning_grid(lrenp_workflow,cvfolds, lrenp_params)

#Bayesian tuning based on the results of the intial random tuning 
set.seed(my_seed)
lrenp_params<-lrenp_model %>%
  parameters() %>%
  update(
    penalty = penalty(range(c(-10L,0))),
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
rf_params<-rf_model %>%
  parameters() %>%
  update(
    mtry = mtry(range = c(1L,length(dplyr::select(juiced_df_train, -outcome))))
    )%>%
  grid_max_entropy(size = 30)

rf_workflow<-workflow() %>%
  add_recipe(recipe) %>%
  add_model(rf_model)    

#inital random tuning
rf_inital_search <- model_tuning_grid(rf_workflow,cvfolds, rf_params)

#Bayesian tuning based on the results of the intial random tuning 
set.seed(my_seed)
rf_params<-rf_model %>%
  parameters() %>%
  update(
    mtry = mtry(range = c(1L,length(dplyr::select(juiced_df_train, -outcome)))),
    trees = trees(range(c(1L:2500L))),
    min_n = min_n(range(c(1L:40L))))

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
xgb_params<-xgb_model %>%
  parameters() %>%
  update(mtry = mtry(range = c(1L,length(dplyr::select(juiced_df_train, -outcome)))))%>%
  grid_max_entropy(size = 30)

xgb_workflow<-workflow() %>%
  add_recipe(recipe) %>%
  add_model(xgb_model)    

#inital random tuning
xgb_inital_search <- model_tuning_grid(xgb_workflow,cvfolds, xgb_params)

#Bayesian tuning based on the results of the intial random tuning 
set.seed(my_seed)
xgb_params<-xgb_model %>%
  parameters() %>%
  update(mtry = mtry(range = c(1L,length(dplyr::select(juiced_df_train, -outcome)))),
         trees = trees(range = c(1L, 2000L)),
         min_n = min_n(range = c(2L, 40L)),
         tree_depth = tree_depth(range = c(1L,15L)),
         learn_rate = learn_rate(range = c(-10L,-1L)),
         loss_reduction = loss_reduction(range = c(-15L, 1.5)),
         sample_size = sample_size(range = c(0.1, 1L)))

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
shlnn_params<-shlnn_model %>%
  parameters() %>%
  grid_max_entropy(size = 30)

shlnn_workflow<-workflow() %>%
  add_recipe(recipe) %>%
  add_model(shlnn_model)    

#inital random tuning
shlnn_inital_search <- model_tuning_grid(shlnn_workflow,cvfolds, shlnn_params)

#Bayesian tuning based on the results of the intial random tuning 
set.seed(my_seed)
shlnn_params<-shlnn_model %>%
  parameters() %>%
  update(hidden_units = hidden_units(range(c(1L,50L))),
         penalty = penalty(range(c(-10L,1))),
         epochs = epochs(range(c(1L,2000L)))) 

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

#support vector machine

svm_model<-parsnip::mlp(cost = tune(), 
    rbf_sigma = tune())%>%
  set_engine("kernlab") %>%
  set_mode("classification")

set.seed(my_seed)
juiced_df_train <- juice(prep(recipe, df_train)
formula <- as.formula(paste(outcome, ".", sep = "~"))
rbfSigma <- sigest(formula, data = juiced_df_train, frac = 1)
                         
set.seed(my_seed)
svm_params<-svm_model %>%
  parameters() %>%
  update(rbf_sigma= rbf_sigma(range=c(rbfSigma[1],rbfSigma[3]),trans = NULL),
        cost = cost()) %>%
  grid_max_entropy(size = 30)
                                  
svm_workflow<-workflow() %>%
  add_recipe(recipe) %>%
  add_model(svm_model)    
  
#inital random tuning
svm_inital_search <- model_tuning_grid(svm_workflow,cvfolds, svm_params)
#Bayesian tuning based on the results of the intial random tuning 
set.seed(my_seed)
svm_params<-svm_model %>%
  parameters() %>%
  update(rbf_sigma = rbf_sigma(range = c(rbfSigma[1],rbfSigma[3]),trans = NULL),
         cost = cost(range=c(-15L, -1L)))
svm_search <- model_tuning(svm_workflow, 
                           cvfolds, 
                           svm_params, 
                           svm_inital_search)       
                                  
print(autoplot(search, metric="roc_auc"))
print(search)

svm_best_model<-select_best(svm_search, metric = primary_metric)
set.seed(my_seed)
svm_finalworkflow<-svm_workflow %>%
  finalize_workflow(svm_best_model) %>%
  fit(df_train)


#k nearest neighbors

knn_model<-parsnip::mlp(neighbor = tune())%>%
  set_engine("kknn") %>%
  set_mode("classification")

set.seed(my_seed)
knn_params<-knn_model %>%
  parameters() %>%
  grid_max_entropy(size = 30)

knn_workflow<-workflow() %>%
  add_recipe(recipe) %>%
  add_model(knn_model)    

#inital random tuning
knn_inital_search <- model_tuning_grid(knn_workflow,cvfolds, knn_params)

#Bayesian tuning based on the results of the intial random tuning 
set.seed(my_seed)
knn_params<-knn_model %>%
  parameters() %>%
  update(neighbor = neighbor(range=c(1L,500L))))

knn_search <- model_tuning(knn_workflow, 
                             cvfolds, 
                             knn_params, 
                             knn_inital_search)       

print(autoplot(search, metric="roc_auc"))

print(search)

knn_best_model<-select_best(knn_search, metric = primary_metric)

set.seed(my_seed)
knn_finalworkflow<-knn_workflow %>%
  finalize_workflow(knn_best_model) %>%
  fit(df_train)


#multivariate adaptive regression spline

mars_model<-parsnip::mlp(num_terms = tune(),
                         prod_degree = tune())%>%
  set_engine("earth") %>%
  set_mode("classification")

set.seed(my_seed)
mars_params<-mars_model %>%
  parameters() %>%
  grid_max_entropy(size = 30)

mars_workflow<-workflow() %>%
  add_recipe(recipe) %>%
  add_model(mars_model)    

#inital random tuning
mars_inital_search <- model_tuning_grid(mars_workflow,cvfolds, mars_params)

#Bayesian tuning based on the results of the intial random tuning 
set.seed(my_seed)
mars_params<-mars_model %>%
  parameters() %>%
  update(num_terms = num_terms(range=c(1L,
            length(dplyr::select(juiced_df_train, 
                                 -outcome)))),
         prod_degree = prod_degree(range=c(1L,3L)))

mars_search <- model_tuning(mars_workflow, 
                             cvfolds, 
                             mars_params, 
                             mars_inital_search)       

print(autoplot(search, metric="roc_auc"))

print(search)

mars_best_model<-select_best(mars_search, metric = primary_metric)

set.seed(my_seed)
mars_finalworkflow<-mars_workflow %>%
  finalize_workflow(mars_best_model) %>%
  fit(df_train)


#decision tree

dt_model<-parsnip::mlp(cost_complexity = tune(),
                       tree_depth = tune(),
                       min_n = tune())%>%
  set_engine("rpart") %>%
  set_mode("classification")

set.seed(my_seed)
dt_params<-dt_model %>%
  parameters() %>%
  grid_max_entropy(size = 30)

dt_workflow<-workflow() %>%
  add_recipe(recipe) %>%
  add_model(dt_model)    

#inital random tuning
dt_inital_search <- model_tuning_grid(dt_workflow,cvfolds, dt_params)

#Bayesian tuning based on the results of the intial random tuning 
set.seed(my_seed)
dt_params<-dt_model %>%
  parameters() %>%
  update(cost_complexity= cost_complexity(range = c(-10L, -1L)),
         tree_depth = tree_depth(range = c(1L, 15L)),
         min_n = min_n(range= c(2L, 40L)))

dt_search <- model_tuning(dt_workflow, 
                            cvfolds, 
                            dt_params, 
                            dt_inital_search)       

print(autoplot(search, metric="roc_auc"))

print(search)

dt_best_model<-select_best(dt_search, metric = primary_metric)

set.seed(my_seed)
dt_finalworkflow<-dt_workflow %>%
  finalize_workflow(dt_best_model) %>%
  fit(df_train)
                                  
