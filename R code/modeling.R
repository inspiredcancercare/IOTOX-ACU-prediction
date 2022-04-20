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
  update(
    penalty = penalty(range(c(-10L,0))),
    mixture= mixture(range(c(0.05,1)))) %>%
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
    mixture= mixture(range(c(0.05,1))))


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
    mtry = mtry(range = c(1L,length(dplyr::select(juiced_df_train, -outcome)))),
    trees = trees(range(c(1000L:2500L))),
    min_n = min_n(range(c(10L:300L))))%>%
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
    trees = trees(range(c(1000L:2500L))),
    min_n = min_n(range(c(10L:300L))))

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
  update(mtry = mtry(range = c(1L,length(dplyr::select(juiced_df_train, -outcome)))),
            trees = trees(range = c(500L, 2500L)),
            min_n = min_n(range = c(2L, 30L)),
            tree_depth = tree_depth(range = c(5L,30L)),
            learn_rate = learn_rate(range = c(-15L,-5L)),
            loss_reduction = loss_reduction(range = c(-15L, -5L)))%>%
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
            trees = trees(range = c(500L, 2500L)),
            min_n = min_n(range = c(2L, 30L)),
            tree_depth = tree_depth(range = c(5L,30L)),
            learn_rate = learn_rate(range = c(-15L,-5L)),
            loss_reduction = loss_reduction(range = c(-15L, -5L)))

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
  update(hidden_units = hidden_units(range(c(1L,10L))),
         penalty = penalty(range(c(-10L,0))),
         epochs = epochs(range(c(1L,1000L)))) %>%
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
  update(hidden_units = hidden_units(range(c(1L,10L))),
         penalty = penalty(range(c(-10L,0))),
         epochs = epochs(range(c(1L,1000L)))) 

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
