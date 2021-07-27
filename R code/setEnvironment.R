#### setting up environment ####
packages<-c("dplyr","tidymodels", "recipes",
            "forcats","stringr",
            "glmnet", "ranger","xgboost","nnet",
            "pROC", "lattice","DALEX","ggplot2",
            "ggplotify","gridExtra","factoextra"
)

lapply(packages, library, character.only = TRUE)

my_prediction<-function(model, df_test, outcome)
{
  df <-predict(model, df_test) %>%
    bind_cols(predict(model, df_test, type="prob")) %>% 
    bind_cols(df_test[outcome])
  return(df)
}

metricOverCutoff<-function(probability = NULL, reference=NULL, positive ="TRUE", negative = "FALSE", metrics =NULL, threshold=c(0.01, 1, 0.05)){
  
  db <- data.frame(threshold=numeric(),
                   metric=character(),
                   estimate=numeric()) %>% as_tibble()
  for(i in threshold){
    class <- as.factor(ifelse(probability >= i, positive, negative))
    if(length(levels(class))==1){
      if(levels(class)==positive){
        levels(class)= c(positive, negative)
        class = ordered(class, levels = levels(as.factor(as.character(reference))))
        
      } else{
        levels(class)= c(negative,positive)
        class = ordered(class, levels = levels(as.factor(as.character(reference))))
      }
    }
    cm <- confusionMatrix(class,
                          as.factor(as.character(reference)), positive = positive)
    for (x in metrics){
      es<- switch( x,
                   "sensitivity" = round(cm$byClass[[1]],2),
                   "specificity" = round(cm$byClass[[2]],2),
                   "accuracy" = round(cm$overall[[1]],2),
                   "kappa" = round(cm$overall[[2]],2),
                   "ppv" = round(cm$byClass[[3]],2),
                   "npv" = round(cm$byClass[[4]],2),
                   "precision" = round(cm$byClass[[5]],2),
                   "recall" = round(cm$byClass[[6]],2),
                   "f1" = round(cm$byClass[[7]],2),
                   "j_index" = round((cm$byClass[[1]] + cm$byClass[[2]])-1,2),
                   "distance" = round((1 - cm$byClass[[1]]) ^ 2 + (1 - cm$byClass[[2]]) ^ 2,2)
      )
      db<-db %>% bind_rows(tibble(threshold = as.numeric(i),metric =  as.character(x), estimate = as.numeric(es)))
    }
  }
  return(db)
}

my_metrics<-metric_set(roc_auc, kap, sens)

primary_metric<-"roc_auc"

ctrl_bayes<-control_bayes(
  no_improve = 20,
  verbose = FALSE,
  save_pred = TRUE,
  #save_workflow =FALSE,
  seed = my_seed
)
model_tuning_grid<-function(workflow, cvfolds, params){
  library(doParallel)
  library(foreach)
  registerDoParallel(detectCores())
  set.seed(my_seed)    
  search<- tune_grid(
    object = workflow,
    resamples = cvfolds,
    grid = params,
    metrics = my_metrics,
    control = control_grid(verbose =FALSE,
                           save_pred = TRUE
    )
  )
  registerDoSEQ() 
  return(search)
}

model_tuning<-function(workflow, folds, params, inital_search){
  library(doParallel)
  library(foreach)
  registerDoParallel(detectCores())
  set.seed(my_seed)
  search<-tune_bayes(workflow,
                     resamples = folds,
                     param_info = params,
                     initial = inital_search,
                     iter = 40,
                     metrics = my_metrics,
                     control= ctrl_bayes 
                     #control= control_stack_bayes()     
  )
  registerDoSEQ()       
  return(search)    
}
