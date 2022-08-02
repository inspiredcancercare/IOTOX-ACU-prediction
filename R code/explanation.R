#### Model agnostic explanations ######
pred_wrapper<-  function(object, newdata)  {
  name<-object$fit$actions$model$spec$engine
  #print(name)
  cali<-switch(name,
               "glmnet" = calibration$glm,
               "earth" = calibration$mars,
               "rpart" = calibration$dt,
               "xgboost"= calibration$gbm,
               "nnet" = calibration$shlnn,
               "kernlab"= calibration$svm,
               "ranger" = calibration$rf,
               "kknn"= calibration$knn)
  #print(cali)
  pred <- predict(object, newdata, type = "prob")
  #print(pred)
  uncali_pred <- data.frame(Pred = pred$.pred_TRUE)
  response <- predict(cali, uncali_pred, type = "response")
  #print(response)
  return(response)
}   

set.seed(my_seed)
df_periCOVID_explain_prep<-recipe(formula, df_periCOVID) %>%
  step_medianimpute(all_numeric(), -recipes::all_outcomes()) %>%
  step_modeimpute(all_nominal(), -recipes::all_outcomes()) %>%
  prep()

df_periCOVID_bake <- bake(df_periCOVID_explain_prep, df_periCOVID) %>% as.data.frame()

#Creat expainer for each algo
#Logistic regression with elastic net penalty
lrenp_model_fit <- lrenp_finalworkflow

lrenp_explainer<-DALEXtra::explain_tidymodels(
  model= lrenp_model_fit,
  data = dplyr::select(df_periCOVID_bake,-outcome),
  y = as.vector(as.numeric(ifelse(df_periCOVID_bake[,outcome]=="TRUE",1,0))),
  predict_function = pred_wrapper,
  label = "Logistic regression with elastic net penalty")

#Randpm forest
rf_model_fit <- rf_finalworkflow

rf_explainer<-DALEXtra::explain_tidymodels(
  model= rf_model_fit,
  data = dplyr::select(df_periCOVID_bake,-outcome),
  y = as.vector(as.numeric(ifelse(df_periCOVID_bake[,outcome]=="TRUE",1,0))),
  predict_function = pred_wrapper,
  label = "Random forest")

#Extreme gradient boosting trees
xgb_model_fit <- xgb_finalworkflow

xgb_explainer<-DALEXtra::explain_tidymodels(
  model= xgb_model_fit,
  data = dplyr::select(df_periCOVID_bake,-outcome),
  y = as.vector(as.numeric(ifelse(df_periCOVID_bake[,outcome]=="TRUE",1,0))),
  predict_function = pred_wrapper,
  label = "Extreme graident boosting trees")

#Single hidden layer neural network
shlnn_model_fit <- shlnn_finalworkflow

shlnn_explainer<-DALEXtra::explain_tidymodels(
  model= shlnn_model_fit,
  data = dplyr::select(df_periCOVID_bake,-outcome),
  y = as.vector(as.numeric(ifelse(df_periCOVID_bake[,outcome]=="TRUE",1,0))),
  predict_function = pred_wrapper,
  label = "Single hidden layer neural network")

#Support vector machine
svm_model_fit <- svm_finalworkflow

svm_explainer<-DALEXtra::explain_tidymodels(
  model= svm_model_fit,
  data = dplyr::select(df_periCOVID_bake,-outcome),
  y = as.vector(as.numeric(ifelse(df_periCOVID_bake[,outcome]=="TRUE",1,0))),
  predict_function = pred_wrapper,
  label = "Support vector machine")

#k nearest neighbors
knn_model_fit <- knn_finalworkflow

knn_explainer<-DALEXtra::explain_tidymodels(
  model= knn_model_fit,
  data = dplyr::select(df_periCOVID_bake,-outcome),
  y = as.vector(as.numeric(ifelse(df_periCOVID_bake[,outcome]=="TRUE",1,0))),
  predict_function = pred_wrapper,
  label = "k nearest neighbors")

#Multivariate adaptive regression spline
mars_model_fit <- mars_finalworkflow

mars_explainer<-DALEXtra::explain_tidymodels(
  model= mars_model_fit,
  data = dplyr::select(df_periCOVID_bake,-outcome),
  y = as.vector(as.numeric(ifelse(df_periCOVID_bake[,outcome]=="TRUE",1,0))),
  predict_function = pred_wrapper,
  label = "Multivariate adaptive regression spline")

#Decision tree
dt_model_fit <- dt_finalworkflow

dt_explainer<-DALEXtra::explain_tidymodels(
  model= dt_model_fit,
  data = dplyr::select(df_periCOVID_bake,-outcome),
  y = as.vector(as.numeric(ifelse(df_periCOVID_bake[,outcome]=="TRUE",1,0))),
  predict_function = pred_wrapper,
  label = "Decision tree")


#### Variable importance analysis ####

#Logistic regression with elastic net penalty
lrenp_vi<-variable_importance(lrenp_explainer,
                              type="variable_importance") 

lrenp_vip <- plot(lrenp_vi, 
                  show_boxplots=FALSE, 
                  subtitle = "",
                  bar_width = 8,
                  max_vars= 10) +
  ggtitle("Variable importance")+
  labs(title="", tag="")+
  ylab("1-AUROC ")+
  scale_fill_manual(values = "black")+
  theme(panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())+
  theme(panel.border= element_blank())+
  theme(axis.line.x = element_line(color="black", size = 0.3),     
        axis.line.y = element_line(color="black", size = 0.3))+
  theme(axis.ticks.x = element_line(colour = "black", size = 0.3),
        axis.ticks.y = element_line(colour = "black", size = 0.3))+
  theme(axis.text.x = element_text(colour = "black", size = 8),
        axis.text.y = element_text(colour = "black"),
        axis.title.x = element_text(colour = "black"),
        axis.title.y = element_text(colour = "black"))

#Random forest
rf_vi <- variable_importance(rf_explainer,
                             #loss_function = loss_auc,
                             type="variable_importance") 

rf_vip <- plot(rf_vi, 
               show_boxplots=FALSE, 
               subtitle = "",
               bar_width = 8,
               max_vars= 10) +
  ggtitle("Variable importance")+
  labs(title="", tag="")+
  ylab("1-AUROC ")+
  scale_fill_manual(values = "black")+
  theme(panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())+
  theme(panel.border= element_blank())+
  theme(axis.line.x = element_line(color="black", size = 0.3),     
        axis.line.y = element_line(color="black", size = 0.3))+
  theme(axis.ticks.x = element_line(colour = "black", size = 0.3),
        axis.ticks.y = element_line(colour = "black", size = 0.3))+
  theme(axis.text.x = element_text(colour = "black", size = 8),
        axis.text.y = element_text(colour = "black"),
        axis.title.x = element_text(colour = "black"),
        axis.title.y = element_text(colour = "black"))

#Extreme gradient boosting trees
xgb_vi<-variable_importance(xgb_explainer,
                            #loss_function = loss_auc,
                            type="variable_importance") 

xgb_vip <- plot(xgb_vi, 
                show_boxplots=FALSE, 
                subtitle = "",
                bar_width = 8,
                max_vars= 10) +
  ggtitle("Variable importance")+
  labs(title="", tag="")+
  ylab("1-AUROC ")+
  scale_fill_manual(values = "black")+
  theme(panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())+
  theme(panel.border= element_blank())+
  theme(axis.line.x = element_line(color="black", size = 0.3),     
        axis.line.y = element_line(color="black", size = 0.3))+
  theme(axis.ticks.x = element_line(colour = "black", size = 0.3),
        axis.ticks.y = element_line(colour = "black", size = 0.3))+
  theme(axis.text.x = element_text(colour = "black", size = 8),
        axis.text.y = element_text(colour = "black"),
        axis.title.x = element_text(colour = "black"),
        axis.title.y = element_text(colour = "black"))

#Single hidden layer neural network
shlnn_vi<-variable_importance(shlnn_explainer,
                              type="variable_importance") 

shlnn_vip <- plot(shlnn_vi, 
                  show_boxplots=FALSE, 
                  subtitle = "",
                  bar_width = 8,
                  max_vars= 10) +
  ggtitle("Variable importance")+
  labs(title="", tag="")+
  ylab("1-AUROC ")+
  scale_fill_manual(values = "black")+
  theme(panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())+
  theme(panel.border= element_blank())+
  theme(axis.line.x = element_line(color="black", size = 0.3),     
        axis.line.y = element_line(color="black", size = 0.3))+
  theme(axis.ticks.x = element_line(colour = "black", size = 0.3),
        axis.ticks.y = element_line(colour = "black", size = 0.3))+
  theme(axis.text.x = element_text(colour = "black", size = 8),
        axis.text.y = element_text(colour = "black"),
        axis.title.x = element_text(colour = "black"),
        axis.title.y = element_text(colour = "black"))

#Support vector machine
svm_vi <- variable_importance(svm_explainer,
                             type="variable_importance") 

svm_vip <- plot(svm_vi, 
               show_boxplots=FALSE, 
               subtitle = "",
               bar_width = 8,
               max_vars= 10) +
  ggtitle("Variable importance")+
  labs(title="", tag="")+
  ylab("1-AUROC ")+
  scale_fill_manual(values = "black")+
  theme(panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())+
  theme(panel.border= element_blank())+
  theme(axis.line.x = element_line(color="black", size = 0.3),     
        axis.line.y = element_line(color="black", size = 0.3))+
  theme(axis.ticks.x = element_line(colour = "black", size = 0.3),
        axis.ticks.y = element_line(colour = "black", size = 0.3))+
  theme(axis.text.x = element_text(colour = "black", size = 8),
        axis.text.y = element_text(colour = "black"),
        axis.title.x = element_text(colour = "black"),
        axis.title.y = element_text(colour = "black"))


#k nearest neighbors
knn_vi <- variable_importance(knn_explainer,
                             type="variable_importance") 

knn_vip <- plot(knn_vi, 
               show_boxplots=FALSE, 
               subtitle = "",
               bar_width = 8,
               max_vars= 10) +
  ggtitle("Variable importance")+
  labs(title="", tag="")+
  ylab("1-AUROC ")+
  scale_fill_manual(values = "black")+
  theme(panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())+
  theme(panel.border= element_blank())+
  theme(axis.line.x = element_line(color="black", size = 0.3),     
        axis.line.y = element_line(color="black", size = 0.3))+
  theme(axis.ticks.x = element_line(colour = "black", size = 0.3),
        axis.ticks.y = element_line(colour = "black", size = 0.3))+
  theme(axis.text.x = element_text(colour = "black", size = 8),
        axis.text.y = element_text(colour = "black"),
        axis.title.x = element_text(colour = "black"),
        axis.title.y = element_text(colour = "black"))


#Multivariate adaptive regression spline
mars_vi <- variable_importance(mars_explainer,
                             type="variable_importance") 

mars_vip <- plot(mars_vi, 
               show_boxplots=FALSE, 
               subtitle = "",
               bar_width = 8,
               max_vars= 10) +
  ggtitle("Variable importance")+
  labs(title="", tag="")+
  ylab("1-AUROC ")+
  scale_fill_manual(values = "black")+
  theme(panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())+
  theme(panel.border= element_blank())+
  theme(axis.line.x = element_line(color="black", size = 0.3),     
        axis.line.y = element_line(color="black", size = 0.3))+
  theme(axis.ticks.x = element_line(colour = "black", size = 0.3),
        axis.ticks.y = element_line(colour = "black", size = 0.3))+
  theme(axis.text.x = element_text(colour = "black", size = 8),
        axis.text.y = element_text(colour = "black"),
        axis.title.x = element_text(colour = "black"),
        axis.title.y = element_text(colour = "black"))

#Decision tree
dt_vi <- variable_importance(dt_explainer,
                             type="variable_importance") 

dt_vip <- plot(dt_vi, 
               show_boxplots=FALSE, 
               subtitle = "",
               bar_width = 8,
               max_vars= 10) +
  ggtitle("Variable importance")+
  labs(title="", tag="")+
  ylab("1-AUROC ")+
  scale_fill_manual(values = "black")+
  theme(panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())+
  theme(panel.border= element_blank())+
  theme(axis.line.x = element_line(color="black", size = 0.3),     
        axis.line.y = element_line(color="black", size = 0.3))+
  theme(axis.ticks.x = element_line(colour = "black", size = 0.3),
        axis.ticks.y = element_line(colour = "black", size = 0.3))+
  theme(axis.text.x = element_text(colour = "black", size = 8),
        axis.text.y = element_text(colour = "black"),
        axis.title.x = element_text(colour = "black"),
        axis.title.y = element_text(colour = "black"))


####Shapley additive explanation analysis####
set.seed(my_seed)
num_case<- 1
case_id <- sample(1:nrow(df_periCOVID), num_case)
obs<- df_periCOVID[case_id,]

set.seed(my_seed)
obs<-bake(df_periCOVID_explain_prep, obs)%>% as.data.frame()

#logistic regression with elastic net penalty
lrenp_important_variable<-lrenp_vi %>% 
  as_tibble() %>%
  filter(variable %in% predictors) %>%
  group_by(variable) %>%
  summarize(mean_dropout_loss= mean(dropout_loss)) %>%
  arrange(desc(mean_dropout_loss),desc(variable)) %>%
  head(n=10) %>%
  pull(variable) %>% 
  as.vector()

set.seed(my_seed)
lrenp_shap<- predict_parts(lrenp_explainer, new_observation = obs, type = "shap", B = 50)

lrenp_shap %>%
  data.frame() %>%
  filter(variable_name %in% lrenp_important_variable) %>%
  group_by(variable) %>% 
  summarise(contribution=mean(contribution)) %>%
  filter(contribution!=0)%>%
  mutate(sign=ifelse(contribution!=0,ifelse(contribution>0,2,1),0)) %>%
  mutate(abs_c=abs(contribution)) %>%
  mutate(variable=fct_reorder(variable, abs_c, max)) %>%
  arrange(desc(abs_c)) %>%
  print() %>%
  ggplot(aes(x=variable, y=contribution, fill=sign))+ 
  geom_bar(stat="identity", width= 0.8)+
  geom_hline(yintercept=0, linetype="dashed", color = 'grey50')+
  scale_y_continuous(name = "Contribution", limits = c(-0.25,0.25))+
  scale_x_discrete(name= "Variables")+
  coord_flip()+
  theme(legend.position = "none",
        axis.text.y   = element_text(size=12, colour="black"),
        axis.text.x   = element_text(size=12, colour="black"),
        panel.background = element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"))

#Random forest
rf_important_variable<-rf_vi %>% 
  as_tibble() %>%
  filter(variable %in% predictors) %>%
  group_by(variable) %>%
  summarize(mean_dropout_loss= mean(dropout_loss)) %>%
  arrange(desc(mean_dropout_loss),desc(variable)) %>%
  head(n=10) %>%
  pull(variable) %>% 
  as.vector()

set.seed(my_seed)
rf_shap<- predict_parts(rf_explainer, new_observation = obs, type = "shap", B = 50)

rf_shap %>%
  data.frame() %>%
  filter(variable_name %in% rf_important_variable) %>%
  group_by(variable) %>% 
  summarise(contribution=mean(contribution)) %>%
  filter(contribution!=0)%>%
  mutate(sign=ifelse(contribution!=0,ifelse(contribution>0,2,1),0)) %>%
  mutate(abs_c=abs(contribution)) %>%
  mutate(variable=fct_reorder(variable, abs_c, max)) %>%
  arrange(desc(abs_c)) %>%
  print() %>%
  ggplot(aes(x=variable, y=contribution, fill=sign))+ 
  geom_bar(stat="identity", width= 0.8)+
  geom_hline(yintercept=0, linetype="dashed", color = 'grey50')+
  scale_y_continuous(name = "Contribution", limits = c(-0.25,0.25))+
  scale_x_discrete(name= "Variables")+
  coord_flip()+
  theme(legend.position = "none",
        axis.text.y   = element_text(size=12, colour="black"),
        axis.text.x   = element_text(size=12, colour="black"),
        panel.background = element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"))

#Extreme gradient boosting trees
xgb_important_variable<-xgb_vi %>% 
  as_tibble() %>%
  filter(variable %in% predictors) %>%
  group_by(variable) %>%
  summarize(mean_dropout_loss= mean(dropout_loss)) %>%
  arrange(desc(mean_dropout_loss),desc(variable)) %>%
  head(n=10) %>%
  pull(variable) %>% 
  as.vector()

set.seed(my_seed)
xgb_shap<- predict_parts(xgb_explainer, new_observation = obs, type = "shap", B = 50)

xgb_shap %>%
  data.frame() %>%
  filter(variable_name %in% xgb_important_variable) %>%
  group_by(variable) %>% 
  summarise(contribution=mean(contribution)) %>%
  filter(contribution!=0)%>%
  mutate(sign=ifelse(contribution!=0,ifelse(contribution>0,2,1),0)) %>%
  mutate(abs_c=abs(contribution)) %>%
  mutate(variable=fct_reorder(variable, abs_c, max)) %>%
  arrange(desc(abs_c)) %>%
  print() %>%
  ggplot(aes(x=variable, y=contribution, fill=sign))+ 
  geom_bar(stat="identity", width= 0.8)+
  geom_hline(yintercept=0, linetype="dashed", color = 'grey50')+
  scale_y_continuous(name = "Contribution", limits = c(-0.25,0.25))+
  scale_x_discrete(name= "Variables")+
  coord_flip()+
  theme(legend.position = "none",
        axis.text.y   = element_text(size=12, colour="black"),
        axis.text.x   = element_text(size=12, colour="black"),
        panel.background = element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"))
#Single hidden layer neural network
shlnn_important_variable<-shlnn_vi %>% 
  as_tibble() %>%
  filter(variable %in% predictors) %>%
  group_by(variable) %>%
  summarize(mean_dropout_loss= mean(dropout_loss)) %>%
  arrange(desc(mean_dropout_loss),desc(variable)) %>%
  head(n=10) %>%
  pull(variable) %>% 
  as.vector()

set.seed(my_seed)
shlnn_shap<- predict_parts(shlnn_explainer, new_observation = obs, type = "shap", B = 50)

shlnn_shap %>%
  data.frame() %>%
  filter(variable_name %in% shlnn_important_variable) %>%
  group_by(variable) %>% 
  summarise(contribution=mean(contribution)) %>%
  filter(contribution!=0)%>%
  mutate(sign=ifelse(contribution!=0,ifelse(contribution>0,2,1),0)) %>%
  mutate(abs_c=abs(contribution)) %>%
  mutate(variable=fct_reorder(variable, abs_c, max)) %>%
  arrange(desc(abs_c)) %>%
  print() %>%
  ggplot(aes(x=variable, y=contribution, fill=sign))+ 
  geom_bar(stat="identity", width= 0.8)+
  geom_hline(yintercept=0, linetype="dashed", color = 'grey50')+
  scale_y_continuous(name = "Contribution", limits = c(-0.25,0.25))+
  scale_x_discrete(name= "Variables")+
  coord_flip()+
  theme(legend.position = "none",
        axis.text.y   = element_text(size=12, colour="black"),
        axis.text.x   = element_text(size=12, colour="black"),
        panel.background = element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"))

#Support vector machine
svm_important_variable<-svm_vi %>% 
  as_tibble() %>%
  filter(variable %in% predictors) %>%
  group_by(variable) %>%
  summarize(mean_dropout_loss= mean(dropout_loss)) %>%
  arrange(desc(mean_dropout_loss),desc(variable)) %>%
  head(n=10) %>%
  pull(variable) %>% 
  as.vector()

set.seed(my_seed)
svm_shap<- predict_parts(svm_explainer, new_observation = obs, type = "shap", B = 50)

svm_shap %>%
  data.frame() %>%
  filter(variable_name %in% svm_important_variable) %>%
  group_by(variable) %>% 
  summarise(contribution=mean(contribution)) %>%
  filter(contribution!=0)%>%
  mutate(sign=ifelse(contribution!=0,ifelse(contribution>0,2,1),0)) %>%
  mutate(abs_c=abs(contribution)) %>%
  mutate(variable=fct_reorder(variable, abs_c, max)) %>%
  arrange(desc(abs_c)) %>%
  print() %>%
  ggplot(aes(x=variable, y=contribution, fill=sign))+ 
  geom_bar(stat="identity", width= 0.8)+
  geom_hline(yintercept=0, linetype="dashed", color = 'grey50')+
  scale_y_continuous(name = "Contribution", limits = c(-0.25,0.25))+
  scale_x_discrete(name= "Variables")+
  coord_flip()+
  theme(legend.position = "none",
        axis.text.y   = element_text(size=12, colour="black"),
        axis.text.x   = element_text(size=12, colour="black"),
        panel.background = element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"))


#k nearst neighbor
knn_important_variable<-knn_vi %>% 
  as_tibble() %>%
  filter(variable %in% predictors) %>%
  group_by(variable) %>%
  summarize(mean_dropout_loss= mean(dropout_loss)) %>%
  arrange(desc(mean_dropout_loss),desc(variable)) %>%
  head(n=10) %>%
  pull(variable) %>% 
  as.vector()

set.seed(my_seed)
knn_shap<- predict_parts(knn_explainer, new_observation = obs, type = "shap", B = 50)

knn_shap %>%
  data.frame() %>%
  filter(variable_name %in% knn_important_variable) %>%
  group_by(variable) %>% 
  summarise(contribution=mean(contribution)) %>%
  filter(contribution!=0)%>%
  mutate(sign=ifelse(contribution!=0,ifelse(contribution>0,2,1),0)) %>%
  mutate(abs_c=abs(contribution)) %>%
  mutate(variable=fct_reorder(variable, abs_c, max)) %>%
  arrange(desc(abs_c)) %>%
  print() %>%
  ggplot(aes(x=variable, y=contribution, fill=sign))+ 
  geom_bar(stat="identity", width= 0.8)+
  geom_hline(yintercept=0, linetype="dashed", color = 'grey50')+
  scale_y_continuous(name = "Contribution", limits = c(-0.25,0.25))+
  scale_x_discrete(name= "Variables")+
  coord_flip()+
  theme(legend.position = "none",
        axis.text.y   = element_text(size=12, colour="black"),
        axis.text.x   = element_text(size=12, colour="black"),
        panel.background = element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"))

#Multivariate adaptive regression spline
mars_important_variable<-mars_vi %>% 
  as_tibble() %>%
  filter(variable %in% predictors) %>%
  group_by(variable) %>%
  summarize(mean_dropout_loss= mean(dropout_loss)) %>%
  arrange(desc(mean_dropout_loss),desc(variable)) %>%
  head(n=10) %>%
  pull(variable) %>% 
  as.vector()

set.seed(my_seed)
mars_shap<- predict_parts(mars_explainer, new_observation = obs, type = "shap", B = 50)

mars_shap %>%
  data.frame() %>%
  filter(variable_name %in% mars_important_variable) %>%
  group_by(variable) %>% 
  summarise(contribution=mean(contribution)) %>%
  filter(contribution!=0)%>%
  mutate(sign=ifelse(contribution!=0,ifelse(contribution>0,2,1),0)) %>%
  mutate(abs_c=abs(contribution)) %>%
  mutate(variable=fct_reorder(variable, abs_c, max)) %>%
  arrange(desc(abs_c)) %>%
  print() %>%
  ggplot(aes(x=variable, y=contribution, fill=sign))+ 
  geom_bar(stat="identity", width= 0.8)+
  geom_hline(yintercept=0, linetype="dashed", color = 'grey50')+
  scale_y_continuous(name = "Contribution", limits = c(-0.25,0.25))+
  scale_x_discrete(name= "Variables")+
  coord_flip()+
  theme(legend.position = "none",
        axis.text.y   = element_text(size=12, colour="black"),
        axis.text.x   = element_text(size=12, colour="black"),
        panel.background = element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"))


#Decision tree
dt_important_variable<-dt_vi %>% 
  as_tibble() %>%
  filter(variable %in% predictors) %>%
  group_by(variable) %>%
  summarize(mean_dropout_loss= mean(dropout_loss)) %>%
  arrange(desc(mean_dropout_loss),desc(variable)) %>%
  head(n=10) %>%
  pull(variable) %>% 
  as.vector()

set.seed(my_seed)
dt_shap<- predict_parts(dt_explainer, new_observation = obs, type = "shap", B = 50)

dt_shap %>%
  data.frame() %>%
  filter(variable_name %in% dt_important_variable) %>%
  group_by(variable) %>% 
  summarise(contribution=mean(contribution)) %>%
  filter(contribution!=0)%>%
  mutate(sign=ifelse(contribution!=0,ifelse(contribution>0,2,1),0)) %>%
  mutate(abs_c=abs(contribution)) %>%
  mutate(variable=fct_reorder(variable, abs_c, max)) %>%
  arrange(desc(abs_c)) %>%
  print() %>%
  ggplot(aes(x=variable, y=contribution, fill=sign))+ 
  geom_bar(stat="identity", width= 0.8)+
  geom_hline(yintercept=0, linetype="dashed", color = 'grey50')+
  scale_y_continuous(name = "Contribution", limits = c(-0.25,0.25))+
  scale_x_discrete(name= "Variables")+
  coord_flip()+
  theme(legend.position = "none",
        axis.text.y   = element_text(size=12, colour="black"),
        axis.text.x   = element_text(size=12, colour="black"),
        panel.background = element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"))
