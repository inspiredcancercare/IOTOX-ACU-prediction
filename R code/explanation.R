#### Model agnostic explanations ######
pred_wrapper<-  function(object, newdata)  {
  name<-object$spec$engine
  cali<-switch(name,
               "glmnet" = lrenp_calibrate_model,
               "ranger" = rf_calibrate_model,
               "xgboost"= xgb_calibrate_model,
               "nnet" = shlnn_calibrate_model
  )
  #print(cali)
  pred <- predict(object, newdata, type = "prob")
  uncali_pred <- data.frame(Pred = pred$.pred_TRUE)
  response <- predict(cali, uncali_pred, type = "response")
  return(response)
}

set.seed(my_seed)
df_periCOVID_bake <- bake(prep(recipe, df_train), df_periCOVID) 

#Creat expainer for each algo
#Logistic regression with elastic net penalty
lrenp_model_fit <- lrenp_finalworkflow %>% pull_workflow_fit() 

lrenp_explainer<-DALEX::explain(
  model= lrenp_model_fit,
  data = dplyr::select(df_periCOVID_bake,-outcome),
  y = as.vector(as.numeric(ifelse(df_periCOVID_bake[,outcome]=="TRUE",1,0))),
  predict_function = pred_wrapper,
  label = "Logistic regression with elastic net penalty")

#Randpm forest
rf_model_fit <- rf_finalworkflow %>% pull_workflow_fit() 

rf_explainer<-DALEX::explain(
  model= rf_model_fit,
  data = dplyr::select(df_periCOVID_bake,-outcome),
  y = as.vector(as.numeric(ifelse(df_periCOVID_bake[,outcome]=="TRUE",1,0))),
  predict_function = pred_wrapper,
  label = "Random forest")

#Extreme gradient boosting trees
xgb_model_fit <- xgb_finalworkflow %>% pull_workflow_fit() 

xgb_explainer<-DALEX::explain(
  model= xgb_model_fit,
  data = dplyr::select(df_periCOVID_bake,-outcome),
  y = as.vector(as.numeric(ifelse(df_periCOVID_bake[,outcome]=="TRUE",1,0))),
  predict_function = pred_wrapper,
  label = "Extreme graident boosting trees")

#Single hidden layer neural network
shlnn_model_fit <- shlnn_finalworkflow %>% pull_workflow_fit() 

shlnn_explainer<-DALEX::explain(
  model= shlnn_model_fit,
  data = dplyr::select(df_periCOVID_bake,-outcome),
  y = as.vector(as.numeric(ifelse(df_periCOVID_bake[,outcome]=="TRUE",1,0))),
  predict_function = pred_wrapper,
  label = "Logistic regression with elastic net penalty")


#### Variable importance analysis ####

loss_auc <- function(observed, predicted){
  pred <- data.frame(fitted.values = predicted,
                     y = observed)
  #print(pred)
  pred_sorted <- pred[order(pred$fitted.values, decreasing = TRUE), ]
  roc_y <- factor(pred_sorted$y)
  levels <- levels(roc_y)
  x <- cumsum(roc_y == levels[1])/sum(roc_y == levels[1])
  y <- cumsum(roc_y == levels[2])/sum(roc_y == levels[2])
  auc <- round(sum((x[2:length(roc_y)]  -x[1:length(roc_y)-1]) * y[2:length(roc_y)]),2)
  #print(1-auc)
  return(1 - auc)
}
attr(loss_auc, "loss_name") <- "One minus AUC"

#Logistic regression with elastic net penalty
lrenp_vi<-DALEX::model_parts(lrenp_explainer,
                            loss_function = loss_auc,
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
rf_vi<-DALEX::model_parts(rf_explainer,
                             loss_function = loss_auc,
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
xgb_vi<-DALEX::model_parts(xgb_explainer,
                             loss_function = loss_auc,
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
shlnn_vi<-DALEX::model_parts(shlnn_explainer,
                             loss_function = loss_auc,
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


#### Shapely additive explanation analysis ####
####Plotting variable importance####
lrenp_vi<-variable_importance(lrenp_explainer,type="raw")

plot(lrenp_vi, max_vars = 10, show_boxplots =FALSE, subtitle="")+
  ggtitle("")+
  ylab("1-AUC")

rf_vi<-variable_importance(rf_explainer,type="raw")

plot(rf_vi, max_vars = 10, show_boxplots =FALSE, subtitle="")+
  ggtitle("")+
  ylab("1-AUC")

xgb_vi<-variable_importance(xgb_explainer,type="raw") 

plot(xgb_vi, max_vars = 10, show_boxplots =FALSE, subtitle="")+
  ggtitle("")+
  ylab("1-AUC")

shlnn_vi<-variable_importance(shlnn_explainer,type="raw")

plot(shlnn_vi, max_vars = 10, show_boxplots =FALSE, subtitle="")+
  ggtitle("")+
  ylab("1-AUC")


####Shapley additive explanation analysis####
set.seed(my_seed)
num_case<- 1
case_id <- sample(1:nrow(df_periCOVID), num_case)
obs<- df_periCOVID[case_id,]

set.seed(my_seed)
obs<-bake(df_prep, obs)%>% as.data.frame()

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