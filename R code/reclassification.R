#### Platt scaling calibration ####
#logistic regression
lr_pred_train<-my_prediction(lr_workflow, df_train, outcome) %>% 
  dplyr::select(-.pred_class,-.pred_FALSE) %>% 
  as.data.frame()

colnames(pred_train)<-c("Pred","Obs")

set.seed(my_seed)
lr_calibrate_model <- glm(Obs~Pred, data = lr_pred_train, family = binomial) 

#pre-COVID
uncali_pred<- data.frame(Pred = lr_predictions$.pred_TRUE)

cali_pred <- predict(lr_calibrate_model, uncali_pred, type = "response")

lr_predictions<-lr_predictions %>% 
  bind_cols(cali_pred_TRUE = cali_pred) 

#Peri-COVID
uncali_pred<- data.frame(Pred = lr_predictions_periCOVID$.pred_TRUE)

cali_pred <- predict(lr_calibrate_model, uncali_pred, type = "response")

lr_predictions_periCOVID<-lr_predictions_periCOVID %>% 
  bind_cols(cali_pred_TRUE = cali_pred) 

formula = as.formula(paste(outcome, "cali_pred_TRUE", sep = "~"))
c <- calibration(formula, 
                 data= lr_predictions_periCOVID, 
                 class = "TRUE",
                 cuts = 30)$data

lr_calibraion_plot<-ggplot() + 
  geom_line(data = c, aes(midpoint, Percent, linetype = model)) +
  scale_linetype_manual(values=c("solid"))+
  geom_point(data = c, aes(midpoint, Percent, shape = model), size = 2) +
  geom_line(aes(c(0, 100), c(0, 100)), linetype = 2, 
            color = 'grey50')+
  theme(axis.text.y   = element_text(size=12),
        axis.text.x   = element_text(size=12),
        axis.title.y  = element_text(size=12),
        axis.title.x  = element_text(size=12),
        panel.background = element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        legend.position="none")+
  xlab("Predicted probability")+
  ylab("Observed event percent")

#ECOG logistic regression
ecog_pred_train<-my_prediction(ecog_workflow, df_train, outcome) %>% 
  dplyr::select(-.pred_class,-.pred_FALSE) %>% 
  as.data.frame()

colnames(pred_train)<-c("Pred","Obs")

set.seed(my_seed)
ecog_calibrate_model <- glm(Obs~Pred, data = ecog_pred_train, family = binomial) 

#Pre-COVID
uncali_pred<- data.frame(Pred = ecog_predictions$.pred_TRUE)

cali_pred <- predict(ecog_calibrate_model, uncali_pred, type = "response")

ecog_predictions <- ecog_predictions %>% 
  bind_cols(cali_pred_TRUE = cali_pred) 

#Peri-COVID
uncali_pred<- data.frame(Pred = ecog_predictions_periCOVID$.pred_TRUE)

cali_pred <- predict(ecog_calibrate_model, uncali_pred, type = "response")

ecog_predictions_periCOVID<-ecog_predictions_periCOVID %>% 
  bind_cols(cali_pred_TRUE = cali_pred) 

formula = as.formula(paste(outcome, "cali_pred_TRUE", sep = "~"))
c <- calibration(formula, 
                 data= ecog_predictions_periCOVID, 
                 class = "TRUE",
                 cuts = 30)$data

ecog_calibraion_plot<-ggplot() + 
  geom_line(data = c, aes(midpoint, Percent, linetype = model)) +
  scale_linetype_manual(values=c("solid"))+
  geom_point(data = c, aes(midpoint, Percent, shape = model), size = 2) +
  geom_line(aes(c(0, 100), c(0, 100)), linetype = 2, 
            color = 'grey50')+
  theme(axis.text.y   = element_text(size=12),
        axis.text.x   = element_text(size=12),
        axis.title.y  = element_text(size=12),
        axis.title.x  = element_text(size=12),
        panel.background = element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        legend.position="none")+
  xlab("Predicted probability")+
  ylab("Observed event percent")

#logistic regression with elastic net penalty
lrenp_pred_train<-my_prediction(lrenp_finalworkflow, df_train, outcome) %>% 
  dplyr::select(-.pred_class,-.pred_FALSE) %>% 
  as.data.frame()

colnames(pred_train)<-c("Pred","Obs")

set.seed(my_seed)
lrenp_calibrate_model <- glm(Obs~Pred, data = lrenp_pred_train, family = binomial) 

#Pre-COVID
uncali_pred<- data.frame(Pred = lrenp_predictions$.pred_TRUE)

cali_pred <- predict(lrenp_calibrate_model, uncali_pred, type = "response")

lrenp_predictions <- lrenp_predictions %>% 
  bind_cols(cali_pred_TRUE = cali_pred) 

#Peri-COVID
uncali_pred<- data.frame(Pred = lrenp_predictions_periCOVID$.pred_TRUE)

cali_pred <- predict(lrenp_calibrate_model, uncali_pred, type = "response")

lrenp_predictions_periCOVID<-lrenp_predictions_periCOVID %>% 
  bind_cols(cali_pred_TRUE = cali_pred) 

formula = as.formula(paste(outcome, "cali_pred_TRUE", sep = "~"))
c <- calibration(formula, 
                 data= lrenp_predictions_periCOVID, 
                 class = "TRUE",
                 cuts = 30)$data

lrenp_calibraion_plot<-ggplot() + 
  geom_line(data = c, aes(midpoint, Percent, linetype = model)) +
  scale_linetype_manual(values=c("solid"))+
  geom_point(data = c, aes(midpoint, Percent, shape = model), size = 2) +
  geom_line(aes(c(0, 100), c(0, 100)), linetype = 2, 
            color = 'grey50')+
  theme(axis.text.y   = element_text(size=12),
        axis.text.x   = element_text(size=12),
        axis.title.y  = element_text(size=12),
        axis.title.x  = element_text(size=12),
        panel.background = element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        legend.position="none")+
  xlab("Predicted probability")+
  ylab("Observed event percent")

#Random forest
rf_pred_train<-my_prediction(rf_finalworkflow, df_train, outcome) %>% 
  dplyr::select(-.pred_class,-.pred_FALSE) %>% 
  as.data.frame()

colnames(pred_train)<-c("Pred","Obs")

set.seed(my_seed)
rf_calibrate_model <- glm(Obs~Pred, data = rf_pred_train, family = binomial) 

#Pre-COVID
uncali_pred<- data.frame(Pred = rf_predictions$.pred_TRUE)

cali_pred <- predict(rf_calibrate_model, uncali_pred, type = "response")

rf_predictions<-rf_predictions %>% 
  bind_cols(cali_pred_TRUE = cali_pred) 

#Peri-COVID
uncali_pred<- data.frame(Pred = rf_predictions_periCOVID$.pred_TRUE)

cali_pred <- predict(rf_calibrate_model, uncali_pred, type = "response")

rf_predictions_periCOVID<-rf_predictions_periCOVID %>% 
  bind_cols(cali_pred_TRUE = cali_pred) 

formula = as.formula(paste(outcome, "cali_pred_TRUE", sep = "~"))
c <- calibration(formula, 
                 data= rf_predictions_periCOVID, 
                 class = "TRUE",
                 cuts = 30)$data

rf_calibraion_plot<-ggplot() + 
  geom_line(data = c, aes(midpoint, Percent, linetype = model)) +
  scale_linetype_manual(values=c("solid"))+
  geom_point(data = c, aes(midpoint, Percent, shape = model), size = 2) +
  geom_line(aes(c(0, 100), c(0, 100)), linetype = 2, 
            color = 'grey50')+
  theme(axis.text.y   = element_text(size=12),
        axis.text.x   = element_text(size=12),
        axis.title.y  = element_text(size=12),
        axis.title.x  = element_text(size=12),
        panel.background = element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        legend.position="none")+
  xlab("Predicted probability")+
  ylab("Observed event percent")

#Extreme gradient boosting trees
xgb_pred_train<-my_prediction(xgb_finalworkflow, df_train, outcome) %>% 
  dplyr::select(-.pred_class,-.pred_FALSE) %>% 
  as.data.frame()

colnames(pred_train)<-c("Pred","Obs")

set.seed(my_seed)
xgb_calibrate_model <- glm(Obs~Pred, data = xgb_pred_train, family = binomial) 

#Pre-COVID
uncali_pred<- data.frame(Pred = xgb_predictions$.pred_TRUE)

cali_pred <- predict(xgb_calibrate_model, uncali_pred, type = "response")

xgb_predictions <- xgb_predictions %>% 
  bind_cols(cali_pred_TRUE = cali_pred) 

#Peri-COVID
uncali_pred<- data.frame(Pred = xgb_predictions_periCOVID$.pred_TRUE)

cali_pred <- predict(xgb_calibrate_model, uncali_pred, type = "response")

xgb_predictions_periCOVID<-xgb_predictions_periCOVID %>% 
  bind_cols(cali_pred_TRUE = cali_pred) 

formula = as.formula(paste(outcome, "cali_pred_TRUE", sep = "~"))
c <- calibration(formula, 
                 data= xgb_predictions_periCOVID, 
                 class = "TRUE",
                 cuts = 30)$data

xgb_calibraion_plot<-ggplot() + 
  geom_line(data = c, aes(midpoint, Percent, linetype = model)) +
  scale_linetype_manual(values=c("solid"))+
  geom_point(data = c, aes(midpoint, Percent, shape = model), size = 2) +
  geom_line(aes(c(0, 100), c(0, 100)), linetype = 2, 
            color = 'grey50')+
  theme(axis.text.y   = element_text(size=12),
        axis.text.x   = element_text(size=12),
        axis.title.y  = element_text(size=12),
        axis.title.x  = element_text(size=12),
        panel.background = element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        legend.position="none")+
  xlab("Predicted probability")+
  ylab("Observed event percent")

#Single hidden layer neural network
shlnn_pred_train<-my_prediction(shlnn_finalworkflow, df_train, outcome) %>% 
  dplyr::select(-.pred_class,-.pred_FALSE) %>% 
  as.data.frame()

colnames(pred_train)<-c("Pred","Obs")

set.seed(my_seed)
shlnn_calibrate_model <- glm(Obs~Pred, data = shlnn_pred_train, family = binomial) 

#Pre-COVID
uncali_pred<- data.frame(Pred = shlnn_predictions$.pred_TRUE)

cali_pred <- predict(shlnn_calibrate_model, uncali_pred, type = "response")

shlnn_predictions<-shlnn_predictions %>% 
  bind_cols(cali_pred_TRUE = cali_pred) 

#Peri-COVID
uncali_pred<- data.frame(Pred = shlnn_predictions_periCOVID$.pred_TRUE)

cali_pred <- predict(shlnn_calibrate_model, uncali_pred, type = "response")

shlnn_predictions_periCOVID<-shlnn_predictions_periCOVID %>% 
  bind_cols(cali_pred_TRUE = cali_pred) 

formula = as.formula(paste(outcome, "cali_pred_TRUE", sep = "~"))
c <- calibration(formula, 
                 data= shlnn_predictions_periCOVID, 
                 class = "TRUE",
                 cuts = 30)$data

shlnn_calibraion_plot<-ggplot() + 
  geom_line(data = c, aes(midpoint, Percent, linetype = model)) +
  scale_linetype_manual(values=c("solid"))+
  geom_point(data = c, aes(midpoint, Percent, shape = model), size = 2) +
  geom_line(aes(c(0, 100), c(0, 100)), linetype = 2, 
            color = 'grey50')+
  theme(axis.text.y   = element_text(size=12),
        axis.text.x   = element_text(size=12),
        axis.title.y  = element_text(size=12),
        axis.title.x  = element_text(size=12),
        panel.background = element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        legend.position="none")+
  xlab("Predicted probability")+
  ylab("Observed event percent")

#Support vector machine
svm_pred_train<-my_prediction(svm_finalworkflow, df_train, outcome) %>% 
  dplyr::select(-.pred_class,-.pred_FALSE) %>% 
  as.data.frame()

colnames(pred_train)<-c("Pred","Obs")

set.seed(my_seed)
svm_calibrate_model <- glm(Obs~Pred, data = svm_pred_train, family = binomial) 

#Pre-COVID
uncali_pred<- data.frame(Pred = svm_predictions$.pred_TRUE)

cali_pred <- predict(svm_calibrate_model, uncali_pred, type = "response")

svm_predictions<-svm_predictions %>% 
  bind_cols(cali_pred_TRUE = cali_pred) 

#Peri-COVID
uncali_pred<- data.frame(Pred = svm_predictions_periCOVID$.pred_TRUE)

cali_pred <- predict(svm_calibrate_model, uncali_pred, type = "response")

svm_predictions_periCOVID<-svm_predictions_periCOVID %>% 
  bind_cols(cali_pred_TRUE = cali_pred) 

formula = as.formula(paste(outcome, "cali_pred_TRUE", sep = "~"))
c <- calibration(formula, 
                 data= svm_predictions_periCOVID, 
                 class = "TRUE",
                 cuts = 30)$data

svm_calibraion_plot<-ggplot() + 
  geom_line(data = c, aes(midpoint, Percent, linetype = model)) +
  scale_linetype_manual(values=c("solid"))+
  geom_point(data = c, aes(midpoint, Percent, shape = model), size = 2) +
  geom_line(aes(c(0, 100), c(0, 100)), linetype = 2, 
            color = 'grey50')+
  theme(axis.text.y   = element_text(size=12),
        axis.text.x   = element_text(size=12),
        axis.title.y  = element_text(size=12),
        axis.title.x  = element_text(size=12),
        panel.background = element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        legend.position="none")+
  xlab("Predicted probability")+
  ylab("Observed event percent")

#k nearest neighbors
knn_pred_train<-my_prediction(knn_finalworkflow, df_train, outcome) %>% 
  dplyr::select(-.pred_class,-.pred_FALSE) %>% 
  as.data.frame()

colnames(pred_train)<-c("Pred","Obs")

set.seed(my_seed)
knn_calibrate_model <- glm(Obs~Pred, data = knn_pred_train, family = binomial) 

#Pre-COVID
uncali_pred<- data.frame(Pred = knn_predictions$.pred_TRUE)

cali_pred <- predict(knn_calibrate_model, uncali_pred, type = "response")

knn_predictions<-knn_predictions %>% 
  bind_cols(cali_pred_TRUE = cali_pred) 

#Peri-COVID
uncali_pred<- data.frame(Pred = knn_predictions_periCOVID$.pred_TRUE)

cali_pred <- predict(knn_calibrate_model, uncali_pred, type = "response")

knn_predictions_periCOVID<-knn_predictions_periCOVID %>% 
  bind_cols(cali_pred_TRUE = cali_pred) 

formula = as.formula(paste(outcome, "cali_pred_TRUE", sep = "~"))
c <- calibration(formula, 
                 data= knn_predictions_periCOVID, 
                 class = "TRUE",
                 cuts = 30)$data

knn_calibraion_plot<-ggplot() + 
  geom_line(data = c, aes(midpoint, Percent, linetype = model)) +
  scale_linetype_manual(values=c("solid"))+
  geom_point(data = c, aes(midpoint, Percent, shape = model), size = 2) +
  geom_line(aes(c(0, 100), c(0, 100)), linetype = 2, 
            color = 'grey50')+
  theme(axis.text.y   = element_text(size=12),
        axis.text.x   = element_text(size=12),
        axis.title.y  = element_text(size=12),
        axis.title.x  = element_text(size=12),
        panel.background = element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        legend.position="none")+
  xlab("Predicted probability")+
  ylab("Observed event percent")


#Multivariate adaptive regression spline
mars_pred_train<-my_prediction(mars_finalworkflow, df_train, outcome) %>% 
  dplyr::select(-.pred_class,-.pred_FALSE) %>% 
  as.data.frame()

colnames(pred_train)<-c("Pred","Obs")

set.seed(my_seed)
mars_calibrate_model <- glm(Obs~Pred, data = mars_pred_train, family = binomial) 

#Pre-COVID
uncali_pred<- data.frame(Pred = mars_predictions$.pred_TRUE)

cali_pred <- predict(mars_calibrate_model, uncali_pred, type = "response")

mars_predictions<-mars_predictions %>% 
  bind_cols(cali_pred_TRUE = cali_pred) 

#Peri-COVID
uncali_pred<- data.frame(Pred = mars_predictions_periCOVID$.pred_TRUE)

cali_pred <- predict(mars_calibrate_model, uncali_pred, type = "response")

mars_predictions_periCOVID<-mars_predictions_periCOVID %>% 
  bind_cols(cali_pred_TRUE = cali_pred) 

formula = as.formula(paste(outcome, "cali_pred_TRUE", sep = "~"))
c <- calibration(formula, 
                 data= mars_predictions_periCOVID, 
                 class = "TRUE",
                 cuts = 30)$data

mars_calibraion_plot<-ggplot() + 
  geom_line(data = c, aes(midpoint, Percent, linetype = model)) +
  scale_linetype_manual(values=c("solid"))+
  geom_point(data = c, aes(midpoint, Percent, shape = model), size = 2) +
  geom_line(aes(c(0, 100), c(0, 100)), linetype = 2, 
            color = 'grey50')+
  theme(axis.text.y   = element_text(size=12),
        axis.text.x   = element_text(size=12),
        axis.title.y  = element_text(size=12),
        axis.title.x  = element_text(size=12),
        panel.background = element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        legend.position="none")+
  xlab("Predicted probability")+
  ylab("Observed event percent")

#Decision tree
dt_pred_train<-my_prediction(dt_finalworkflow, df_train, outcome) %>% 
  dplyr::select(-.pred_class,-.pred_FALSE) %>% 
  as.data.frame()

colnames(pred_train)<-c("Pred","Obs")

set.seed(my_seed)
dt_calibrate_model <- glm(Obs~Pred, data = dt_pred_train, family = binomial) 

#Pre-COVID
uncali_pred<- data.frame(Pred = dt_predictions$.pred_TRUE)

cali_pred <- predict(dt_calibrate_model, uncali_pred, type = "response")

dt_predictions<-dt_predictions %>% 
  bind_cols(cali_pred_TRUE = cali_pred) 

#Peri-COVID
uncali_pred<- data.frame(Pred = dt_predictions_periCOVID$.pred_TRUE)

cali_pred <- predict(dt_calibrate_model, uncali_pred, type = "response")

dt_predictions_periCOVID<-dt_predictions_periCOVID %>% 
  bind_cols(cali_pred_TRUE = cali_pred) 

formula = as.formula(paste(outcome, "cali_pred_TRUE", sep = "~"))
c <- calibration(formula, 
                 data= dt_predictions_periCOVID, 
                 class = "TRUE",
                 cuts = 30)$data

dt_calibraion_plot<-ggplot() + 
  geom_line(data = c, aes(midpoint, Percent, linetype = model)) +
  scale_linetype_manual(values=c("solid"))+
  geom_point(data = c, aes(midpoint, Percent, shape = model), size = 2) +
  geom_line(aes(c(0, 100), c(0, 100)), linetype = 2, 
            color = 'grey50')+
  theme(axis.text.y   = element_text(size=12),
        axis.text.x   = element_text(size=12),
        axis.title.y  = element_text(size=12),
        axis.title.x  = element_text(size=12),
        panel.background = element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        legend.position="none")+
  xlab("Predicted probability")+
  ylab("Observed event percent")

#### Risk threshold determination####
metrics = c("sensitivity", "specificity")
thres = seq(from = 0.1, to = 1 , by = 0.001)
positive = "TRUE"
negative = "FALSE"

#logistic regression 
lr_thres<-metricOverCutoff(lr_predictions[,"cali_pred_TRUE"], 
                           lr_predictions[,outcome], 
                           positive = positive, 
                           negative = negative, 
                           metrics =metrics, 
                           threshold = thres)

#ECOG logistic regression 
ecog_thres<-metricOverCutoff(ecog_predictions[,"cali_pred_TRUE"], 
                             ecog_predictions[,outcome], 
                             positive = positive, 
                             negative = negative, 
                             metrics =metrics, 
                             threshold = thres)


#Logistic regression with elastic net penalty 
lrenp_thres<-metricOverCutoff(lrenp_predictions[,"cali_pred_TRUE"], 
                              lrenp_predictions[,outcome], 
                              positive = positive, 
                              negative = negative, 
                              metrics =metrics, 
                              threshold = thres)

#Random forest
rf_thres<-metricOverCutoff(rf_predictions[,"cali_pred_TRUE"], 
                           rf_predictions[,outcome], 
                           positive = positive, 
                           negative = negative, 
                           metrics =metrics, 
                           threshold = thres)

#Extreme gradient boosting trees
xgb_thres<-metricOverCutoff(xgb_predictions[,"cali_pred_TRUE"], 
                            xgb_predictions[,outcome], 
                            positive = positive, 
                            negative = negative, 
                            metrics =metrics, 
                            threshold = thres)

#Single hidden layer neural network
shlnn_thres<-metricOverCutoff(shlnn_predictions[,"cali_pred_TRUE"], 
                              shlnn_predictions[,outcome], 
                              positive = positive, 
                              negative = negative, 
                              metrics =metrics, 
                              threshold = thres)

#Support vector machine
svm_thres<-metricOverCutoff(svm_predictions[,"cali_pred_TRUE"], 
                              svm_predictions[,outcome], 
                              positive = positive, 
                              negative = negative, 
                              metrics =metrics, 
                              threshold = thres)

#k nearest neighbors
knn_thres<-metricOverCutoff(knn_predictions[,"cali_pred_TRUE"], 
                              knn_predictions[,outcome], 
                              positive = positive, 
                              negative = negative, 
                              metrics =metrics, 
                              threshold = thres)

#Multivariate adaptive regression spline
mars_thres<-metricOverCutoff(mars_predictions[,"cali_pred_TRUE"], 
                              mars_predictions[,outcome], 
                              positive = positive, 
                              negative = negative, 
                              metrics =metrics, 
                              threshold = thres)

#Decision tree
dt_thres<-metricOverCutoff(dt_predictions[,"cali_pred_TRUE"], 
                              dt_predictions[,outcome], 
                              positive = positive, 
                              negative = negative, 
                              metrics =metrics, 
                              threshold = thres)
