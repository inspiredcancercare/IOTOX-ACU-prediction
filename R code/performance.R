####Reexamine confusion matrix for each algorithm using new risk threshold####
#Logistic regression
thres<- lr_thres$threshold

lr_predictions$new_class<- ifelse(lr_predictions$cali_pred_TRUE>=thres, 
                                  "TRUE", "FALSE")

obs_class<- lr_predictions %>% pull(outcome)

cali_pred_class <- as.factor(lr_predictions$new_class)

lr_cali_cm <- confusionMatrix(cali_pred_class,
                              obs_class, 
                              positive = "TRUE")

lr_cali_auc <- pROC::roc(as.vector(obs_class),
                         as.matrix(lr_predictions$cali_pred_TRUE))


lr_predictions_periCOVID <- my_prediction(lr_workflow, df_periCOVID , outcome)

lr_predictions_periCOVID$new_class<- ifelse(lr_predictions_periCOVID$cali_pred_TRUE>=thres, 
                                            "TRUE", "FALSE")

obs_class_periCOVID<- lr_predictions_periCOVID %>% pull(outcome)

cali_pred_class <- as.factor(lr_predictions_periCOVID$new_class)

lr_cali_cm_periCOVID <- confusionMatrix(cali_pred_class,
                                        obs_class_periCOVID, 
                                        positive = "TRUE")

lr_cali_auc_periCOVID <- pROC::roc(as.vector(obs_class_periCOVID),
                                   as.matrix(lr_predictions_periCOVID$cali_pred_TRUE))

#ECOG logistic regression
thres<- ecog_thres$threshold

ecog_predictions$new_class<- ifelse(ecog_predictions$cali_pred_TRUE>=thres, 
                                    "TRUE", "FALSE")

obs_class<- ecog_predictions %>% pull(outcome)

cali_pred_class <- as.factor(ecog_predictions$new_class)

ecog_cali_cm <- confusionMatrix(cali_pred_class,
                                obs_class, 
                                positive = "TRUE")

ecog_cali_auc <- pROC::roc(as.vector(obs_class),
                           as.matrix(ecog_predictions$cali_pred_TRUE))


ecog_predictions_periCOVID <- my_prediction(ecog_workflow, df_periCOVID , outcome)

ecog_predictions_periCOVID$new_class<- ifelse(ecog_predictions_periCOVID$cali_pred_TRUE>=thres, 
                                              "TRUE", "FALSE")

obs_class_periCOVID<- ecog_predictions_periCOVID %>% pull(outcome)

cali_pred_class <- as.factor(ecog_predictions_periCOVID$new_class)

ecog_cali_cm_periCOVID <- confusionMatrix(cali_pred_class,
                                          obs_class_periCOVID, 
                                          positive = "TRUE")

ecog_cali_auc_periCOVID <- pROC::roc(as.vector(obs_class_periCOVID),
                                     as.matrix(ecog_predictions_periCOVID$cali_pred_TRUE))
#Logistic regression with elastic net penalty
thres<- lrenp_thres$threshold

lrenp_predictions$new_class<- ifelse(lrenp_predictions$cali_pred_TRUE>=thres, 
                                     "TRUE", "FALSE")

obs_class<- lrenp_predictions %>% pull(outcome)

cali_pred_class <- as.factor(lrenp_predictions$new_class)

lrenp_cali_cm <- confusionMatrix(cali_pred_class,
                                 obs_class, 
                                 positive = "TRUE")

lrenp_cali_auc <- pROC::roc(as.vector(obs_class),
                            as.matrix(lrenp_predictions$cali_pred_TRUE))


lrenp_predictions_periCOVID <- my_prediction(lrenp_workflow, df_periCOVID , outcome)

lrenp_predictions_periCOVID$new_class<- ifelse(lrenp_predictions_periCOVID$cali_pred_TRUE>=thres, 
                                               "TRUE", "FALSE")

obs_class_periCOVID <- lrenp_predictions_periCOVID %>% pull(outcome)

cali_pred_class <- as.factor(lrenp_predictions_periCOVID$new_class)

lrenp_cali_cm_periCOVID <- confusionMatrix(cali_pred_class,
                                           obs_class_periCOVID, 
                                           positive = "TRUE")

lrenp_cali_auc_periCOVID <- pROC::roc(as.vector(obs_class_periCOVID),
                                      as.matrix(lrenp_predictions_periCOVID$cali_pred_TRUE))
#Random forest
thres<- rf_thres$threshold

rf_predictions$new_class<- ifelse(rf_predictions$cali_pred_TRUE>=thres, 
                                  "TRUE", "FALSE")

obs_class<- rf_predictions %>% pull(outcome)

cali_pred_class <- as.factor(rf_predictions$new_class)

rf_cali_cm <- confusionMatrix(cali_pred_class,
                              obs_class, 
                              positive = "TRUE")

rf_cali_auc <- pROC::roc(as.vector(obs_class),
                         as.matrix(rf_predictions$cali_pred_TRUE))


rf_predictions_periCOVID <- my_prediction(rf_workflow, df_periCOVID , outcome)

rf_predictions_periCOVID$new_class<- ifelse(rf_predictions_periCOVID$cali_pred_TRUE>=thres, 
                                            "TRUE", "FALSE")

obs_class_periCOVID<- rf_predictions_periCOVID %>% pull(outcome)

cali_pred_class <- as.factor(rf_predictions_periCOVID$new_class)

rf_cali_cm_periCOVID <- confusionMatrix(cali_pred_class,
                                        obs_class_periCOVID, 
                                        positive = "TRUE")

rf_cali_auc_periCOVID <- pROC::roc(as.vector(obs_class_periCOVID),
                                   as.matrix(rf_predictions_periCOVID$cali_pred_TRUE))
#Extreme gradient boosting trees
thres<- xgb_thres$threshold

xgb_predictions$new_class<- ifelse(xgb_predictions$cali_pred_TRUE>=thres, 
                                   "TRUE", "FALSE")

obs_class<- xgb_predictions %>% pull(outcome)

cali_pred_class <- as.factor(xgb_predictions$new_class)

xgb_cali_cm <- confusionMatrix(cali_pred_class,
                               obs_class, 
                               positive = "TRUE")

xgb_cali_auc <- pROC::roc(as.vector(obs_class),
                          as.matrix(xgb_predictions$cali_pred_TRUE))


xgb_predictions_periCOVID <- my_prediction(xgb_workflow, df_periCOVID , outcome)

xgb_predictions_periCOVID$new_class<- ifelse(xgb_predictions_periCOVID$cali_pred_TRUE>=thres, 
                                             "TRUE", "FALSE")

obs_class_periCOVID <- xgb_predictions_periCOVID %>% pull(outcome)

cali_pred_class <- as.factor(xgb_predictions_periCOVID$new_class)

xgb_cali_cm_periCOVID <- confusionMatrix(cali_pred_class,
                                         obs_class_periCOVID, 
                                         positive = "TRUE")

xgb_cali_auc_periCOVID <- pROC::roc(as.vector(obs_class_periCOVID),
                                    as.matrix(xgb_predictions_periCOVID$cali_pred_TRUE))

#Single hidden layer neural network
thres<- dt_thres$threshold

dt_predictions$new_class<- ifelse(dt_predictions$cali_pred_TRUE>=thres, 
                                     "TRUE", "FALSE")

obs_class<- dt_predictions %>% pull(outcome)

cali_pred_class <- as.factor(dt_predictions$new_class)

dt_cali_cm <- confusionMatrix(cali_pred_class,
                                 obs_class, 
                                 positive = "TRUE")

dt_cali_auc <- pROC::roc(as.vector(obs_class),
                            as.matrix(dt_predictions$cali_pred_TRUE))


dt_predictions_periCOVID <- my_prediction(dt_workflow, df_periCOVID , outcome)

dt_predictions_periCOVID$new_class<- ifelse(dt_predictions_periCOVID$cali_pred_TRUE>=thres, 
                                               "TRUE", "FALSE")

obs_class_periCOVID <- dt_predictions_periCOVID %>% pull(outcome)

cali_pred_class <- as.factor(dt_predictions_periCOVID$new_class)

dt_cali_cm_periCOVID <- confusionMatrix(cali_pred_class,
                                           obs_class_periCOVID, 
                                           positive = "TRUE")

dt_cali_auc_periCOVID <- pROC::roc(as.vector(obs_class_periCOVID),
                                      as.matrix(dt_predictions_periCOVID$cali_pred_TRUE))

#Support vector machine
thres<- svm_thres$threshold

svm_predictions$new_class<- ifelse(svm_predictions$cali_pred_TRUE>=thres, 
                                     "TRUE", "FALSE")

obs_class<- svm_predictions %>% pull(outcome)

cali_pred_class <- as.factor(svm_predictions$new_class)

svm_cali_cm <- confusionMatrix(cali_pred_class,
                                 obs_class, 
                                 positive = "TRUE")

svm_cali_auc <- pROC::roc(as.vector(obs_class),
                            as.matrix(svm_predictions$cali_pred_TRUE))


svm_predictions_periCOVID <- my_prediction(svm_workflow, df_periCOVID , outcome)

svm_predictions_periCOVID$new_class<- ifelse(svm_predictions_periCOVID$cali_pred_TRUE>=thres, 
                                               "TRUE", "FALSE")

obs_class_periCOVID <- svm_predictions_periCOVID %>% pull(outcome)

cali_pred_class <- as.factor(svm_predictions_periCOVID$new_class)

svm_cali_cm_periCOVID <- confusionMatrix(cali_pred_class,
                                           obs_class_periCOVID, 
                                           positive = "TRUE")

svm_cali_auc_periCOVID <- pROC::roc(as.vector(obs_class_periCOVID),
                                      as.matrix(svm_predictions_periCOVID$cali_pred_TRUE))


#k nearest neighbors
thres<- knn_thres$threshold

knn_predictions$new_class<- ifelse(knn_predictions$cali_pred_TRUE>=thres, 
                                     "TRUE", "FALSE")

obs_class<- knn_predictions %>% pull(outcome)

cali_pred_class <- as.factor(knn_predictions$new_class)

knn_cali_cm <- confusionMatrix(cali_pred_class,
                                 obs_class, 
                                 positive = "TRUE")

knn_cali_auc <- pROC::roc(as.vector(obs_class),
                            as.matrix(knn_predictions$cali_pred_TRUE))


knn_predictions_periCOVID <- my_prediction(knn_workflow, df_periCOVID , outcome)

knn_predictions_periCOVID$new_class<- ifelse(knn_predictions_periCOVID$cali_pred_TRUE>=thres, 
                                               "TRUE", "FALSE")

obs_class_periCOVID <- knn_predictions_periCOVID %>% pull(outcome)

cali_pred_class <- as.factor(knn_predictions_periCOVID$new_class)

knn_cali_cm_periCOVID <- confusionMatrix(cali_pred_class,
                                           obs_class_periCOVID, 
                                           positive = "TRUE")

knn_cali_auc_periCOVID <- pROC::roc(as.vector(obs_class_periCOVID),
                                      as.matrix(knn_predictions_periCOVID$cali_pred_TRUE))


#Multivariate adaptive regression spline
thres<- mars_thres$threshold

mars_predictions$new_class<- ifelse(mars_predictions$cali_pred_TRUE>=thres, 
                                     "TRUE", "FALSE")

obs_class<- mars_predictions %>% pull(outcome)

cali_pred_class <- as.factor(mars_predictions$new_class)

mars_cali_cm <- confusionMatrix(cali_pred_class,
                                 obs_class, 
                                 positive = "TRUE")

mars_cali_auc <- pROC::roc(as.vector(obs_class),
                            as.matrix(mars_predictions$cali_pred_TRUE))


mars_predictions_periCOVID <- my_prediction(mars_workflow, df_periCOVID , outcome)

mars_predictions_periCOVID$new_class<- ifelse(mars_predictions_periCOVID$cali_pred_TRUE>=thres, 
                                               "TRUE", "FALSE")

obs_class_periCOVID <- mars_predictions_periCOVID %>% pull(outcome)

cali_pred_class <- as.factor(mars_predictions_periCOVID$new_class)

mars_cali_cm_periCOVID <- confusionMatrix(cali_pred_class,
                                           obs_class_periCOVID, 
                                           positive = "TRUE")

mars_cali_auc_periCOVID <- pROC::roc(as.vector(obs_class_periCOVID),
                                      as.matrix(mars_predictions_periCOVID$cali_pred_TRUE))


#Decision tree
thres<- dt_thres$threshold

dt_predictions$new_class<- ifelse(dt_predictions$cali_pred_TRUE>=thres, 
                                     "TRUE", "FALSE")

obs_class<- dt_predictions %>% pull(outcome)

cali_pred_class <- as.factor(dt_predictions$new_class)

dt_cali_cm <- confusionMatrix(cali_pred_class,
                                 obs_class, 
                                 positive = "TRUE")

dt_cali_auc <- pROC::roc(as.vector(obs_class),
                            as.matrix(dt_predictions$cali_pred_TRUE))


dt_predictions_periCOVID <- my_prediction(dt_workflow, df_periCOVID , outcome)

dt_predictions_periCOVID$new_class<- ifelse(dt_predictions_periCOVID$cali_pred_TRUE>=thres, 
                                               "TRUE", "FALSE")

obs_class_periCOVID <- dt_predictions_periCOVID %>% pull(outcome)

cali_pred_class <- as.factor(dt_predictions_periCOVID$new_class)

dt_cali_cm_periCOVID <- confusionMatrix(cali_pred_class,
                                           obs_class_periCOVID, 
                                           positive = "TRUE")

dt_cali_auc_periCOVID <- pROC::roc(as.vector(obs_class_periCOVID),
                                      as.matrix(dt_predictions_periCOVID$cali_pred_TRUE))


####McNamer's test for model comparison####
#ML vs logistic regression

#lrenp vs lr
roc.test(obs_class, data.frame(lrenp_predictions$cali_pred_TRUE,
                               lr_predictions$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)


roc.test(obs_class_periCOVID, data.frame(lrenp_predictions_periCOVID$cali_pred_TRUE,
                                         lr_predictions_periCOVID$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

#rf vs lr
roc.test(obs_class, data.frame(rf_predictions$cali_pred_TRUE,
                               lr_predictions$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

roc.test(obs_class_periCOVID, data.frame(rf_predictions_periCOVID$cali_pred_TRUE,
                                         lr_predictions_periCOVID$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

#xgb vs lr
roc.test(obs_class, data.frame(xgb_predictions$cali_pred_TRUE,
                               lr_predictions$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

roc.test(obs_class_periCOVID, data.frame(xgb_predictions_periCOVID$cali_pred_TRUE,
                                         lr_predictions_periCOVID$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided",
         boot.stratified=TRUE)

#shlnn vs lr
roc.test(obs_class, data.frame(shlnn_predictions$cali_pred_TRUE,
                               lr_predictions$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

roc.test(obs_class_periCOVID, data.frame(shlnn_predictions_periCOVID$cali_pred_TRUE,
                                         lr_predictions_periCOVID$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

#svm vs lr
roc.test(obs_class, data.frame(svm_predictions$cali_pred_TRUE,
                               lr_predictions$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

roc.test(obs_class_periCOVID, data.frame(svm_predictions_periCOVID$cali_pred_TRUE,
                                         lr_predictions_periCOVID$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

#knn vs lr
roc.test(obs_class, data.frame(knn_predictions$cali_pred_TRUE,
                               lr_predictions$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

roc.test(obs_class_periCOVID, data.frame(knn_predictions_periCOVID$cali_pred_TRUE,
                                         lr_predictions_periCOVID$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

#mars vs lr
roc.test(obs_class, data.frame(mars_predictions$cali_pred_TRUE,
                               lr_predictions$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

roc.test(obs_class_periCOVID, data.frame(mars_predictions_periCOVID$cali_pred_TRUE,
                                         lr_predictions_periCOVID$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

#dt vs lr
roc.test(obs_class, data.frame(dt_predictions$cali_pred_TRUE,
                               lr_predictions$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

roc.test(obs_class_periCOVID, data.frame(dt_predictions_periCOVID$cali_pred_TRUE,
                                         lr_predictions_periCOVID$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)


#ML vs ECOG logistic regression
#lrenp vs ecog
roc.test(obs_class, data.frame(lrenp_predictions$cali_pred_TRUE,
                               ecog_predictions$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

roc.test(obs_class_periCOVID, data.frame(lrenp_predictions_periCOVID$cali_pred_TRUE,
                                         ecog_predictions_periCOVID$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

#rf vs ecog
roc.test(obs_class, data.frame(rf_predictions$cali_pred_TRUE,
                               ecog_predictions$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

roc.test(obs_class_periCOVID, data.frame(rf_predictions_periCOVID$cali_pred_TRUE,
                                         ecog_predictions_periCOVID$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

#xgb vs ecog
roc.test(obs_class, data.frame(xgb_predictions$cali_pred_TRUE,
                               ecog_predictions$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

roc.test(obs_class_periCOVID, data.frame(xgb_predictions_periCOVID$cali_pred_TRUE,
                                         ecog_predictions_periCOVID$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

#shlnn vs ecog
roc.test(obs_class, data.frame(shlnn_predictions$cali_pred_TRUE,
                               ecog_predictions$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

roc.test(obs_class_periCOVID, data.frame(shlnn_predictions_periCOVID$cali_pred_TRUE,
                                         ecog_predictions_periCOVID$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

#svm vs ecog
roc.test(obs_class, data.frame(svm_predictions$cali_pred_TRUE,
                               ecog_predictions$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

roc.test(obs_class_periCOVID, data.frame(svm_predictions_periCOVID$cali_pred_TRUE,
                                         ecog_predictions_periCOVID$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

#knn vs ecog
roc.test(obs_class, data.frame(knn_predictions$cali_pred_TRUE,
                               ecog_predictions$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

roc.test(obs_class_periCOVID, data.frame(knn_predictions_periCOVID$cali_pred_TRUE,
                                         ecog_predictions_periCOVID$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

#mars vs ecog
roc.test(obs_class, data.frame(mars_predictions$cali_pred_TRUE,
                               ecog_predictions$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

roc.test(obs_class_periCOVID, data.frame(mars_predictions_periCOVID$cali_pred_TRUE,
                                         ecog_predictions_periCOVID$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

#dt vs ecog
roc.test(obs_class, data.frame(dt_predictions$cali_pred_TRUE,
                               ecog_predictions$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)

roc.test(obs_class_periCOVID, data.frame(dt_predictions_periCOVID$cali_pred_TRUE,
                                         ecog_predictions_periCOVID$cali_pred_TRUE),
         method="delong", 
         alternative = "two.sided", 
         boot.stratified=TRUE)
