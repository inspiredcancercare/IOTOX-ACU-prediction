	Online supplementary materials and code repository
#Machine learning (ML) technique overview 

In this section, we briefly introduce each ML algorithm, over-sampling, and model interpretation technique that are using in the study.   

Machine learning algorithm overview 

Logistic regression with regularization (regularized LR) 

Logistic regression (LR) provides a powerful yet simple approach to modeling which is suitable when certain assumptions are met (e.g., no multicollinearity among the independent variables). However, LR models tend to overfit the training data as the data volume increase and the assumptions are more likely to be violated. Regularization methods provide a way to mitigate the overfitting issue via penalizing or shrinking both the number of predictors in the model and their estimated coefficients.1 Thus, the regularization also effectively maintains the simplicity of the models, making it suitable for complex data with high dimensionality and huge volume. There are three forms of regularization, ridge penalty, least absolute selection and shrinkage operator (LASSO) penalty, and Elastic nets (a combination of ridge and LASSO penalty). The regularized LR have demonstrated excellent performance in complex medical prediction, such as predicting subjective financial burden associated with breast cancer treatment.2 

 

Multivariate adaptive regression spline (MARS) 

Although the regularized LR is powerful, it cannot account for potential interaction effects between independent variables, which may decrease the model performance. Multivariate adaptive regression spline is an ensemble of linear/logistic models which are able to take any nonlinear relation among predictors into account to achieve higher performance. Specifically, the algorithm automatically partitions predictors and models a linear model for each predictor group. The connections between the regression lines are known as knots and which each has a pair of basis functions. The algorithm iteratively evaluates the contribution of each basis function to determine the final model. In this way, MARS is robust to outliers, works well with high dimensional data, and suitable for complex nonlinear regression problems.3  

 

K-nearest neighbors (KNN) 

The KNN is a memory-based algorithm that makes predictions based on the shared characteristics between data points. The KNN algorithm makes classifications using the most common class of k observations that are similar to the new data being predicted.4 Unlike other algorithms, KNN models cannot be summarized into a closed-form mathematical expression and require all training data to be available when making predictions with new data. Therefore, KNN models are computationally expensive. Despite this caveat, KNN has been successfully used to solve many complex classification problems and is useful for data preprocessing purposes, e.g., KNN imputation of missing values.4 We included the KNN algorithm in this study because of its unique way of predicting and potential capability to identify extra high-risk individuals that other algorithms failed to detect.  

 

Support vector machines (SVM) 

Support vector machines make classification via finding an optimal boundary called hyperplane that best distinguishes two classes. In the case of binary classification, the SVM algorithm uses a method, such as the kernel function, to project data into a high-dimensional feature space and finds a linear boundary that the distances between the boundary and the closest data points from either class are the largest. Finally, the algorithm projects the boundary back to the original feature space resulting a final decision boundary which is nonlinear for most cases. The SVM has been widely used in many areas, including medicine.1 Its promising performance demonstrated in previous research1,5,6 intrigues us to examine its performance in predicting complications for patients undergoing head and neck free tissue reconstruction.  

 

Decision tree (DT) 

The DT algorithm is the simplest tree-based nonparametric algorithm. The DT algorithm works by partitioning data into a number of smaller groups with similar response values of certain predictors using a set of splitting rules.4 The greatest advantages of the DT models are: (1) the interpretation of the models is intuitive and easily accepted by most healthcare professionals; (2) the models can be easily implemented into practice because of their rule-based nature.7 Despite the benefit, the predictive performance of DT models may not be optimal when compared to other complex algorithms.4 Nevertheless, the algorithm is the foundation of other powerful ensemble algorithms, including random forest and extreme gradient boosting tree, and still included in the current study as a baseline of the ensemble algorithms. 

 

Random forest (RF) 

An RF model comprises many de-correlated decision trees that jointly determine the final classification. In short, the algorithm builds many trees on randomly bootstrapped copies of the training data and then uses the aggregated predictions from all the trees as the final output.8 Through the aggregation process, the predictive performance of RF models is significantly improved as compared to the DT models, while the interpretability of the model is sacrificed. Many RF-based applications have been developed and demonstrated great values in supporting clinical decision makings.9 

 

Gradient boosting machine (GBM) 

Gradient boosting is a complex yet powerful and popular ensemble algorithm. The way it works is similar to the RF. Whereas the RF algorithm tries to build and combine many well-trained decision trees, the GBM develops an ensemble of shallow trees. The algorithm starts with building a weak shallow tree and sequentially train many trees with each tree learning and improving by addressing the biggest mistake that its previous tree makes. Theoretically, the principles of the gradient boosting framework can be applied to stack any type of model. Applying it to decision trees is most effective in general.4 Thus, we implemented a decision tree-based GBM in this study. The GBM use in medicine includes cancer patient mortality, immune checkpoint inhibitor therapy toxicity, and deterioration of patients with the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) prediction as well as others.10–13  In this study, we implemented the GBM to model our data using Extreme Gradient Boosting (XGBoost) library in R. 

 

Single-hidden-layer neural network (SHLNN) 

Artificial neural networks that solve problems via loosely simulating the way the human brain works have received substantial attention in recent decades. The SHLNN is the simplest form of artificial neural networks  and has shown promising performance in medical classification problems, such as predicting mortality for patients undergoing liver transplantation.14 The algorithm consists of three basic layers, input, hidden, and output layers. The input layer contains several nodes (neurons) representing original predictors. The algorithm uses the information from the input layer to form a hidden layer with any number of nods assigned with a weight. Next, the algorithm transforms the input information when they pass through the hidden layer’s nodes to the output layer using the equation shown in Eq. 1. The number of hidden layers node and corresponding weight was determined by an iterative learning process.1    

 

y=activation(Σ(weight×input)+bias)                                                                                          (1)
y=activationΣweight×input+bias                                                                                          (1)
 
 

 

Voting ensemble (VOTE) 

The last algorithm explored in this study is an ensemble of the abovementioned algorithms. Previous studies have demonstrated the potential benefit of grouping multiple different algorithms through a process known as ensemble learning to create a new algorithm that may have a more robust predictive performance. 15 There are many ensemble learning techniques. An un-weighted voting algorithm was used to group our algorithms because of its simplicity and effectiveness in grouping algorithms for a better model.1 Our VOTE algorithm used a majority rule (> 4) to determine whether a patient should be classified as at high risk of post-surgical complications. For instance, if a patient is deemed to have an increased risk of total flap loss by five or more base algorithms, the patient will be considered high risk by the ensemble algorithm. 

 

SMOTE over-sampling technique 

As discussed in our main manuscript, our samples were imbalanced, with most samples without the outcomes of interest. Given the proven negative effects of imbalanced samples on machine learning16, an over-sampling technique known as Synthetic Minority Over-sampling Technique (SMOTE) was used to balance our data. The SMOTE is a widely discussed and most used algorithm that effectively balances data by adding new data points related to the outcomes.17 Specifically, the SMOTE used a KNN algorithm to create new data points using the information from existing minority class, e.g., cases with total flap loss. In this study, the SMOTE over-sampling was implemented using the R’ recipes’ and ‘themis’ packages on the training sets of each isolated resample during the cross-validation process.18,19 The hyperparameter of the KNN was set to 5 for training all models in this study, meaning that the algorithm created a minority case for each existing minority class using 5 nearest neighbors of the minority class. 

 

Model interpretation techniques 

The interpretability of ML models is one of the top issues being critiqued and barriers to ML implementation in practices, especially health practices. Several explanation techniques have been developed to ‘unbox’ ML models and explain how they use the data to make predictions using a human-friendly way. Two explanation techniques were used to globally explain our models. The first one is permutation feature importance (PFI) which is the most used method to identify important predictors for an ML model. As mentioned in our main paper, the PFI measures the importance of a predictor by measuring the prediction error change when randomly altering the predictor’s value. A predicter is “important” if the prediction error increases when shuffling the predictor’s value.  

The PFI provides a compressed insight into model’s behavior. However, the PFI is not able to measure the association between predictors’ values and model outputs. For instance, the PFI may determine that smoking status is an important factor for regularized LR to predict total flap loss. It does not show whether a current or former smoker is more likely to get a positive prediction. Therefore, a second explanation method, accumulated local effects (ALE), was used in this study.  

The ALE is an expansion of partial dependence plots (PDP) which estimate the relationships between model output and predictors of interest.20 The ALE uses conditional distribution to avoid counting data points that are not possible in the real world, making it a robust method for estimating predictors’ effects in a highly inter-correlated dataset. For categorical predictors, the ALE estimates the average predicted positive probabilities for each category of the predicters. For numerical predictors, the ALE demonstrates relationships, e.g., linear, monotonic, or more complex, between values of the predictors and predicted positive probabilities on average. In this study, we examined the ALE for predictors that were deemed important for four or more models by the PFI. We averaged the ALE from different ML models for each important predictor to estimate each predictor’s final ALE. All model interpretation methods were performed using the R ‘DALEXtra’ package.21  

 
