	Online supplementary materials and code repository
# Machine learning (ML) technique overview 

In this page, we briefly introduce each ML algorithm and model interpretation technique that are using in the study.   

Machine learning algorithm overview 
==========================================================
Logistic regression with regularization (regularized LR) 
--------------------------------------------------------
Logistic regression (LR) provides a powerful yet simple approach to modeling which is suitable when certain assumptions are met (e.g., no multicollinearity among the independent variables). However, LR models tend to overfit the training data as the data volume increase and the assumptions are more likely to be violated. Regularization methods provide a way to mitigate the overfitting issue via penalizing or shrinking both the number of predictors in the model and their estimated coefficients.<sup>1</sup> Thus, the regularization also effectively maintains the simplicity of the models, making it suitable for complex data with high dimensionality and huge volume. There are three forms of regularization, ridge penalty, least absolute selection and shrinkage operator (LASSO) penalty, and Elastic nets (a combination of ridge and LASSO penalty). The regularized LR have demonstrated excellent performance in complex medical prediction, such as predicting subjective financial burden associated with breast cancer treatment.<sup>2</sup> 


Random forest (RF) 
--------------------------------------------------------
An RF model comprises many de-correlated decision trees that jointly determine the final classification. In short, the algorithm builds many trees on randomly bootstrapped copies of the training data and then uses the aggregated predictions from all the trees as the final output.<sup>3</sup> Through the aggregation process, the predictive performance of RF models is significantly improved as compared to the DT models, while the interpretability of the model is sacrificed. Many RF-based applications have been developed and demonstrated great values in supporting clinical decision makings.<sup>4</sup>

 

Gradient boosting machine (GBM) 
---------------------------------------------------------------
Gradient boosting is a complex yet powerful and popular ensemble algorithm. The way it works is similar to the RF. Whereas the RF algorithm tries to build and combine many well-trained decision trees, the GBM develops an ensemble of shallow trees. The algorithm starts with building a weak shallow tree and sequentially train many trees with each tree learning and improving by addressing the biggest mistake that its previous tree makes. Theoretically, the principles of the gradient boosting framework can be applied to stack any type of model. Applying it to decision trees is most effective in general.4 Thus, we implemented a decision tree-based GBM in this study. The GBM use in medicine includes cancer patient mortality, immune checkpoint inhibitor therapy toxicity, and deterioration of patients with the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) prediction as well as others.<sup>5-8</sup> In this study, we implemented the GBM to model our data using Extreme Gradient Boosting (XGBoost) library in R. 

 

Single-hidden-layer neural network (SHLNN) 
---------------------------------------------------------
Artificial neural networks that solve problems via loosely simulating the way the human brain works have received substantial attention in recent decades. The SHLNN is the simplest form of artificial neural networks  and has shown promising performance in medical classification problems, such as predicting mortality for patients undergoing liver transplantation.<sup>9</sup> The algorithm consists of three basic layers, input, hidden, and output layers. The input layer contains several nodes (neurons) representing original predictors. The algorithm uses the information from the input layer to form a hidden layer with any number of nods assigned with a weight. Next, the algorithm transforms the input information when they pass through the hidden layer’s nodes to the output layer using the equation shown in Eq. 1. The number of hidden layers node and corresponding weight was determined by an iterative learning process.<sup>1</sup>


y = activation (Σ(weight × input) + bias)

 


Model interpretation techniques 
========================================
The interpretability of ML models is one of the top issues being critiqued and barriers to ML implementation in practices, especially health practices. Several explanation techniques have been developed to ‘unbox’ ML models and explain how they use the data to make predictions using a human-friendly way. Two explanation techniques were used to globally explain our models. The first one is permutation variable importance (PVI) which is the most used method to identify important predictors for an ML model. As mentioned in our main paper, the PVI measures the importance of a predictor by measuring the prediction error change when randomly altering the predictor’s value. A predicter is “important” if the prediction error increases when shuffling the predictor’s value.  

The PVI provides a compressed insight into model’s behavior. However, the PVI is not able to measure the association between predictors’ values and model outputs. For instance, the PVI may determine that smoking status is an important factor for regularized LR to predict total flap loss. It does not show whether a current or former smoker is more likely to get a positive prediction. Therefore, a second explanation method, accumulated local effects (ALE), was used in this study.  

The ALE is an expansion of partial dependence plots (PDP) which estimate the relationships between model output and predictors of interest.<sup>10</sup> The ALE uses conditional distribution to avoid counting data points that are not possible in the real world, making it a robust method for estimating predictors’ effects in a highly inter-correlated dataset. For categorical predictors, the ALE estimates the average predicted positive probabilities for each category of the predicters. For numerical predictors, the ALE demonstrates relationships, e.g., linear, monotonic, or more complex, between values of the predictors and predicted positive probabilities on average. In this study, we examined the ALE for predictors that were deemed important for four or more models by the PFI. We averaged the ALE from different ML models for each important predictor to estimate each predictor’s final ALE. All model interpretation methods were performed using the R ‘DALEXtra’ package.<sup>11</sup>

 
