	Online supplementary materials and code repository
# Machine learning (ML) technique overview 

In this page, we briefly introduce each ML algorithm and model interpretation techniques. We cover the following topic on this page:
1. [An overview of various machine learning algorithms](#machine-learning-algorithm-overview)
2. [A brief introduction to two global and one local model-agnostic explanation approaches](#model-interpretation-techniques) 
<br></br>

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


<p align ="center">
	y = activation (Σ(weight × input) + bias)
	</p>
<br></br>

Model interpretation techniques 
========================================
The interpretability of ML models is one of the top issues being critiqued and barriers to ML implementation in practices, especially health practices. Several model-agnostic explanation techniques have been developed to ‘unbox’ ML models and explain how they use data to make predictions using a human-friendly way. Permutation variable importance (PVI) is the most used method to identify important predictors for an ML model. The PVI measures the importance of a predictor by measuring the prediction error change when randomly altering the predictor’s value. A predicter is “important” if the prediction error increases when shuffling the predictor’s value.  

Permutation variable importance (PVI) 
--------------------------------------------------------
The PVI provides a compressed insight into model’s behavior. However, the PVI is not able to measure the associations between predictors’ values and model outputs. For instance, the PVI may determine that smoking status is an important factor for a regularized LR model to predict total flap loss. It does not show whether a current or former smoker is more likely to get a positive prediction. Therefore, we need additional techniques, such as accumulated local effects (ALE) to uncover the asscoations.  

Accmulated local effects (ALE) 
--------------------------------------------------------
The ALE is an expansion of partial dependence plots (PDP) which estimate the relationships between model output and predictors of interest.<sup>10</sup> The ALE uses conditional distribution to avoid counting data points that are not possible in the real world, making it a robust method for estimating predictors’ effects in a highly inter-correlated dataset. For categorical predictors, the ALE estimates the average predicted positive probabilities for each category of the predicters. For numerical predictors, the ALE demonstrates relationships, e.g., linear, monotonic, or more complex, between values of the predictors and predicted positive probabilities on average. Oftentimes, we examin the ALE for predictors that were deemed important by the PVI.

SHapely Additive exPlanations (SHAP) 
--------------------------------------------------------
In addition to global model-agnostic methods, there are many explanation approaches that provide additional insights into model outputs at the instance level (single patient), which is especially useful for entrusting clinicians and supporting individualized care. SHapely Additive exPlanations (SHAP), an approach inspired by the game theory, is increasingly used to provide such insights.<sup>10</sup> Specifically, SHAP computes the contribution of each predictor value of an instance by comparing the output and the mean output if we vary the predictor’s value with all other possible values and any combination of other predictors. The result could deliver a message, such as the model prediction increases by 10% the risk of hospital admission for patient A due to being a smoker compared to the average prediction. Interested readers can find more detailed information about SHAP in <sup>10</sup>.
 
<br></br>

References
=====================
1. Sidey-Gibbons JAM, Sidey-Gibbons CJ. Machine learning in medicine: a practical introduction. BMC Med Res Methodol. 2019;19(1):1–18.  

2. Sidey-Gibbons C, Asaad M, Pfob A, Boukovalas S, Lin Y-L, Offodile A. Machine learning algorithms to predict financial toxicity associated with breast cancer treatment. J Clin Oncol. 2020 May 20;38(15_suppl):2047.  

3. Pal M. Random forest classifier for remote sensing classification. Int J Remote Sens. 2005 Jan 22;26(1):217–22.  

4. Medic G, Kließ MK, Atallah L, Weichert J, Panda S, Postma M, et al. Evidence-based clinical decision support systems for the prediction and detection of three disease states in critical care: a systematic literature review. F1000Research. 2019;8(1728).  

5. Xu Y, Yang X, Huang H, Peng C, Ge Y, Wu H, et al. Extreme gradient boosting model has a better performance in predicting the risk of 90-day readmissions in patients with ischaemic stroke. J Stroke Cerebrovasc Dis. 2019;28(12):104441.  

6. Manz C, Chivers C, Liu MQ, Regli SB, Changolkar S, Evans CN, et al. Prospective validation of a machine learning algorithm to predict short-term mortality among outpatients with cancer. J Clin Oncol. 2020;38(15).  

7. Iivanainen S, Ekström J, Virtanen H, Koivunen J. Predicting onset and continuity of patient-reported symptoms in patients receiving immune checkpoint inhibitor (ICI) therapies using machine learning. Arch Clin Med Case Reports. 2020;04(03):344–51.  

8. Heldt FS, Vizcaychipi MP, Peacock S, Cinelli M, McLachlan L, Andreotti F, et al. Early risk assessment for COVID-19 patients from emergency department data using machine learning. Sci Rep. 2021;11(1):1–13. 

9. Zhang M, Yin F, Chen B, Li B, Li YP, Yan LN, et al. Mortality risk after liver transplantation in hepatocellular carcinoma recipients: a nonlinear predictive model. Surgery. 2012;151(6):889–97.  

10. Molnar C. Interpretable machine learning: a guide for making black box models explainable [Internet]. 2020. Available from: https://christophm.github.io/interpretable-ml-book/ 
