# ==============================================================================================================================================================================================
# =====================                            EVALUATE MODEL                             ==================================================================================================
# =======    Measure the Performance of the considered model by means of the following metrics:     ============================================================================================
# =======    - Accuracy         : ratio of correct predictions to total predictions                        =====================================================================================
# =========  - Auc              : area under the ROC curve, represents the model's ability to properly discriminate between one class or another (for a perfect classifier AUC = 1). ===========
# =========  - Precision        : True Positive (TP)/[TP + False Positive (FP)]                            =====================================================================================
# =========  - Recall           : TP/[TP + False Negative (FN)]                                            =====================================================================================
# =========  - F-Score          : (2 Recall x Precision)/(Recall+Precision)                                =====================================================================================
# ==============================================================================================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------ General informations ------------------------------------------------------------------------------------------------------------------------------------

__author__ = "Maria Infantino"
__version__ = "1.0.0"
__email__ = "maria.infantino92@gmail.com"
__project__ = "TEST_ML Descartes Underwriting"
__date_ = "February 2021"

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------ Requested modules ---------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------ Function ------------------------------------------------------------------------------------------------------------------------------------------------

# ================ EVALUATE MODEL ============================================================================================================================================

def evaluate_models(X_names,dtf_test,model):


        ## Inputs:
        #           X_names  : features columns selected as input for training
        #           dtf_test : testing database
        #           model    : trained model
        #
        #
        ## Outputs:
        #           accuracy    : accuracy
        #           auc         : auc
        #           det_metr    : classification report (Precision, Recall, F-Score) for each class
        #
        #
        ## Figures saved:
        #           -
        #
        #
        ## csv saved:
        #          -

 

        ## Testing data
        X_test = dtf_test[X_names].values #consider ad input only the selected features
        y_test = dtf_test["TARGET_FLAG"].values
       

        ## Prediction according to model 
        predicted = model.predict(X_test)
        #predicted_prob = model.predict_proba(X_test)

        ## Compute Evaluation Metrics: 

        # Accuracy
        accuracy = metrics.accuracy_score(y_test, predicted)
        print("Accuracy (overall correct predictions): "+ str(round(accuracy,2)))
        # Auc  
	auc = metrics.roc_auc_score(y_test, predicted)
	print("AUC: " + str(round(auc,2)))
        # Recall
        recall = metrics.recall_score(y_test, predicted)
        print("Recall (all 1s predicted right): "+str(round(recall,2)))
        # Precision
        precision = metrics.precision_score(y_test, predicted)
        print("Precision (confidence when predicting 1): "+ str(round(precision,2)))
        # Classification Report: Precision, Recall, F-Score,Support
        det_metr = metrics.classification_report(y_test, predicted)
        print(det_metr)

  
        return accuracy, auc, det_metr


        










