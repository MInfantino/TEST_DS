# ==========================================================================================================================
# =====================             Prediction on the dataset AUTO INSURANCE               =================================
# ==========================================================================================================================
print('\nstart...\n\n')
print('===============================================================================================')
print('\t\t\t TEST ML: Prediction on the dataset AUTO INSURANCE ')
print('===============================================================================================')

# -------------------------------------------------------------------------------------------------
# ------------------ General informations ---------------------------------------------------------

__author__ = "Maria Infantino"
__version__ = "1.0.0"
__email__ = "maria.infantino92@gmail.com"
__project__ = "TEST_ML Descartes Underwriting"
__date_ = "18 February 2021"

# ---------------------------------------------------------------------------------------------------------------------------
# ------------------ Requested modules --------------------------------------------------------------------------------------
# for data
import pandas as pd
import numpy as np
# for plotting
import matplotlib.pyplot as plt
# for statistics
from scipy import stats
# for machine learning
from sklearn import preprocessing, ensemble, metrics, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

# ---------------------------------------------------------------------------------------------------------------------------
# ------------------ Define Inputs ------------------------------------------------------------------------------------------

# number of features selected for the training (in Feature Selection Section)
nFeat = 15

# path where input data are stored: 
path_inputdata = 'data_input/auto-insurance-fall-2017/'
# path where results are saved: 
path_output = 'results/'

# ---------------------------------------------------------------------------------------------------------------------------
# ------------------ Load Inputs --------------------------------------------------------------------------------------------

nSection = 0 # initialisation of the section list for printing

nSection = nSection + 1
print('\n'+str(nSection)+') Load Input Data')

# load training database
train_db_or = pd.read_csv(path_inputdata+'train_auto.csv',sep=',') 
# load blind test database (for blinding prediction)
blind_test_db = pd.read_csv(path_inputdata+'test_auto.csv',sep=',') 

# ---------------------------------------------------------------------------------------------------------------------------
# ------------------ Data Analysis ------------------------------------------------------------------------------------------
# Summarize and visualize the content of dataset

nSection = nSection + 1
print('\n'+str(nSection)+') Data Analysis')

from data_analysis import data_analysis
data_analysis(train_db_or)


# ---------------------------------------------------------------------------------------------------------------------------
# ------------------ Data Preprocessing -------------------------------------------------------------------------------------
# Preprocess data into a readable format:
# 1) Split original training database into a effective training and testing database
# 2) Convert currency into numerical values          
# 3) Traiting NaN values 
# 4) Convert categorical columns into numerical       
# 5) Scale data through normalization  

nSection = nSection + 1
print('\n'+str(nSection)+') Data Preprocessing')

# split original training database into training and testing
train_db, test_db = model_selection.train_test_split(train_db_or,test_size=0.2)

from data_preprocessing import data_preprocessing
# preprocess train database
train_db_scaled = data_preprocessing(train_db)
# preprocess test database
test_db_scaled = data_preprocessing(test_db)
# preprocess blind test database
blind_test_db_scaled = data_preprocessing(blind_test_db)


# ---------------------------------------------------------------------------------------------------------------------------
# ------------------ Feature Selection --------------------------------------------------------------------------------------
# Select a subset of relevant variables to build the machine learning model:
# 1) Compute the Pearson Correlation Coefficient between all the input Features to check how they are correlated.
# 2) Compute Importance of the features according to Random Forest Classifier (RFC) Approach
# 3) Compute Importance of the features according to Recursive Feature Elimination (RFE) Approach

nSection = nSection + 1
print('\n'+str(nSection)+') Features Selection Strategies:')

from features_selection import random_forest_classifier,compute_correlation,recursive_feature_elimination
# Compute correlation between all the features
print('\t Features Correlation Matrix')
compute_correlation(train_db_scaled,path_output)
# Compute Importance of the features according to Random Forest Classifier (RFC)
print('\t Random Forest Classifier (RFC)')
selFeatures_RFC = random_forest_classifier(train_db_scaled,path_output,nFeat)
# Compute Importance of the features according to Recursive Feature Elimination (RFE)
print('\t Recursive Feature Elimination (RFE)')
selFeatures_RFE = recursive_feature_elimination(train_db_scaled,path_output,nFeat)

# Here the features selected with the RFE have been preferred 
selFeatures = selFeatures_RFE #selFeatures_RFC

# ---------------------------------------------------------------------------------------------------------------------------
# ------------------ Train, Predict and Evaluate Peformance of ML Model -----------------------------------------------------
# Apply different algorithms suitable for binary classification problems:
# - Logistic Regression (LR) Classifier                                       
# - K-Nearest Neighbors (KNN) Classifier                                    
# - Support Vector Machines (SVM) Classifier                             
# - Naive Bayes (NV) Classifier                                   
# - Neural Networks: Multi-Layer Perceptron (MLP) Classifier       


nSection = nSection + 1
print('\n'+str(nSection)+') Train, Predict and Evaluate Performance of ML Models:')

from training_models import logistic_regression, multilayer_perceptron,k_nearest_neighbors,support_vector_machines,naive_bayes
from evaluate_models import evaluate_models

print('\n ------------------------- LOGISTIC REGRESSION (LR) CLASSIFIER ----------------------------\n')
# Train Model and Save Prediction
model_lre,predicted_lre = logistic_regression(selFeatures,train_db_scaled,blind_test_db_scaled,blind_test_db["INDEX"].values,path_output)
# Evaluate Performance of the Model
accuracy_lre, auc_lre, class_rep_lre = evaluate_models(selFeatures,test_db_scaled,model_lre)

print('\n ------------------------- K-NEAREST NEIGHBORS (KNN) CLASSIFIER  ----------------------------\n')
# Train Model and Save Prediction
model_knn,predicted_knn = k_nearest_neighbors(selFeatures,train_db_scaled,blind_test_db_scaled,blind_test_db["INDEX"].values,path_output)
# Evaluate Performance of the Model
accuracy_knn, auc_knn, class_rep_knn = evaluate_models(selFeatures,test_db_scaled,model_knn)

print('\n ------------------------- SUPPORT VECTOR MACHINES (SVM) CLASSIFIER  -------------------------\n')
# Train Model and Save Prediction
model_svm,predicted_svm = support_vector_machines(selFeatures,train_db_scaled,blind_test_db_scaled,blind_test_db["INDEX"].values,path_output)
# Evaluate Performance of the Model
accuracy_svm, auc_svm, class_rep_svm = evaluate_models(selFeatures,test_db_scaled,model_svm)

print('\n ------------------------- NAIVE BAYES CLASSIFIER (NB) ----------------------------------------\n')
# Train Model and Save Prediction
model_nb,predicted_nb = naive_bayes(selFeatures,train_db_scaled,blind_test_db_scaled,blind_test_db["INDEX"].values,path_output)
# Evaluate Performance of the Model
accuracy_nb, auc_nb, class_rep_nb = evaluate_models(selFeatures,test_db_scaled,model_nb)

print('\n ------------------------- NEURAL NETWORKS: MULTI-LAYER PERCEPTRON (MLP) CLASSIFIER ----------\n')

# Train Model and Save Prediction
model_mlp,predicted_mlp = multilayer_perceptron(selFeatures,train_db_scaled,blind_test_db_scaled,blind_test_db["INDEX"].values,path_output)
# Evaluate Performance of the Model
accuracy_mlp, auc_mlp, class_rep_mlp = evaluate_models(selFeatures,test_db_scaled,model_mlp)


print('\n===============================================================================================')
print('\n\n\n...end')
