# ===================================================================================================================================================
# =====================                         TRAINING MODELS:                                     ================================================
# =====================  Implementation of Algorithms suitable for binary classification problems:   ================================================
# =====================  - Logistic Regression (LR) Classifier                                       ================================================
# =====================  - K-Nearest Neighbors (KNN) Classifier                                      ================================================
# =====================  - Support Vector Machines (SVM) Classifier                                  ================================================
# =====================  - Naive Bayes (NV) Classifier                                               ================================================
# =====================  - Neural Networks: Multi-Layer Perceptron (MLP) Classifier                  ================================================
# ===================================================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------ General informations ------------------------------------------------------------------------------------------------------------------------------------

__author__ = "Maria Infantino"
__version__ = "1.0.0"
__email__ = "maria.infantino92@gmail.com"
__project__ = "TEST_ML Descartes Underwriting"
__date_ = "18 February 2021"

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------ Requested modules ---------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing, ensemble, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------ Function ------------------------------------------------------------------------------------------------------------------------------------------------

# ================ LOGISTIC REGRESSION CLASSIFIER =======================================================================================================================================

# Logistic Regression is one of the most commonly used machine learning algorithms to predict binary classes (as in this case).
# It works by assigning an output probability between 0 and 1 based on the Sigmoid Function to each input. Based on the threshold which is 0.5 by default, 
# anything higher than that will be categorized as 1 and anything below that as 0.

def logistic_regression(X_names,dtf_train,dtf_test,index_list,output_folder):

        ## Inputs:
        #           X_names                            : features columns selected as input for training
        #           dtf_train                          : training database
        #           dtf_test                           : testing database for prediction
        #           index_list                         : list of index for saving the p_target
        #           output_folder                      : path for saving prediction
        #
        #
        ## Outputs:
        #           model                              : trained model
        #           predicted                          : predicted values
        #
        #
        ## Figures saved:
        #           -
        #
        #
        ## csv saved:
        #           p_target_logistic_regression.csv   : predicted values 
 

        # training data
        X_train = dtf_train[X_names].values #consider ad input only the selected features
        y_train = dtf_train["TARGET_FLAG"].values
        # testing database
        X_test = dtf_test[X_names].values #consider ad input only the selected features
  

	# call model
        model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

        model.fit(X_train,y_train)
        predicted = model.predict(X_test)

        # save predictions
        data = {'INDEX':index_list,
                'p_target':predicted}

        data_csv = pd.DataFrame(data)
        data_csv.to_csv(output_folder+'/p_target_logistic_regression.csv')

        return model, predicted


# =======================================================================================================================================================================================
# ================ K-NEAREST NEIGHBORS (KNN) CLASSIFIER =================================================================================================================================

# KNN Classifier operates by checking the distance from some test example to the known values of some training example. 
# The group of data points/class that would give the smallest distance between the training points and the testing point is the class that is selected.

def k_nearest_neighbors(X_names,dtf_train,dtf_test,index_list,output_folder):

        ## Inputs:
        #           X_names                            : features columns selected as input for training
        #           dtf_train                          : training database
        #           dtf_test                           : testing database for prediction
        #           index_list                         : list of index for saving the p_target
        #           output_folder                      : path for saving prediction
        #
        #
        ## Outputs:
        #           model                              : trained model
        #           predicted                          : predicted values
        #
        #
        ## Figures saved:
        #           -
        #
        #
        ## csv saved:
        #           p_k_nearest_neighbors.csv         : predicted values 
 
  

        # training data
        X_train = dtf_train[X_names].values #consider ad input only the selected features
        y_train = dtf_train["TARGET_FLAG"].values
        # testing database
        X_test = dtf_test[X_names].values #consider ad input only the selected features
  

	# call model
        model = KNeighborsClassifier(n_neighbors=5)

        model.fit(X_train,y_train)
        predicted = model.predict(X_test)

        # save predictions
        data = {'INDEX':index_list,
                'p_target':predicted}

        data_csv = pd.DataFrame(data)
        data_csv.to_csv(output_folder+'/p_k_nearest_neighbors.csv')

        return model, predicted

# =======================================================================================================================================================================================
# ================ SUPPORT VECTOR MACHINES (SVM) CLASSIFIER =======================================================================================================================================

# Support Vector Machines work by drawing a line between the different clusters of data points to group them into classes. 
# Points on one side of the line will be one class and points on the other side belong to another class.

def support_vector_machines(X_names,dtf_train,dtf_test,index_list,output_folder):

        ## Inputs:
        #           X_names                            : features columns selected as input for training
        #           dtf_train                          : training database
        #           dtf_test                           : testing database for prediction
        #           index_list                         : list of index for saving the p_target
        #           output_folder                      : path for saving prediction
        #
        #
        ## Outputs:
        #           model                              : trained model
        #           predicted                          : predicted values
        #
        #
        ## Figures saved:
        #           -
        #
        #
        ## csv saved:
        #           p_support_vector_machines.csv      : predicted values 
 

        # training data
        X_train = dtf_train[X_names].values #consider ad input only the selected features
        y_train = dtf_train["TARGET_FLAG"].values
        # testing database
        X_test = dtf_test[X_names].values #consider ad input only the selected features
  

	# call model
        model = SVC()

        model.fit(X_train,y_train)
        predicted = model.predict(X_test)

        # save predictions
        data = {'INDEX':index_list,
                'p_target':predicted}

        data_csv = pd.DataFrame(data)
        data_csv.to_csv(output_folder+'/p_support_vector_machines.csv')

        return model, predicted


# =======================================================================================================================================================================================
# ================ NAIVE BAYES (NB) CLASSIFIER ==========================================================================================================================================

# A Naive Bayes Classifier determines the probability that an example belongs to some class, calculating the probability that an event will occur given that some input event has occurred.
# It is assumed that all the predictors of a class have the same effect on the outcome, that the predictors are independent.

def naive_bayes(X_names,dtf_train,dtf_test,index_list,output_folder):

        ## Inputs:
        #           X_names                            : features columns selected as input for training
        #           dtf_train                          : training database
        #           dtf_test                           : testing database for prediction
        #           index_list                         : list of index for saving the p_target
        #           output_folder                      : path for saving prediction
        #
        #
        ## Outputs:
        #           model                              : trained model
        #           predicted                          : predicted values
        #
        #
        ## Figures saved:
        #           -
        #
        #
        ## csv saved:
        #           p_naive_bayes.csv                  : predicted values 
 


        # training data
        X_train = dtf_train[X_names].values #consider ad input only the selected features
        y_train = dtf_train["TARGET_FLAG"].values
        # testing database
        X_test = dtf_test[X_names].values #consider ad input only the selected features
  

	# call model
        model = GaussianNB()

        model.fit(X_train,y_train)
        predicted = model.predict(X_test)

        # save predictions
        data = {'INDEX':index_list,
                'p_target':predicted}

        data_csv = pd.DataFrame(data)
        data_csv.to_csv(output_folder+'/p_naive_bayes.csv')

        return model, predicted


# =======================================================================================================================================================================================
# ================ NEURAL NETWORKS: MULTI-LAYER PERCEPTRON (MLP) CLASSIFIER =============================================================================================================

# MLPClassifier relies on an underlying Neural Network to perform the task of classification.

def multilayer_perceptron (X_names,dtf_train,dtf_test,index_list,output_folder):


        ## Inputs:
        #           X_names                            : features columns selected as input for training
        #           dtf_train                          : training database
        #           dtf_test                           : testing database for prediction
        #           index_list                         : list of index for saving the p_target
        #           output_folder                      : path for saving prediction
        #
        #
        ## Outputs:
        #           model                              : trained model
        #           predicted                          : predicted values
        #
        #
        ## Figures saved:
        #           -
        #
        #
        ## csv saved:
        #           p_target_multilayer_perceptron.csv : predicted values 

 

        # training data
        X_train = dtf_train[X_names].values #consider ad input only the selected features
        y_train = dtf_train["TARGET_FLAG"].values
        # testing database
        X_test = dtf_test[X_names].values #consider ad input only the selected features

        ''' 
	# design the MLP classifier by searching different combinations of hyperparameters
	parameter_space = {
	    'hidden_layer_sizes': [(10,5,2), (6,6,6), (6,)],    
	    'activation': ['tanh', 'relu'],
	    'solver': ['sgd', 'adam'],
	    'alpha': [0.0001, 0.05],
	    'learning_rate': ['constant','adaptive'],
	}

        mlp = MLPClassifier(max_iter=100)
        model = GridSearchCV(mlp, parameter_space)
        '''

        # design the MLP classifier : 3 layers 
        # (as rule of thumb the number of neurons should be lower than the number of inputs)
        n_input = len(X_names) # number of inputs
        nnl1 = int(round(n_input/1.5)) # number of neurons layer 1 
        if nnl1 < 2:
           nnl1 = 2
        nnl2 = int(round(n_input/3))  # number of neurons layer 2
        if nnl2 < 2:
           nnl2 = 2
        nnl3 = 2    # number of neurons layer 3
        hidden_layer_size = tuple([nnl1,nnl2,nnl3]) # number of neurons per layer - must be tuple
        number_hidden_layer = len(hidden_layer_size) # number of hidden layers
	activation = 'relu' # activation function for the hidden layer 
	alpha = 0.0001 # L2 penalty (regularization term) parameter.
	max_iter = 100000 # maximum number of epochs of training
	model = MLPClassifier(activation=activation,hidden_layer_sizes=hidden_layer_size,
		                alpha=alpha,random_state=0, max_iter=max_iter,warm_start=False)
    
        model.fit(X_train,y_train)    
        predicted = model.predict(X_test)

        # save predictions
        data = {'INDEX':index_list,
                'p_target':predicted}

        data_csv = pd.DataFrame(data)
        data_csv.to_csv(output_folder+'/p_target_multilayer_perceptron.csv')

        return model, predicted

    
