# ==========================================================================================================================================================
# =====================                                   FEATURES SELECTION :                    ==========================================================
# =======    Implementation of strategies to select a subset of relevant variables to build the machine learning model.        =============================
# =======    The following functions are included:                                                                             =============================
# =========  - compute_correlation: compute the Pearson correlation coefficient between all the input Features to check how they are correlated. ===========
# =========  - random_forest_classifier: compute importance of the features according to Random Forest Classifier (RFC) approach                 ===========
# =========  - recursive_feature_elimination: compute importance of the features according to Recursive Feature Elimination (RFE) approach       ===========
# ==========================================================================================================================================================

# -----------------------------------------------------------------------------------------------------------------------------
# ------------------ General informations -------------------------------------------------------------------------------------

__author__ = "Maria Infantino"
__version__ = "1.0.0"
__email__ = "maria.infantino92@gmail.com"
__project__ = "TEST_ML Descartes Underwriting"
__date_ = "18 February 2021"

# -----------------------------------------------------------------------------------------------------------------------------
# ------------------ Requested modules ----------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import preprocessing,ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import os

# ------------------------------------------------------------------------------------------------------------------------------
# ------------------ Functions -------------------------------------------------------------------------------------------------


# ========================= Correlation between the input variables ============================================================
# Compute Pearson correlation coefficient between each feature 
 
def compute_correlation(database,path_output):

        ## Inputs:
        #           database                     :   datase investigated
        #           path_output                  :   path where save outputs
        #
        #
        ## Outputs:
        #           -
        #
        #
        ## Figures saved:
        #           features_correlation_map.png : correlation matrix of the features 
        #
        #
        ## csv saved:
        #           -


	#Correlation
        df = database.drop(["TARGET_FLAG","TARGET_AMT"], axis=1)
	correlation = df.corr()

	#Tick labels
	matrix_cols = correlation.columns.tolist()

        #Plotting
        fig = plt.figure(figsize=(19, 15))
        plt.matshow(correlation, fignum=fig.number)
        plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=10, rotation=90)
        plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=10)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)  
        cb.set_label('Pearson Correlation Coefficient', fontsize=16)
        #plt.show()

        ## Save Figure 
        output_folder = path_output+'/Features_Importance_Analysis'
        if os.path.exists(output_folder)!= 1:
           os.mkdir(output_folder) 
        fig.savefig(output_folder+'/features_correlation_map.png', bbox_inches = "tight")



# ================ Features Importance according to Random Forest Classifier ===================================================
# This technique creates decision trees on randomly selected data samples, gets prediction from each tree and selects the best 
# solution by means of voting. It also provides a pretty good indicator of the feature importance.

def random_forest_classifier(database,path_output,nFeat):


        ## Inputs:
        #           database                                      :   datase investigated
        #           path_output                                   :   path where save outputs
        #           nFeat                                         :   number of input Features to extract (scalar)
        #
        #
        ## Outputs:
        #           cols                                          :   list of nFeat most important features 
        #
        #
        ## Figures saved:
        #           features_importance_RandomForestClassfier.png :   importance of each feature 
        #
        #
        ## csv saved:
        #           features_importance_RandomForestClassfier.csv :   importance of each feature  




	X = database.drop(["TARGET_FLAG","TARGET_AMT"], axis=1).values
	y = database["TARGET_FLAG"].values
	feature_names = database.drop(["TARGET_FLAG","TARGET_AMT"], axis=1).columns.tolist()

        ## Importance
	model = ensemble.RandomForestClassifier(n_estimators=100,criterion="entropy", random_state=0)
	model.fit(X,y)
	importances = model.feature_importances_

        ## Put in a pandas dataframe
	dtf_importances = pd.DataFrame({"IMPORTANCE":importances,"VARIABLE":feature_names}).sort_values("IMPORTANCE",ascending=False)
	dtf_importances['cumsum'] =  dtf_importances['IMPORTANCE'].cumsum(axis=0)
	dtf_importances = dtf_importances.set_index("VARIABLE")

	## Plot
	fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
	fig.suptitle("Features Importance according to Random Forest Classifier", fontsize=12)
	ax[0].title.set_text('variables')
	dtf_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(kind="barh", legend=False, ax=ax[0],fontsize = 5).grid(axis="x")
	ax[0].set(ylabel="")

	ax[1].title.set_text('cumulative')
	dtf_importances[["cumsum"]].plot(kind="line", linewidth=4,legend=False, ax=ax[1])
	ax[1].set(xlabel="", xticks=np.arange(len(dtf_importances)),xticklabels=dtf_importances.index)

	plt.xticks(rotation=90,fontsize = 5)
	plt.grid(axis='both')
	#plt.show()

        idc_rfc = dtf_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE",ascending = False)

        cols_impo = idc_rfc[0:nFeat] #importance table of the selected features
        cols = list(cols_impo.index) #list of the selected features

        ## Save Figure and .csv of the Feature Importance
        output_folder = path_output+'/Features_Importance_Analysis'
        if os.path.exists(output_folder)!= 1:
           os.mkdir(output_folder) 
        fig.savefig(output_folder+'/features_importance_RandomForestClassfier.png', bbox_inches = "tight")

        data_csv = pd.DataFrame(idc_rfc)
        idc_rfc.to_csv(output_folder+'/features_importance_RandomForestClassfier.csv')
        

        return cols

# ================ Features Importance according to Recursive Feature Elimination (RFE) ========================================
# This technique begins by building a model on the entire set of predictors and computing an importance score for each predictor. 
# The least important predictor(s) are then removed, the model is re-built, and importance scores are computed again. 
# The subset size is a tuning parameter for RFE. The optimal subset is then used to train the final model.

def recursive_feature_elimination(database,path_output,nFeat):


        ## Inputs:  
        #           database                                            :   datase investigated
        #           path_output                                         :   path where save outputs
        #           nFeat                                               :   number of input Features to extract (scalar)
        #
        #
        ## Outputs:
        #           cols                                                :   list of nFeat most important features 
        #
        #
        ## Figures saved:
        #           -
        #
        #
        ## csv saved:
        #           features_importance_RacursiveFeatureElimination.csv :   ranking of features




        X = database.drop(["TARGET_FLAG","TARGET_AMT"], axis=1)
        y = database["TARGET_FLAG"]
	log = LogisticRegression()

        # Assumption: a subset of 15 features has been considered
	rfe = RFE(log,nFeat)
	rfe = rfe.fit(X.values,y.values.ravel())

	#identified columns Recursive Feature Elimination
	idc_rfe = pd.DataFrame({"rfe_support" :rfe.support_,
		               "columns" : [i for i in X.columns],
		               "ranking" : rfe.ranking_,
		              })
	cols = idc_rfe[idc_rfe["rfe_support"] == True]["columns"].tolist()

        ## Save .csv of the Feature Importance
        output_folder = path_output+'/Features_Importance_Analysis'
        if os.path.exists(output_folder)!= 1:
           os.mkdir(output_folder) 

        data_csv = pd.DataFrame(idc_rfe)
        idc_rfe.to_csv(output_folder+'/features_importance_RacursiveFeatureElimination.csv')

        return cols



