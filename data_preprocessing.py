# ==========================================================================================================================
# =====================             DATA PREPROCESSING:                     ================================================
# =====================  1) Convert currency into numerical values          ================================================
# =====================  2) Treat NaN values                                ================================================
# =====================  3) Convert categorical columns into numerical      ================================================
# =====================  4) Scale data through normalization                ================================================
# ==========================================================================================================================

# -----------------------------------------------------------------------------------------------
# ------------------ General informations -------------------------------------------------------

__author__ = "Maria Infantino"
__version__ = "1.0.0"
__email__ = "maria.infantino92@gmail.com"
__project__ = "TEST_ML Descartes Underwriting"
__date_ = "18 February 2021"

# -----------------------------------------------------------------------------------------------
# ------------------ Requested modules ----------------------------------------------------------
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing

# -----------------------------------------------------------------------------------------------
# ------------------ Function -------------------------------------------------------------------
def data_preprocessing(database):

        ## Inputs:
        #           database        :   orginal set of data to be (pre)processed
        #
        #
        ## Outputs:
        #           database_scaled :   resulting processed dataset
        #
        #
        ## Figures saved:
        #           -
        #
        #
        ## csv saved:
        #           -


        # Drop first column INDEX
        database = database.drop(["INDEX"], axis=1)

	# Convert currency into numerical values
	def replaceDollar(x):
	    if x.astype(str).str.contains(r'\$').any():        
		x=x.replace('[\$,]', '', regex=True).astype(np.float64)
	    return x
	database = database.apply(lambda x: replaceDollar(x),axis=0)

	# Replace NaN values with average of each column
	database = database.fillna(database.mean())

	# Convert categorical columns into numerical 
	num_cols = database._get_numeric_data().columns #numerical columns
	cat_cols = list(set(database.columns) - set(num_cols)) #categorical columns
	database = pd.get_dummies(data = database,columns = cat_cols ) #convert categorical into numerical

	# Scale data through normalization (if the distribution of data is normal use StandardScaler otherwise MinMaxScaler)
	k2, p = stats.normaltest(np.array(database.drop(["TARGET_FLAG","TARGET_AMT"], axis=1)), axis=0, nan_policy='propagate') # perform statistical test to check if the data are normally distributed
  
	if(np.all(p>0.05)): # hypothesis test accepted: data can be considered normally distributed
	  scaler = preprocessing.StandardScaler()  # use StandardScaler for scaling
	  X = scaler.fit_transform(database.drop(["TARGET_FLAG","TARGET_AMT"], axis=1)) # scale input columns
	  database_scaled= pd.DataFrame(X, columns=database.drop(["TARGET_FLAG","TARGET_AMT"], axis=1).columns, index=database.index) # convert scaled to pandas
	  database_scaled["TARGET_FLAG"] = database["TARGET_FLAG"] # add outputs columns to the scaled pandas 
	  database_scaled["TARGET_AMT"] = database["TARGET_AMT"] # add outputs columns to the scaled pandas 
	else: # hypothesis test rejected: data cannot be considered normally distributed
	  scaler = preprocessing.MinMaxScaler(feature_range=(0,1))  # use MinMaxScaler for scaling
	  X = scaler.fit_transform(database.drop(["TARGET_FLAG","TARGET_AMT"], axis=1)) # scale input columns
	  database_scaled= pd.DataFrame(X, columns=database.drop(["TARGET_FLAG","TARGET_AMT"], axis=1).columns, index=database.index) # convert scaled to pandas
	  database_scaled["TARGET_FLAG"] = database["TARGET_FLAG"] # add outputs columns to the scaled pandas 
	  database_scaled["TARGET_AMT"] = database["TARGET_AMT"] # add outputs columns to the scaled pandas 

	return database_scaled
