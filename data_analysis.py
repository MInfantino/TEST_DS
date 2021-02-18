# ==========================================================================================================================
# =======================                DATA ANALYSIS:                  ===================================================
# =======================   Summarize and visualize the dataset content  ===================================================
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
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------------------
# ------------------ Function -------------------------------------------------------------------
def data_analysis(database):

        ## Inputs:
        #           database      :   set of data to be analysed
        #
        #
        ## Outputs:
        #           -
        #
        #
        ## Figures saved:
        #           -
        #
        #
        ## csv saved:
        #           -



	## Dimension of the dataset
        dim = database.shape
        #print('Dataset size: '+ str(dim))

	## Statistical summary of each attribute
        stat_sum = database.describe()
        #print('Dataset statistical summary:')
        #print(stat_sum)

	## Target Distribution: check the number of observations that belong to each class
	class_distr = database.groupby('TARGET_FLAG').size()
        #print('Target distribution:')
        #print(class_distr)

        ## Histogram of each numeric input variable 
	database.drop(["INDEX","TARGET_AMT","TARGET_FLAG"], axis=1).hist()

        ## Multivariate plot: interaction between numerical variables
	pd.scatter_matrix(database.drop(["INDEX","TARGET_AMT","TARGET_FLAG"], axis=1))
	#plt.show()
