'''
Functions used to prepare data, test and tune several Machine learning models.
Jesus I. Ramirez Franco
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score as accuracy 
from sklearn.metrics import f1_score as f1
from sklearn.metrics import precision_score as precision 
from sklearn.metrics import recall_score as recall
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import ParameterGrid
import pandas as pd
from sklearn.preprocessing import StandardScaler


def standardrize(df):
	'''
	Standardrizes a pandas data frame.
	Input:
		- df (Pandas data frame): Data frame that contains the features
		  used in the model.
	Returns a Pandas data frame.
	'''
	scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
	scaler.fit(df)
	return scaler.transform(df)

def predict_model(model, split, params_dict):
	'''
	Generates the predictions based in a ML model.
	Inputs:
		- model (scikit-learn classifier model): method to be used to get predictions.
		- split (dict): a dictionary that contains the data separated in 4 groups, 
		  x_train, x_test, y_train and y_test
		- params_dict (ParameterGrid): object that contains the parameters of the model.
	Returns the params used and an array with predictions.
	'''
	params_str = str(model)+'.- '+', '.join([k+': '+str(v) for k, v in params_dict.items()])
	mod = model(**params_dict)
	mod.fit(split['x_train'], split['y_train'])
	pred = mod.predict(split['x_test'])
	return params_str, pred

def get_metrics(prediction, y_test):
	'''
	Computes accuracy, precision, recall, ROC-AUC and F1 metrics for 
	consideroing predictions produced by a ML and actual values of a 
	dependent variables.
	Inputs:
		- prediction: an array with predictions.
		- y_test: an array with actual values.
	Returns a dictionary with metrics of a ML model.
	'''
	Accuracy = accuracy(prediction, y_test)
	Precision = precision(prediction, y_test)
	Recall = recall(prediction, y_test)
	try:
		AUC = roc_auc(prediction, y_test)
	except ValueError:
		AUC = 0
	F1 = f1(prediction, y_test)
	
	metrics_dict = {
		'Accuracy': Accuracy,
		'Precision': Precision,
		'Recall': Recall,
		'AUC': AUC,
		'F1': F1
	}
	return metrics_dict

def present_results(results_dict, sort_var):
	'''
	Creates a Pandas data frame from a dcitionary.
	Inputs:
		- results_dict (dict): a dictionary where every k is a string with
		  parameters of a model and every value is a dictionary with metrics of
		  that model.
		- sort_var (str): name of the variable to sort the rows of the 
		  data frame.
	Returns a Pandas data frame
	'''
	df = pd.DataFrame.from_dict(results_dict, orient='index').sort_values(by=sort_var, ascending=False)
	return df

def tune_model(model, split, grid):
	'''
	Creates different versions of a ML model according to a parameters 
	grid and computes their metrics.
	Inputs:
		- model (scikit-learn classifier model): method to be used to get predictions.
		- split (dict): a dictionary that contains the data separated in 4 groups, 
		  x_train, x_test, y_train and y_test
		- grid (ParameterGrid): object that contains the parameters of the model.
	'''
	results = {}
	for params in grid:
		params_str, pred = predict_model(model, split, params)
		results[params_str] = get_metrics(pred, split['y_test'])
	return results


def different_models(model_grid_dict, split, sort_var='AUC'):
	'''
	Computes different ML models and different versions of everyone, based on 
	different parameters grids. 
	Inputs: 
		- model_grid_dict (dict): dictionary where every key is the name of the model 
		  and every value is a dictionary containing the scikit-learn method and the grid
		  with the paramters.
		- split (dict): a dictionary that contains the data separated in 4 groups, 
		  x_train, x_test, y_train and y_test
		- sort_var (str): name of the metric to sort the values in descending order.
	Returns a data frame with the results.
	'''
	results = pd.DataFrame()
	for k, v in model_grid_dict.items():
		df = present_results(tune_model(v['model'], split, v['grid']), sort_var)
		results = results.append(df)
	return results.sort_values(by='AUC', ascending=False)

def specific_model(model, split, params_dict):
	'''
	Creates a model class object.
	Inputs:
		- model (scikit-learn classifier model): method to be used to get predictions.
		- split (dict): a dictionary that contains the data separated in 4 groups, 
		  x_train, x_test, y_train and y_test
		- params_dict (ParameterGrid): object that contains the parameters of the model.
	Returns the params used and an array with predictions.
	'''
	params_str = str(model)+'.- '+', '.join([k+': '+str(v) for k, v in params_dict.items()])
	mod = model(**params_dict)
	mod.fit(split['x_train'], split['y_train'])
	return mod

def important_features(mod, top=None):
	'''
	Identifies the index of the most importnat predictors of the model.
	Inputs:
	mod (scikit-learn model): model trained.
	top (int): number of rows consider in the results.
	Returns a Pandas data frame
	'''
	features = mod.feature_importances_
	features_df = pd.DataFrame(features).sort_values(by=0, ascending=False)
	#features_df = features_df.reset_index()
	features_df.columns = ['imp']
	if top:
		return features_df[:10]
	else:
		return features_df

def average_df(df_list, sort_var):
	'''
	Computes the average of a list of data frames
	Inputs:
		- df_list (list): a list with Pandas data frames with the same structure.
	Returns a Pandas data frame
	'''
	n = len(df_list)
	res = df_list[0]
	for df in df_list[1:]:
		res += df
	res = res/n
	return res.sort_values(by=sort_var, ascending=False)