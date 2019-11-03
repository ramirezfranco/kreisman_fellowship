import urllib3
import math
import random
from scipy import stats
import pandas as pd
import geopandas as gpd
import json
import creds
from sodapy import Socrata
from shapely.geometry import Point



def points_in_polygons(point_gdf, polygon_gdf):
	'''
	Count the points within polygons and returns a data frame 
	where every row is a tract and indicates the total numbers of
	points in that polygon
	
	Inputs:
		point_gdf: Geodataframe with information of geographic points.
		polygon_gdf: Geodataframe with information of geographic polygons of 
		neighborhoods.
		file_name: name of the file where the information is going to be stored.
	 Returns: a data frame and a csv file
	'''
	working_list = []
	for index, point in point_gdf.iterrows():
		for i, polygon in polygon_gdf.iterrows():
			if point.geometry.within(polygon.geometry) == True:
				working_list.append([index] + list(point) + [polygon['tractce10']])
				pass

	df = pd.DataFrame(working_list, columns = point_gdf.columns+['tract'])
	return df


def get_data(dataset_identifier, content_type="json", **kwargs):
	'''
	Gets datasets from Chicago Data portal.
	Inputs:
		- dataset_identifier (str): identifier of the dataset. See documentation 
		  in Chicago data portal.
		- content_type (str): type of response.
		- **kwargs (dict): parameters of the request. 
	Returns a Pandas data frame
	'''
	client = Socrata("data.cityofchicago.org", creds.dataportal_token)
	resp = client.get(dataset_identifier, content_type="json", **kwargs)
	df = pd.DataFrame.from_records(resp)
	return df


def str2points(df, long_name, lat_name, outfile_name):
	'''
	Converts georeferences saved as strings to Point objects.
	Inputs:
		- df (Pandas data frame):
		- long_name and lat_name (str):
		- outfile_name (str):
	Returns a geopandas data frame.
	'''
	geometry = [Point(xy) for xy in zip(df[long_name], df[lat_name])]
	crs = {'init': 'epsg:4326'}
	geo_df = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
	geo_df.to_file("clean_data/"+outfile_name)
	return geo_df


def make_url_req(var_list, year, state='17', county='031'):
	'''
	Forms a URL to make resquests in the ACS5 API at the tract level
	Inputs:
		- var_list (str): name of the variable. See names of the variables 
		  in https://api.census.gov/data.
		- year (str): year of the data.
		- state='17' (str): State FISP code.
		- county='031' (str): County FISP code.
	Returns a URL string.
	'''
	base = 'https://api.census.gov/data/{}/acs/acs5/subject?get=NAME,{}&for=tract:*&in=state:{}%20county:{}&key={}'
	variables = ','.join(var_list)
	return base.format(year, variables, state, county, creds.census_key)


def get_json(url):
	'''
	Decodes a json object from a URL request.
	Inputs:
		-url(str): in this case, a url containing the parameters needed to make
		 a request in the Census API. But could be used to open any json 
		 object from a url.
	Returns: in this case, a list of list. But returns the information 
	contained in the json object.
	'''
	http = urllib3.PoolManager()
	r = http.request('GET', url)
	if r.data == b'':
		return 'no data found'
	else:
		return json.loads(r.data.decode('utf-8'))


def num_to_str(df, col):
	'''
	Converts a column with numbers to a column with strings.
	Inputs:
		- df (Pandas data frame): Data frame that contains the column of
		  interest.
		- col (str): name of the column.
	Nothing, only modifies the data frame.
	'''
	df[col] = df[col].apply(lambda x: str(x))


def str_to_float(df, col):
	'''
	Converts a column with strings to a column with floats.
	Inputs:
		- df (Pandas data frame): Data frame that contains the column of
		  interest.
		- col (str): name of the column.
	Nothing, only modifies the data frame.
	'''
	df[col] = df[col].apply(lambda x: float(x))


def variable_df(variable_list, year, save=True):
	'''
	Makes a request in ACS5 API and creates a Pandas data frame from the
	response.
	Inputs:
		- variable_list(list): list with names of variables to get. 
		- year (str): year to consider. 
		- save (boolean): if True, save the resulting data frame as csv file
	'''
	myurl = make_url_req(variable_list, year)
	raw_data = get_json(myurl)
	df = pd.DataFrame(raw_data[1:], columns=raw_data[0])
	if save:
		df.to_csv('clean_data/acs_'+year+'.csv', index=False)
	return df


def variable_by_tract(variable_name):
	'''
	Merge two geo data frames by census tract. One of the data frames is assumed to containg
	unique recodrs, while the other not necessarily.
	Inputs:
		- variable_name (str): name of the variable to consider.
	Return a data frame
	'''
	#tracts = gpd.read_file('raw_data/geo_export_fe9f2155-ba22-4697-91ff-daeee48c8d0b.shp')
	variable_tracs = gpd.read_file('clean_data/geo_{}_tracts.shp'.format(variable_name))
	variable = gpd.read_file('clean_data/geo_{}.shp'.format(variable_name))

	if len(variable_tracs) == len(variable):
		df = variable_tracs[['id','year', 'tract']]

	else:
		df = variable.merge(variable_tracs, how='left', on='unique')
		df = df[['id','year', 'tract']]

	df = df.groupby(by=['year', 'tract']).count()[['id']].reset_index()
	df.columns = ['year', 'tract', variable_name]
	return df

def merge_several_df(df_list):
	'''
	Merge a list of data frames.
	Inputs:
		- df_list (list): List containing different data frames.
	Returns a data frame.
	'''
	df0 = df_list[0]
	for df in df_list[1:]:
		df0 = df0.merge(df, how='outer', on=['tract', 'year'])
	return df0
