'''
Program that identifies which points are located within which polygon, considering a geo data frame 
with geometry point and another with polygons.

Jesus I. Ramirez Franco
'''
import multiprocessing as mp 
import pandas as pd 
import geopandas as gpd 
from shapely.geometry import Point
import sys

tracts = gpd.read_file('raw_data/geo_export_fe9f2155-ba22-4697-91ff-daeee48c8d0b.shp')

if len(sys.argv) > 1:
	file_name = 'clean_data/'+ str(sys.argv[1])
	try:
		geo_points = gpd.read_file(file_name)
	except:
		print('Please provide a shp file')
else:
	print('you must provide the name of a shape file in the second position')


def find_tract(point):
	'''
	Identifies the tract where the point is located
	'''
	count = 0
	for i, polygon in tracts.iterrows():
		if point[1].within(polygon.geometry):
			return (point[0], polygon['tractce10'])

'''
The following lines create a list of tuples with an identifier in the position 0 
and a geometry point in position 1.
'''
inputs = {}
for i, r in geo_points.iterrows():
	if r['unique'] not in inputs.keys():
		inputs[r['unique']] = r['geometry']

inputs = [[k,v] for k, v in inputs.items()]

# Parallel programing part
if __name__ == '__main__':
    p = mp.Pool(processes=3)
    r = p.map_async(find_tract, inputs)

    p.close()
    p.join()

    results = r.get()
    results = [res for res in results if res]
    results_df = pd.DataFrame(results, columns = ['unique', 'tract'])

    geo_points = geo_points.merge(results_df, how='left', on='unique')
    geo_points.to_file(file_name[:-4]+"_tracts.shp")
    print('a new shape file was created and stored in clean_data folder')
