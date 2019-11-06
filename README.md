# Building abandonment in the City of Chicago
In this repository, you will find all the code and results of the home abandonment project in Chicago.

## Objective
Identify the most critical predictors of building abandonment in the City of Chicago at the census tract level using machine learning models and data from the American Community Survey and the Chicago Data Portal.

## Methods
The work is divided into four groups of activities:
1. Getting the data. 
2. Preprocessing the data.
3. Tuning the models.
4. Producing results.

### 1. Getting the data
The two primary sources of the data used in this exercise are the City of Chicago Data Portal (CCDP) and the American Community Survey (ACS5). Both sources have their API that allows us to get the data using Python code directly. The variables from the ACS5 are available at the census tract level every year, and most of them presented as percentage rates. On the other hand, the variables that come from the CCDP are georeferenced administrative records. The period considered for this exercise is 2012 - 2017.

Data from the ACS5 was obtained using a URL request per year. This task was performed in *acs_variables.ipynb* file using the function "variable_df" from the util.py file.

Data from the CCDP was obtained, making a Socrata request per variable. An example of this task could be found in *chi_dataportal_example.ipynb*. The primary function is "get_data" stored in util.py. This function requires the identifier of the dataset and a Socrata version of SQL queries to specify the years, the columns, and filters to obtain from the dataset.

The following are the variables considered from every source:

i.**ACS**

- 'S1101_C01_002E': Average household size
- 'S1401_C02_001E': Percent Population 3 years and over enrolled in school
- 'S1401_C02_030E': Percent Population 18 to 24 years!!Enrolled in college or graduate school
- 'S1501_C02_002E': Percent Population 18 to 24 years!!Less than high school graduate
- 'S1501_C02_008E': Percent Population 25 years and over!!9th to 12th grade, no diploma
- 'S1601_C02_003E': Percent Speak a language other than English
- 'S1701_C03_001E': Percent below poverty level
- 'S1810_C03_001E': Percent with a disability
- 'S1901_C01_012E': Median income
- 'S2201_C02_001E': Percent Households with FOOD STAMPS/Supplemental Nutrition Assistance Program (SNAP)
- 'S2301_C03_001E': Employment/Population Ratio/ Population 16 years and over
- 'S2701_C03_001E': Percent Insured

ii.**CCDP**

- 'buss_licences'
- 'building_violations'
- 'vehicle_theft'
- 'burglary'
- 'robbery'
- 'public_peace_violation'
- 'weapons_violation'
- 'sexual_assault'
- 'homicides'
- 'rodents'
- 'garbage'
- 'sanitation'
- 'abandon_vehicles'
- 'pot_holes'
- 'tree_trims'
- 'street_lights'

### 2. Preprocessing the data
The data from the CCDP is available as data points, and this work requires rates at the census tract level. In general, all the variables from the data portal was pre-processed following these steps:

1. Create a unique identifier.
2. Convert georeferences stored as strings to geo data points.
3. Identifies the census tract where every point is located.
4. Count and group the occurrences of points by census tract.
5. Convert absolute values to per 1000 inhabitants rates.

Most of the pre-processing task was performed in the *data_preparation.ipynb* file; however, point number 3 is performed using parallel processing (3 processes), considering that most of the datasets in CCDP contain hundreds of thousands of rows. The code for this task is stored in *mp_points_in_polygons.py*.

The data from the ACS5 requires almost no pre-processing. 

Once we have the data at the census tract level for the period 2012-2017, it was necessary to create time splits and standardize the data before training the models. The following table contains the distribution of data in the time splits:

<table class="tg">
  <tr>
    <th class="tg-yla0">Split<br></th>
    <th class="tg-yla0">x train period</th>
    <th class="tg-yla0">y train period</th>
    <th class="tg-yla0">x test period</th>
    <th class="tg-yla0">y test period</th>
  </tr>
  <tr>
    <td class="tg-yla0">1</td>
    <td class="tg-nrix">2012</td>
    <td class="tg-nrix">2014</td>
    <td class="tg-nrix">2013</td>
    <td class="tg-nrix">2015</td>
  </tr>
  <tr>
    <td class="tg-yla0">2</td>
    <td class="tg-nrix">2012-2013</td>
    <td class="tg-nrix">2014-2015</td>
    <td class="tg-nrix">2014</td>
    <td class="tg-nrix">2016</td>
  </tr>
  <tr>
    <td class="tg-yla0">3</td>
    <td class="tg-nrix">2012-2014</td>
    <td class="tg-nrix">2014-2016</td>
    <td class="tg-nrix">2015</td>
    <td class="tg-nrix">2017</td>
  </tr>
</table>

The dependent variable is the number of abandoned building per census tract, and required two necessary pre-processing steps: 
1. Divide the absolute number by the numbers of inhabitants/1000.
2. Then, convert the rate to a dummy variable. One if the rate is over 0.3 or 0 otherwise.

It is essential to mention that the dependent variable is two years after the independent variables because ACS5 data is published with almost one year lag. 

### 3. Machine learning models
In this study,  we used three different ML models: logistic regression, Random Forest, and Ada Boost classifier and different parameters for them. Initially, we considered also the SVC classifier; however, this model did not converge, and it is not included in the final version.

The code for this section could be found in the *models.ipynb* file, and the auxiliary functions in classifiers.py.

We trained every model using the three splits created and computed the following metrics: Accuracy, Precision, Recall, AUC, and F1. Then, we calculate the average of the metrics of the three splits. Considering the nature of the building abandonment problem, in this case, we used the AUC metric to choose the best model. The following table presents the top 5 models with the best performance:
<table class="tg">
  <tr>
    <th class="tg-uzvj">Model</th>
    <th class="tg-uzvj">Accuracy</th>
    <th class="tg-uzvj">Precision</th>
    <th class="tg-uzvj">Recall</th>
    <th class="tg-uzvj">AUC</th>
    <th class="tg-7btt">F1</th>
  </tr>
  <tr>
    <td class="tg-lboi">RandomForestClassifier'.- criterion: entropy, n_estimators: 200, random_state: 42</td>
    <td class="tg-lboi">0.780702</td>
    <td class="tg-lboi">0.875469</td>
    <td class="tg-lboi">0.777385</td>
    <td class="tg-lboi">0.784653</td>
    <td class="tg-0pky">0.820988</td>
  </tr>
  <tr>
    <td class="tg-lboi">RandomForestClassifier'.- criterion: gini, n_estimators: 200, random_state: 42</td>
    <td class="tg-lboi">0.780284</td>
    <td class="tg-lboi">0.865107</td>
    <td class="tg-lboi">0.782496</td>
    <td class="tg-lboi">0.782607</td>
    <td class="tg-0pky">0.818897</td>
  </tr>
  <tr>
    <td class="tg-lboi">RandomForestClassifier'.- criterion: entropy, n_estimators: 150, random_state: 42</td>
    <td class="tg-lboi">0.778613</td>
    <td class="tg-lboi">0.872750</td>
    <td class="tg-lboi">0.776311</td>
    <td class="tg-lboi">0.782269</td>
    <td class="tg-0pky">0.819150</td>
  </tr>
  <tr>
    <td class="tg-0pky">LogisticRegression'.- C: 0.1, max_iter: 200, penalty: l2, random_state: 42, solver: lbfgs</td>
    <td class="tg-0pky">0.776942</td>
    <td class="tg-0pky">0.885279</td>
    <td class="tg-0pky">0.765949</td>
    <td class="tg-0pky">0.781647</td>
    <td class="tg-0pky">0.819972</td>
  </tr>
  <tr>
    <td class="tg-0pky">LogisticRegression'.- C: 0.1, max_iter: 100, penalty: l2, random_state: 42, solver: lbfgs</td>
    <td class="tg-0pky">0.776942</td>
    <td class="tg-0pky">0.885279</td>
    <td class="tg-0pky">0.765949</td>
    <td class="tg-0pky">0.781647</td>
    <td class="tg-0pky">0.819972</td>
  </tr>
</table>

After choosing the best model, we used it to identify the most critical features and the probability of every census tract to have more than 0.3 abandoned buildings per 1000 inhabitants using fresh data of 2017 to get predictions for 2019.

### 4. Presenting results
With the prediction probabilities by tract, it is possible to create a map of risk of building abandonment for the City of Chicago. The QGIS project could be found in maps/pred_maps.qgz.
![Predictions 2019](maps/pred_map_2019_tuned.png)

## Description of files and folders
### Files
- acs_variables.ipynb: Jupyter notebook with the requests of ACS5 variables for the period 2012 - 2017.
- acs_variables.json: List of ACS5 variables considered.
- chi_dataportal_example.ipynb: Jupyter notebook with a couple of examples of requests of CCDP variables.
- classifiers.py: Python file with functions to train and tune ML models, and present results.
- crime.ipynb: Jupyter notebook with the requests of the different types of crimes from the Chicago Data Portal. 
- data_preparation.ipynb: Jupyter notebook where the datasets are prepared.
- dataportal.json: List of CCDP variables considered.
- models.ipynb: Jupyter notebook where the splits were created, and the models trained and tuned, and used to produce predictions.
- mp_points_in_polygon.py: Multiprocessing program to find the polygons within a group of points is located.
- util.py: Python file with auxiliary functions to make requests, clean, and pre-process data.
### Folders

The folders in the public version of this repository do not contain all the files used in the project due to size issues. 
- clean_data: Data used to train and tune the ML models. The local version of the repository contains intermediate files used during the pre-processing and cleaning part.
- maps: 
    - pred_maps.qgz: QGIS project to create maps with results.
    - pred_map_2019_upper.png: Map with 10% of the census tracts with the highest probability of building abandonment.
    - pred_map_2019_tuned.png: Map with results.
    - abandoned_build_year.png: Map with the available building abandonment reports from 2001 to date.
- raw_data: Data obtained from the API requests. In this case, only one example is included.

