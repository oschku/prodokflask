# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Created on Mon Sep 21 15:40:40 2020

@author: Oskari Honkasalo


"""



import pickle 
import requests
import geopandas as gpd
import json
import pandas as pd
from sklearn.neighbors import BallTree
import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Point
from datetime import date, timedelta 
from math import sin, cos, sqrt, atan2, radians
import joblib as joblib
from babel.numbers import format_currency
import sys
# import tensorflow as tf, use this if/when we use tensorflows models. So far XGBoost has much better performance
import xgboost as xgb
import os
from dotenv import load_dotenv
from pathlib import Path
from . import tm35
from dotenv import load_dotenv



p = Path(__file__).parents[2]

dotenv_path = os.path.join(p, '.env')
load_dotenv(dotenv_path)

env_dir = str(os.environ['WORK_DIR'])

if env_dir == 'docker':
    wdir  = '/app/project'
elif env_dir == 'local':
    wdir = str(p)

pickle_files = ['output', 'housing_types', 'postnumbers'] # All files that use the pickle format
DATA_DIR = wdir + "/data"
MODEL_DIR = wdir + "/models"
COL_PATH =  DATA_DIR + '/input/input_cols.csv'  # Path for the model training input
 


files = []
city_list = ['Helsinki', 'Espoo', 'Vantaa']
kuntoluokat = {'Uusi':2, 'Erinomainen':1, 'Hyvä':0, 'Tyydyttävä':-1, 'Huono':-2} # Ordinal variable: the building's condition (from new to bad)
tonttiluokat = ['Oma', 'Vuokra', 'Valinnainen_vuokratontti'] # Categorical variabe: land ownership (own vs lease)
HubName = {2:'metro', 3:'raitiovaunu', 4:'juna'} # Hub type in closest stop
travel_time_names = ['travel_times_to_5975375_walk_t', 'travel_times_to_5975375_pt_r_tt', 'travel_times_to_5975375_car_r_t', 
                     'travel_times_to_5975375_walk_d'] # travel times by medium to grid ykr-grid number 5975375, which is located in the city centre
hsy_columns = ['raklkm_as', 'raklkm_muu', 'aluetehok'] # column names for the data by Helsinki Region Environmental Services Authoroity 

latest_checkpoint = "cp-0195.ckpt" # Latest NN Model checkpoint used in tensorflow

ykr_file_location = DATA_DIR + '/geojson/ykr_grids.geojson' # Location for the shapefile containing the ykr-grids and their id's
travel_time_location = DATA_DIR + '/time_to_5975375.txt' # text file location for the travel times to the city center ykr-grid 
shp_location = DATA_DIR + '/shp/' # Location for all shapefiles used in the script
geojson_location = DATA_DIR + '/geojson/' # Location for geojson files used in the script
xgb_file_name = MODEL_DIR + '/xgb_model.pkl'



def get_data(UserInput, query_id):
    
    user_input = UserInput.query.filter_by(query_id = query_id).all()

    output_dict = {}

    

    for row in user_input:
        output_dict.update(row.__dict__)

    print(output_dict)
    
    hoitovastike_per_nelio = output_dict.get('vastike')/output_dict.get('asuinala')

    rak_ika = 2021 - output_dict.get('rakennusvuosi')

    if rak_ika <= 2:
        uudiskohde = 1
    else:
        uudiskohde = 0
    
    output_dict.update({'uudiskohde':uudiskohde})
    output_dict.update({'hoitovastike_per_nelio': hoitovastike_per_nelio})
    output_dict.update({'rak_ika':rak_ika})

    print(output_dict)


    return output_dict



class ui_input():
    def get_pickle(filedir, filename):
        '''
        Retreives the user's inputs from the ui application 
        '''
        infile = open(filedir + "/" + filename + ".pickle", 'rb')
        new_file = pickle.load(infile)
        return new_file
    
        
    
    def fetch_data(user_data):
        '''
        Retreives coordinate data from the user input's address. With the coordinate
        data all the geographical data is retreived from various sources.
        Returns a dictionary of all the necessary data used in the model
        '''

        osoite = user_data.get('osoite')
        kunta = user_data.get('kunta')
        location_string = osoite + ', ' + kunta 
        lat, lng = geodata.geocode(location_string, kunta)
        
        shore_dist = geodata.distance_to_shoreline(location_string, lat, lng)
        hubdist, hub_type = geodata.hub_distances_hsl(lat, lng)
        travel_times = geodata.travel_times_hsl(lat, lng)
        hsy_data = geodata.building_grid_data(lat, lng)
        cpi = geodata.get_cpi()
        
        
        nn_input_data = {'ui_data': user_data, 'dist_shore': shore_dist, 'dist_metro_train': hubdist, 'hub_type': hub_type,
                      'travel_time': travel_times, 'hsy_grid_data': hsy_data, 'ek_indeksi': cpi}
        
        return nn_input_data
    
    
    
    def get_app_data(DATA_DIR, UserInput, query_id):
        '''
        Input: DATA_DIR = directory string for ui inputs save location
        Runs the UI application and saves the user inputs in the DATA_DIR location as a pickle file
        '''
        
        user_data = get_data(UserInput, query_id)
        print(user_data)

        
        for f in pickle_files:
            files.append(ui_input.get_pickle(DATA_DIR, f))
        
        nn_input_data = ui_input.fetch_data(user_data)
        
        return nn_input_data
    
    
    
    
    def convert_data(nn_input_data):
        '''
        Input: input data dictionary, that is returned from the function get_app_data
        Converts all the input data into an array that is used in the model inference.
        The converted data is saved as a pickle file, which the inference function reads in.
    
        '''    
        hub_number = nn_input_data.get('hub_type')
        housing_type = pd.get_dummies(pd.DataFrame(
            {
            'Kohdetyyppi':[nn_input_data['ui_data']['asuntotyyppi']]
            }), prefix_sep='.')
        postnumber = pd.get_dummies(pd.DataFrame(
            {
            'pn':[nn_input_data['ui_data']['postinumero'].lstrip("0")]
            }), prefix_sep='.')
        tonttiluokka = pd.get_dummies(pd.DataFrame(
            {
            'Tontin_omistajuus':[nn_input_data['ui_data']['tontti']]
            }), prefix_sep='.')
        hubtype = pd.get_dummies(pd.DataFrame(
            {
            'HubName':[HubName[hub_number]]
            }), prefix_sep='.')
        
        
        
        # Get pre-determined lists of dummy variables in pickle format:
        files_to_unpickle = ['housing_types', 'postnumbers', 'tontti']
        dummy_list = []
        for filename in files_to_unpickle:
            dummy_list.append(ui_input.get_pickle(DATA_DIR, filename))
        
        
        
        # Make dataframes for the dummy variables    
        dummies_frame = pd.get_dummies(dummy_list[0])
        housing_type = housing_type.reindex(columns = dummies_frame.columns, fill_value=0)
        
        dummies_frame = pd.get_dummies(dummy_list[1])
        postnumber = postnumber.reindex(columns = dummies_frame.columns, fill_value=0)
        
        dummies_frame = pd.get_dummies(dummy_list[2])
        tontti = tonttiluokka.reindex(columns = dummies_frame.columns, fill_value=0)
        
        hub_dummies_frame = pd.get_dummies(['HubName.juna', 'HubName.metro'])
        hub = hubtype.reindex(columns = hub_dummies_frame.columns, fill_value=0)
        
        
        
        
        # Data manipulations to transform the ui data into tabular and numerical form
        kunto = kuntoluokat.get(nn_input_data['ui_data']['kunto'])
        
        final_input = pd.DataFrame(housing_type)
        postnumber = pd.DataFrame(postnumber)
        travel_times = pd.DataFrame({travel_time_names[0]:nn_input_data['travel_time']['walk_t'],
                                     travel_time_names[1]:nn_input_data['travel_time']['pt_r_tt'],
                                     travel_time_names[2]:nn_input_data['travel_time']['car_r_t'],
                                     travel_time_names[3]:nn_input_data['travel_time']['walk_d']}).reset_index()
        
        
        
        hsy_df = nn_input_data['hsy_grid_data']
        hsy_df = hsy_df[hsy_df.columns.intersection(hsy_columns)]
        ui_data = nn_input_data.get('ui_data')
        ui_data['kunto'] = kunto
        ui_data = pd.Series(ui_data).to_frame().T
        ui_data = pd.DataFrame.from_records(ui_data)
        ui_data = ui_data.iloc[:,4:]
        print(ui_data)
        #ui_data.pop('tontti')
        try:
            ui_data.drop('created_on', inplace=True, axis=1)
        except KeyError:
            pass
        ui_data.drop('query_id', inplace=True, axis=1)
        
        
        
        
        keys = ['ek_indeksi', 'dist_metro_train', 'dist_shore']
        nn_input_single_units = {your_key: nn_input_data[your_key] for your_key in keys}
        nn_input_single_units = pd.Series(nn_input_single_units).to_frame().T
        nn_input_single_units = pd.DataFrame.from_records(nn_input_single_units)
        
        
        all_dataframes = pd.concat([final_input, postnumber, travel_times, ui_data, tontti, 
                                    hub, hsy_df, nn_input_single_units], axis=1) 
        
    
        with open(DATA_DIR + '/input/nn_inference_input.pickle', 'wb') as handle:
                    pickle.dump(all_dataframes, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
    
    
    
    def inference(COL_PATH, UserInput, query_id):

        # Read in the data (to reorder the columns correctly):
        
        dataset = pd.read_csv(COL_PATH, skipinitialspace=True, encoding = "utf-8", sep=","
                              )   
        dataset.columns = dataset.columns.str.replace('ä','a')
        dataset.columns = dataset.columns.str.replace('ö','o')
        train_dataset = dataset.iloc[:,:len(dataset.columns)-1]
        print(train_dataset)
              
        
        # Load model from checkpoint, load standard scaler and infeerence input   
        # latest = "C:/tf/checkpoints/" + latest_checkpoint
        # best_model=tf.keras.models.load_model(latest)
        norm = joblib.load(DATA_DIR + '/norm/norm.pkl')
        nn_inference_input = joblib.load(DATA_DIR + '/input/nn_inference_input.pickle')
        
        
        # re-arrange columns correctly
        orig_columns = list(train_dataset.columns) 
        nn_inference_input_reordered = nn_inference_input.reindex(columns=orig_columns)
        print(nn_inference_input_reordered)
        nn_inference_input_reordered.to_csv(DATA_DIR + '/nn_input.csv')
        
        nn_input = norm.transform(nn_inference_input_reordered)
        
        
        # Inference w/tensorflow
        # prediction = best_model.predict(nn_input)
        # size = nn_inference_input['Asuinala'][0]
        # prediction = prediction*size
        # print('neural network :' + str(prediction))
        
        # Inference with xgboost
        xgb_model_loaded = pickle.load(open(xgb_file_name, "rb"))
        dpred = xgb.DMatrix(nn_input)
        prediction = xgb_model_loaded.predict(dpred)
        try:
            size = nn_inference_input.iloc[0]['asuinala']
        except:
            u_input = UserInput.query.filter_by(query_id = query_id).all()
            output_dict2 = {}
            for row in u_input:
                output_dict2.update(row.__dict__)
                print(row)
            size = output_dict2.get('asuinala')
            
        prediction = prediction*size
        print('xgboost: ' + str(prediction))
        
        return prediction
    
    
    
    

    
    def count_price(UserInput, query_id):    
        nn_input_data = ui_input.get_app_data(DATA_DIR, UserInput, query_id)
        ui_input.convert_data(nn_input_data)
        price = ui_input.inference(COL_PATH, UserInput, query_id)
        return price





class geodata():
    '''
    This class contains the functions that retreive extra data from external sources 
    and all the geoprocessing functions
    '''

    def geocode(location, kunta):
        """ 
        A simple geocoding function using the HERE geocoding API Endpoint. 
        Takes a street name string as an input and returns its coordinates 
        
        """
        kunta_nro = 0
        mun_exists = False

        # Here API Endpoint
        URL = 'https://geocode.search.hereapi.com/v1/geocode'
        
        # Here API KEY
        apikey = 'woLpbeva3ce3NBuiEdD6GWPSsOMZVNgkKphzytQ5YtI'
          
        # Parameters for API request
        PARAMS = {'q':location, 'apiKey':apikey}
        
        r = requests.get(url = URL, params = PARAMS)
        data = r.json()
        
        for i in range(len(data['items'])):
            if kunta == data['items'][i]['address']['city']:
                print(data['items'][i]['address']['city'])
                kunta_nro = i
                mun_exists = True

        if len(data['items']) > 0:

            if data['items'][0]['address']['countryCode'] != 'FIN':
                raise ValueError('Incorrect street input/country', 'country')
            
            if mun_exists == False:
                raise ValueError('Street not found in municipality', 'city')
                print('exists' + kunta)
            

            if not 'streets' in data['items'][0]['scoring']['fieldScore'] or len(data['items'][0]['scoring']['fieldScore']['streets']) < 1:
                raise ValueError('No streets found', 'no_streets')
            elif not 'houseNumber' in data['items'][0]['scoring']['fieldScore']:
                if data['items'][0]['scoring']['queryScore'] < 0.8:
                    raise ValueError('Bad street query', 'bad_score')
                else:
                    raise ValueError('Street missing house number', 'multiple_streets')


            if mun_exists == True:
                try:
                    latitude = data['items'][kunta_nro]['position']['lat'] 
                    longitude = data['items'][kunta_nro]['position']['lng']
                    return latitude, longitude
                    print('not_exists' + kunta)
                except:
                    raise ValueError('Incorrect street input', 'street')
        
        else:
            raise ValueError('Incorrect input', 'street')
        
        
        
    
    
    
    def get_nearest(src_points, candidates, k_neighbors=1):  
        # get_nearest and nearest_neighbor functions sourced from the following site:
        # https://automating-gis-processes.github.io/site/notebooks/L3/nearest-neighbor-faster.html
        
        """Find nearest neighbors for all source points from a set of candidate points"""
    
        # Create tree from the candidate points
        tree = BallTree(candidates, leaf_size=15, metric='haversine')
    
        # Find closest points and distances
        distances, indices = tree.query(src_points, k=k_neighbors)
    
        # Transpose to get distances and indices into arrays
        distances = distances.transpose()
        indices = indices.transpose()
    
        # Get closest indices and distances (i.e. array at index 0)
        # note: for the second closest points, you would take index 1, etc.
        closest = indices[0]
        closest_dist = distances[0]
    
        # Return indices and distances
        return (closest, closest_dist)
    
    
    
    def nearest_neighbor(left_gdf, right_gdf, return_dist=False):
        """
        For each point in left_gdf, find closest point in right GeoDataFrame and return them.
    
        NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
        """
    
        left_geom_col = left_gdf.geometry.name
        right_geom_col = right_gdf.geometry.name
    
        # Ensure that index in right gdf is formed of sequential numbers
        right = right_gdf.copy().reset_index(drop=True)
    
        # Parse coordinates from points and insert them into a numpy array as RADIANS
        left_radians = np.array(left_gdf[left_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())
        right_radians = np.array(right[right_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())
    
        # Find the nearest points
        # -----------------------
        # closest ==> index in right_gdf that corresponds to the closest point
        # dist ==> distance between the nearest neighbors (in meters)
    
        closest, dist = geodata.get_nearest(src_points=left_radians, candidates=right_radians)
        closest_points_geom_col = right_gdf.geometry.name
    
        # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
        closest_points = right.loc[closest]
    
        # Ensure that the index corresponds the one in left_gdf
        closest_points = closest_points.reset_index(drop=True)
    
        # Add distance if requested
        if return_dist:
            # Convert to meters from radians
            R = 6371000  # Earth's radius in meters
    
            lat1 = radians(np.array(left_gdf[left_geom_col].apply(lambda geom: (geom.y)))[0])
            lon1 = radians(np.array(left_gdf[left_geom_col].apply(lambda geom: (geom.x)))[0])
            lat2 = radians(np.array(closest_points[closest_points_geom_col].apply(lambda geom: (geom.y)))[0])
            lon2 = radians(np.array(closest_points[closest_points_geom_col].apply(lambda geom: (geom.x)))[0])
            
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            
            a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            
            distance = R * c
            closest_points['distance'] = distance
    
        return closest_points
    
    
    
    def hub_distances_hsl(lat, lng):
        """   
        Parameters: lat, lng
        Return: distance, stop type
        This function retrieves the locations of HSL (Helsinki Regional Transport) stops and calculates the distance
        to the nearest stops (URL filtered to take rail and metro only). Returns distance and stop type
        
        """
        
        # create a geodataframe for the input parameter
        df_coords = pd.DataFrame({'Type':['input'], 'Lat':[lat], 'Lng':[lng]})
        gdf = gpd.GeoDataFrame(
            df_coords, geometry=gpd.points_from_xy(df_coords.Lng, df_coords.Lat))
        
    
        # request from HSL API Endpoint URL
        URL = 'https://opendata.arcgis.com/datasets/b2aa879ce93c4068ac63b64d71f24947_0.geojson?where=VERKKO%20%3E%3D%202%20AND%20VERKKO%20%3C%3D%204'
        r = requests.get(URL)
        data = r.json()
        
        # The json dictionary must be converted to a string and then back to geojson
        json_string = json.dumps(data)
        hsl_stops = gpd.read_file(json_string) 
        hsl_stops = hsl_stops[hsl_stops.VERKKO != 3] # Filter out tram stops (3)
        
        # Fetches the closest stop information
        closest_stop = geodata.nearest_neighbor(gdf, hsl_stops, return_dist=True)
        
        # filter out the distance and the stop type and return them
        dist = closest_stop.iloc[0,17]
        stop_type = closest_stop.iloc[0, 14]
        
        return dist, stop_type
    
    
    
    def travel_times_hsl(lat, lng):
        """
        Uses lat and lng from the input to fetch the travel times to the specified ykr-grid 
        (community structure grids 250x250m).
        The specified ykr-grid is determined by the parameter ykr_grid_location. 
        
        """
       
        # Get the ykr grid geojson file
        ykr_grids = gpd.read_file(ykr_file_location) 
    
        
        # create a GeoDataFrame for the input point parameter
        df_coords = pd.DataFrame({'Type':['input'], 'Lat':[lat], 'Lng':[lng]})
        gdf = gpd.GeoDataFrame(
            df_coords, geometry=gpd.points_from_xy(df_coords.Lng, df_coords.Lat))
        
        
        # Declare the two GeoDataFrames projection to EPSG:4326
        ykr_grids.crs = {'init':'epsg:4326'}
        gdf.crs = {'init':'epsg:4326'}
        
        # Spatial join, where the left table is the point coordinate of the input
        # and the right table is the ykr grid table. The join result gives the YKR_ID
        # where the input point lies within / intersects
        ykr_id = gpd.sjoin(gdf, ykr_grids, how='left', op='intersects')['YKR_ID'][0]
        
        # The YKR_ID is then used to retrieve the corresponding travel time from
        # the time travel matrix and a travel time list is returned
        travel_times_df = pd.read_csv(travel_time_location, sep=';')
        travel_time = travel_times_df[travel_times_df['from_id']==ykr_id]
            
        return travel_time
    
    
     
    def building_grid_data(lat, lng):
        """
        Uses the point coordinate input to get HSY (Helsinki Region Environmental Services Authoroity) data. 
        This data includes areal building efficiency. Returns a list of data
        
        """
    
        # create a GeoDataFrame for the input point parameter
        df_coords = pd.DataFrame({'Type':['input'], 'Lat':[lat], 'Lng':[lng]})
        gdf = gpd.GeoDataFrame(
            df_coords, geometry=gpd.points_from_xy(df_coords.Lng, df_coords.Lat), crs={'init': 'epsg:4326'})    
        
        hsy_grids = gpd.read_file(geojson_location + 'hsy_rak_ruudut.geojson', crs={'init': 'epsg:4326'})
        
        
        # Spatial join, where the left table is the point coordinate of the input
        # and the right table is the hsy table. The join result gives the hsy grid 
        # where the input point lies within / intersects
        hsy_data = gpd.sjoin(gdf, hsy_grids, how='left', op='intersects')
        
        return hsy_data
    
    
    
    def distance_to_shoreline(location, lat, lng):
        """
        Uses location as an address string input and queries the MML API Endpoint for
        TM35-FIN coordinates. The function reads a shoreline shapefile and measures
        the queried point's distance to the shoreline. Returns float min_dist  
        """
        try:
            URL = "https://avoin-paikkatieto.maanmittauslaitos.fi/geocoding/v1/pelias/search"
            
            # The parameters for the MML Geocoding REST API
            # NOTE: The current account is a free account and the API-KEY is required. For future reference,
            # a commercial account might necessary
            params = {'text':location, 'sources':'interpolated-road-addresses',
                    'outputFormat':'json', 'api-key':'b7f6c192-e61d-4ff8-9d0a-7f75f59e814f',
                    'crs':'EPSG:3067','lang':'fi'}
            
            r = requests.get(URL, params=params)
            r = r.json()
            
            x_coord = r['features'][0]['geometry']['coordinates'][0]
            y_coord = r['features'][0]['geometry']['coordinates'][1]

        except:
            print('Exception error, used tm35')
            x_coord, y_coord = tm35.latlon_to_xy(lat, lng)

        
        print(x_coord, y_coord)

        point = Point(x_coord, y_coord)   
        
        # Read in the shoreline shapefile and convert to shapely MultiPolygon
        sea_area = gpd.read_file(shp_location + "merialue_tm35.shp")
        polyg = sea_area['geometry']
        poly = MultiPolygon(polyg.all())
        
        # Measure the distance to the shoreline
        min_dist = 30000
        for polygon in poly:
            dist = polygon.exterior.distance(point)
            min_dist = min(min_dist, dist)
    
        return min_dist
    
    
    def get_cpi(optional_arg = 'date_time'):
        '''
        This function retreives the most recent CPI index from Statistics Finland.
        If a specific date is given as the 'date_time' input, then the CPI for that month.
    
        '''
        URL = 'http://pxnet2.stat.fi/PXWeb/api/v1/fi/StatFin/hin/khi/kk/statfin_khi_pxt_11xl.px'
        
        if optional_arg == 'date_time':
            today = date.today() - timedelta(45)
            year = today.strftime("%Y")
            month = today.strftime("%m")
        else:
            year = optional_arg.strftime("%Y")
            month = optional_arg.strftime("%m")
        
        time_stamp = year + "M" + month
        params = '''{
          "query": [
            {
              "code": "Kuukausi",
              "selection": {
                "filter": "item",
                "values": [    
                  ''' + '''"''' + time_stamp +  '''"''' + '''
                ]
              }
            }
          ],
          "response": {
            "format": "json-stat"
          }
        }'''
        
        r = requests.post(URL, data=params)
        r = r.json()
        cpi_value = r['dataset']['value'][0]
        
        return cpi_value







def calculate(UserInput, query_id):
    price = ui_input.count_price(UserInput, query_id)
    print(price)
    return price



