import datetime as dt
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pyhdf.SD as SD

stations = ['gan']

'''
all stations are GAN, TTB, DLT

1. Load files in .min format
2. Extract out X and Y 
3. Calculate H, MLT 
4. Insert X, Y, H, MLT into a pandas DataFrame and save to feather

sq(H) = sq(X) + sq(Y)

###

print(reader.loc['2011-05-01 00:00:00'].X) --> how to access individual values within a DataFrame

'''

magFiles = glob.glob('C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/MAG/GAN/{0}*dmin.min'.format(stations[0]))
Concat_magFiles = []

for f in magFiles: # Works for processing vmin files from INTERMAGNET (provisional data files)
    reader = pd.read_csv(f, skiprows=22, sep='\s+')
    
    # The formatting of IMAG's definitive data files have abberant headers. We shall rectify them here: 
    reader = reader[3:len(reader)] # Gets of irrelevant first 3 rows of data 
    reader = reader.drop(['of', 'arc', 'to.1', 'equivalent', '|'], axis = 1) # Gets rid of irrelevant columns
    reader = reader.rename(columns = {'#':'Date', 'Declination' : 'Time', 'to' : 'DayNum', 'be' : 'X', 'converted' : 'Y', 'from' : 'Z', 'minutes' : 'F'}) # Renames columns
    reader.index = range(0, len(reader))

    # Sets Date & Time as new index // Inserts UTC as its own column
    UTC = reader.Date + ' ' + reader.Time
    new_idx = pd.date_range(start = UTC[0], end = UTC[len(UTC)-1], freq = '1 Min')
    reader.index = new_idx
    reader = reader = reader.drop(['Date', 'Time'], axis = 1) # Drops now irrelevant Date + Time columns
    reader['UTC'] = UTC.values

    # Set the Dtype of all columns to float64
    reader = reader.astype({'X' : 'float64', 'Y':'float64', 'Z':'float64', 'F':'float64'})

    Concat_magFiles.append(reader)
    print("{0} completed !".format(f))

# Concatenates all individual files into 1 DataFrame
magData = pd.concat(Concat_magFiles, axis=0, ignore_index=False)

# Changes all invalid values to np.nan
magData.loc[magData['X'] >= 99999, 'X'] = np.nan
magData.loc[magData['Y'] >= 99999, 'Y'] = np.nan
magData.loc[magData['Z'] >= 99999, 'Y'] = np.nan
magData.loc[magData['F'] >= 99999, 'Y'] = np.nan

# Linearly interpolates data that are missing
method = 'linear'
limit = 15
magData['X'] = magData.X.interpolate(method=method, limit = limit)
magData['Y'] = magData.Y.interpolate(method=method, limit = limit)

# Calculates and inserts H into magData DataFrame 
H = np.sqrt(np.square(magData['X'])+np.square(magData['Y']))
magData.insert(5, "H", H)

print(magData)

# Save data as feather
station = 'gan' 
DataDump = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/MAG/{0}_processed'.format(station)

if not os.path.exists(DataDump):
    os.makedirs(DataDump)

magData.to_feather(DataDump + '/{0}_dmin_processed_{1}.feather'.format(station, limit))

