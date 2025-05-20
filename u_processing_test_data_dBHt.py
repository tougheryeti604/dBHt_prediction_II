'''
WHAT ARE WE DOING HERE ?

1. Extract out all storm times, then add lead + recovery times for TEST STORMS
2. Extract out OMNI + ACE + Supermag for TEST STORM times (based on stormlist.csv) 
3. Split our data into INPUT-LABEL pairs, but concatenated over an entire storm period
4. LABEL data contains dBHt values only

###@ This is a marker that this file is definitive CAA 08042025 0931hrs

'''

import json
import os
import pickle
import random
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from itertools import dropwhile


def extract_storm_time(file, configData): 
    print("Extracting test storms now ...")
    test_storm_list = []

    s_time = configData['test_storm_stime']
    e_time = configData['test_storm_etime']
    
    test_storms = zip(s_time, e_time)  

    for s_time, e_time in test_storms: 
        s_time = datetime.strptime(s_time, '%Y-%m-%d %H:%M:%S')
        e_time = datetime.strptime(e_time, '%Y-%m-%d %H:%M:%S')
        test_storm_list.append([s_time, e_time])
        
    stormList = pd.DataFrame(test_storm_list, columns = ['start', 'end']) # Converts our list of lists into a PD DataFrame (Each list in our list of storms contains s and e times)
    
    print('Number of test storms: {0}'.format(len(stormList)))

    return stormList

def extract_datasets(stormList, config, ACE_OMNI, INTERMAG):
    print("\nNow preparing model input data ...")

    sets = []

    intermag = pd.read_feather(INTERMAG)
    ACEnOMNI = pd.read_feather(ACE_OMNI)
    
    # Doing this as our ground based magnetic field data is limited and will constrain our datasets. 
    start = intermag.index.values[0] # time + date of first available intermag data
    end = intermag.index.values[len(intermag.index.values)-1] # time + date of last available intermag data

    ACEnOMNI = ACEnOMNI[start:end] # Constrain data from ACEnOMNI for available intermag dates

    allData = pd.concat([ACEnOMNI, intermag], axis = 1, ignore_index=False) # concats ACEnOMNI data with intermag data
    allData = allData.drop('DayNum', axis = 1) # Final Pandas DataFrame with all REQUIRED input data
    allData['dBHt'] = np.sqrt(((allData['X'].diff(1))**2)+((allData['Y'].diff(1))**2)) # calculates dBHt values for each storm

    # Constrain storm times for available intermag dates
    stormList = constrain_stormList(stormList, start, end)

    # Now we split all of our data up into individual storm times 
    allData_sets = split_allData(allData, stormList) # allData sets contain the OMNI+ACE+INTERMAG data for each individual test storm. 

    # Get our label data for each storm 
    allLabel_sets = get_label(allData_sets)

    return allData_sets, allLabel_sets

def constrain_stormList(stormList, start, end):
    print("\nConstraining our storm list to fit INTERMAG availability ...")

    idx = 0
    all_storms = zip(stormList['start'].values, stormList['end'].values) # tuple of all start / end storm-time pairs (as extracted from stormlist.csv)

    for _ in all_storms:
        stime = _[0]  
        etime = _[1]

        if etime < start: 
            stormList = stormList.drop(idx) # Drops storm that is out of the constraints

        elif stime > end: 
            stormList = stormList.drop(idx) # Drops storm that is out of the constraints
        
        else:
            stormList = stormList

        idx += 1 # 
    
    # reindexes stormList so that indexes are in numerical order
    new_index = list(range(0, len(stormList)))
    stormList.index = new_index 

    print("\nAfter constraning data to INTERMAG availability, we have {0} storms left".format(len(stormList)))

    return stormList

def split_allData(allData, stormList): 
    print("\nSplitting up allData ...")

    processed = []
    remaining_storms = zip(stormList['start'].values, stormList['end'].values) # Get list of tuples (s time, e time) again, to iterate through all remaining sets of test storms 

    for _ in remaining_storms: 
        stime = _[0] # get the start and end time of each test storm
        etime = _[1]

        single = allData[stime:etime] # single is ACE + OMNI + INTERMAG data for 1 test storm 
        processed.append(single)
        
    return processed

def get_label(allData_sets):
    print("\nNow getting target labels ...")

    label = []

    for set in allData_sets: # set is all data for 1 storm
        # temporary DF for our label data 
        tempdf = pd.DataFrame(set['dBHt'].values, columns = ['Label dBHt'])
        tempidx = pd.date_range(start = set.index[0], end = set.index[len(set)-1], freq = '1 Min') # set the index of the temp label DF to that of the corresponding storm
        tempdf.index = tempidx

        UTC = pd.date_range(start = tempdf.index[0], end = tempdf.index[len(tempdf)-1], freq='1Min') # Add UTC column to our label data
        tempdf['Label UTC'] = UTC

        label.append(tempdf)

    return label   

def rolling_block(Input, Label, configData): 
    print('\n Getting Input=Label pairs...')

    th = configData['time_history']
    ft = configData['forecast']
    wt = configData['window']

    idx = len(Input)

    data_sets = []
    no_na_data_sets = []
    count = 0 # This is a marker, that this version is definitive, CAA 290425, 1151Hrs

    for _ in range(idx): # For each storm ... // [:] is placed to reduce debugging time, remove later
        print("Getting for storm number {0}".format(_+1))
        one_input = pd.DataFrame(Input[_]) # one_input is the input data for each storm
        one_label = pd.DataFrame(Label[_]) # one_label is the label data for each storm
        end_time = one_input.index[len(one_input)-1] # the end time of each storm
        start_time = one_label.index[0] # the start time of each storm

        '''
        If our last label data ends at t = t 
        Then, our last input data must end at t-wt-ft

        If our input data starts at T = T
        our label data must start from T = T+th+ft, and end at T+th+ft+wt
        '''

        input_end_time = end_time - pd.Timedelta(minutes=ft + wt) # Gets the timing of the ENDING input data for each storm
        label_start_time = start_time + pd.Timedelta(minutes = th+ft) # Gets the timing of the STARTING label data for each storm

        one_label = one_label[label_start_time:] # our label data must start from T = T+th+ft
        one_input = one_input[:input_end_time] # Reducing one_input to the input_end_time as we want the rolling window for our inputs to not go past t-wt-ft

        windows = dropwhile(lambda windows: len(windows) < 30, one_input.rolling('30min', min_periods = 30)) # Drops windows that are not of size 30
        
        ### Above works correctly CAA 070525

        for i in windows: # i is each 30 minute window of input data 
            label_start = i.index[0] + pd.Timedelta(minutes = th+ft)
            label_end = i.index[0] + pd.Timedelta(minutes = th+ft+wt-1)

            l = one_label[label_start : label_end] # l will be the corresponding label data for i 

            i.index = list(range(len(i))) # Reindexing so that INPUT and LABEL can be properly concatenated 
            l.index = list(range(len(l)))

            pair = pd.concat([i, l], axis = 1, ignore_index=False) # pair contains corresponding INPUT and LABEL pairs

            count += 1

            data_sets.append(pair) # Not dropping any INPUT-LABEL pairs for test storms

            if pd.isna(pair).values.any() == False: 
                no_na_data_sets.append(pair) # List of all INPUT-LABEL pairs

            # drop INPUT-LABEL pairs with np.nan values

            else: 
                no_na_data_sets = no_na_data_sets

    print("\nThere are {0} Input-Label pairs in total. After removing pairs with np.na values...".format(count))

    print("\nWe have {0} Input-Label pairs !".format(len(no_na_data_sets)))

    return data_sets, no_na_data_sets

def test_saver(AllFiles, AllFilesNoNA, configData): 
    print("\nNumber of test datasets: {0}".format( len(AllFiles) ))

    test = []
    test_dir_I = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/all_test/'
    test_dir_II = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/filtered_test/'
    num = 0

    print('\nsaving test datasets ...')

    for data_set in AllFiles: 
        num += 1
        name = ' {0} PredictingFrom_'.format(num) + str(data_set['Label UTC'].values[0])[0:9]
        data_set.to_csv(test_dir_I + name + '_limit15.csv')

    num = 0

    for no_nan_data_set in AllFilesNoNA: 
        num += 1
        name = ' {0} PredictingFrom_'.format(num) + str(data_set['Label UTC'].values[0])[0:9]
        no_nan_data_set.to_csv(test_dir_II + name + '_limit15.csv')

    return 

def main():
    stormFile = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/stormList.csv'
    config = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/config.json'

    ACE_OMNI = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/ACE_OMNI_processed/ace_and_omni_15_interp.feather'
    INTERMAG = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/MAG/gan_processed/gan_dmin_processed_15.feather'

    # Extract out for config data 
    with open(config, 'r') as f: 
        configData = json.load(f)

    # Extracts out storm times from CSV file
    stormList = extract_storm_time(stormFile, configData)

    # Extract out OMNI + ACE + INTERMAG data and LABEL data for each individual storm period 
    Input, Label = extract_datasets(stormList, configData, ACE_OMNI, INTERMAG)

    '''
    Input and Label are lists of Pandas Dataframes. Each element is one data set (1 for each test storm).
    Now we need to match our input data to our label data
    For time start s, time history h, forecast f, window w
    --> Our input data will be from (s) -> (s+h)
    --> Our label data for input data (s) -> (s+h) will be from (s+h+f) to (s+h+f+w)
    '''

    # Matches INPUT data to LABEL data, according to provided time history, forecast and window values
    # Returns two lists: one list with all INPUT-LABEL pairs for our test storm, the other list with INPUT-LABEL pairs that have np.nan values removed
    AllFiles, AllFilesNoNA = rolling_block(Input, Label, configData) 

    # Save all our test files in order, no splitting of data
    test_saver(AllFiles, AllFilesNoNA, configData)
    
    print('\nTest data preparation done !!', '\n', 'GAN intermag data only, dmin only')

    return


if __name__ == '__main__':

	main() 




