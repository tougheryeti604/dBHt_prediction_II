'''
WHAT ARE WE DOING HERE ?

Functionally identical to MKII, but we are extracting dBHt values only.
We shall remove lines of code that calculate whether dBHt exceeds threshold

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


def extract_storm_time(file, configData): 
    storm_time_list = []

    # Time is in hours
    lead = configData['lead']
    recovery = configData['recovery']

    # Extracts out list of all storm times 
    storm_times = pd.read_csv(file, header=None, names=['dates'])['dates'].values
    
    for time in storm_times[:]: 
         stime = datetime.strptime(time, '%Y-%m-%d %H:%M:%S') -pd.Timedelta(hours=lead) # strptime method allows us to add / subtract from values in YYYY-MM-DD HH-MM-SS format
         etime = datetime.strptime(time, '%Y-%m-%d %H:%M:%S') +pd.Timedelta(hours=recovery)

         period = [stime, etime]
         storm_time_list.append(period) # Each element in the list are the start+end times of 1 storm
    
    # Remove storm times that overlap with test storm times 
    all_test_s = configData['test_storm_stime']
    all_test_e = configData['test_storm_etime']

    for test_s, test_e in zip(all_test_s, all_test_e): # Iterates through all our test storms
        test_s = datetime.strptime(test_s, '%Y-%m-%d %H:%M:%S')
        test_e = datetime.strptime(test_e, '%Y-%m-%d %H:%M:%S')

        for indiv in storm_time_list: # Iterates through the start and end time of each CSVlist storm
            start = indiv[0]
            end = indiv[1]

            if start > test_e: # Storm happens completely after test storm
                storm_time_list = storm_time_list 

            elif end < test_s: # Storm happens completely before test storm
                storm_time_list = storm_time_list

            else:
                print("Removing an overlapping storm...")
                storm_time_list.remove(indiv) # Remove the overlapping storm entirely

    stormList = pd.DataFrame(storm_time_list, columns = ['start', 'end']) # Converts our list of lists into a PD DataFrame (Each list in our list of storms contains s and e times)
    print('Number of storms left: {0}'.format(len(stormList)))

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
    allData_sets = split_allData(allData, stormList) # allData sets contain the OMNI+ACE+INTERMAG data for each individual storm. Here, we have 156 storms

    # Get our label data for each storm (156 sets)
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

    return stormList

def split_allData(allData, stormList): 
    print("\nSplitting up allData ...")

    processed = []
    remaining_storms = zip(stormList['start'].values, stormList['end'].values) # Get list of tuples (s time, e time) again, to iterate through all remaining sets of data (156 storms) 

    for _ in remaining_storms: 
        stime = _[0] # get the start and end time of each storm
        etime = _[1]

        single = allData[stime:etime] # single is ACE + OMNI + INTERMAG data for 1 storm 
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

        for i in one_input.rolling('30min'): # i is each 30 minute window of input data 
            input_start = i.index[0]
            input_end = i.index[len(i)-1]
            label_start = i.index[0] + pd.Timedelta(minutes = th+ft)
            label_end = i.index[0] + pd.Timedelta(minutes = th+ft+wt-1)

            l = one_label[label_start : label_end] # l will be the corresponding label data for i 

            i.index = list(range(len(i))) # Reindexing so that INPUT and LABEL can be properly concatenated 
            l.index = list(range(len(l)))

            pair = pd.concat([i, l], axis = 1, ignore_index=False) # pair contains corresponding INPUT and LABEL pairs

            count += 1

            if pd.isna(pair).values.any() == False: 
                data_sets.append(pair) # List of all INPUT-LABEL pairs

            # drop INPUT-LABEL pairs with np.nan values

            else: 
                data_sets = data_sets

    print("\nThere are {0} Input-Label pairs in total. After removing datasets with np.na values...".format(count))

    print("\nWe have {0} Input-Label pairs !".format(len(data_sets)))

    return data_sets   

def split_saver(AllFiles, configData): 
    print("\nFinally, we are splitting our data sets into Test // Val, then saving them")
    print("\nNumber of datasets: {0}".format( len(AllFiles) ))

    val = []
    train = []
    val_dir = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/val/'
    train_dir = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/train/'

    number_of_sets = len(AllFiles)

    sss = ShuffleSplit(n_splits=1, train_size=0.2, random_state=configData['random_seed'])

    for val_idx, train_idx in sss.split( list( range(number_of_sets) ) ): 
        val_idx = val_idx # Get the list of validation data indexes from the split generator object
        train_idx = train_idx # Get the list of training data indexes from the split generator object
    
    for i in val_idx: 
        val.append(AllFiles[i]) # Appending individual INPUT-LABEL pairs as intended, CAA 1615hrs
    
    for j in train_idx:
        train.append(AllFiles[j]) # Appending individual INPUT-LABEL pairs as intended, CAA 1615hrs

    print("Number of training datasets: {0}".format( len(train)))
    print("Number of validation datasets: {0}".format( len(val)))

    print('\nsaving validation datasets ...')
    num = 0

    for dataset in val: 
        num += 1
        name = '{0}_PredictingFrom_'.format(num) + str(dataset['Label UTC'].values[0])[0:9]
        dataset.to_csv(val_dir+name+'_limit15_.csv', index = True)

    print('\nsaving training datasets ...')
    num = 0

    for dataset in train:
        num += 1
        name = '{0}_PredictingFrom_'.format(num) + str(dataset['Label UTC'].values[0])[0:9]
        dataset.to_csv(train_dir+name+'_limit15_.csv', index = True)

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

    print("\nAfter constraning data to INTERMAG availability, we have {0} storms left".format(len(Input)))

    '''
    Input and Label are lists of Pandas Dataframes. Each element is one data set (1 for each storm). We currently have 156.
    Now we need to match our input data to our label data
    For time start s, time history h, forecast f, window w
    --> Our input data will be from (s) -> (s+h)
    --> Our label data for input data (s) -> (s+h) will be from (s+h+f) to (s+h+f+w)
    '''

    AllFiles = rolling_block(Input, Label, configData) # Matches INPUT data to LABEL data, according to provided time history, forecast and window values

    print(AllFiles[:1])

    # Splits all our data into train // val and saves them in a file
    split_saver(AllFiles, configData)
    
    print('\nData preparation done !!', '\n', 'GAN intermag data only, dmin only, no test data, nan values ignored')


    return


if __name__ == '__main__':

	main() 

    # Marker to keep track of when I last edited this: 050525  ~




