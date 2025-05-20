import glob
import os

import numpy as np
import pandas as pd

from pyhdf.HDF import *
from pyhdf.VS import *

os.environ["CDF_LIB"] = "~/lib"

import cdflib

# defining reletive file paths to data 
omni_dir = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/OMNI/' # File directory of omni data 
plasmaDir = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/ACE/' # File directory of ace 'swepam' data 
magDir = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/ACE/' # File directory of ace 'mag' data 
dataDump = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/ACE_OMNI_processed/' # File directory to solve processed data 

if not os.path.exists(dataDump):
    os.makedirs(dataDump)

def ACE_DATA_toPd(): 
    plasmaPath = glob.glob(plasmaDir + 'swepam_data_64sec_year*')
    magPath = glob.glob(magDir + 'mag_data_16sec_year*')

    plasmaData = []
    magData = []
    
    print("Extracting SW plasma data now ...")
    for path in plasmaPath: ### [:1] to speed up processing for debugging purposes
        
        plasmaDF = HDF_to_DF(path) # Extracts HDF data into a Pandas DataFrame

        '''
        We want the following data: 
        Vx, Vy, Vz, density and temperature >> (use everything in the GSM coordinate system)
          Vx = 'proton_speed'
          Vy = 'y_dot_GSM'
          Vz = 'z_dot_GSM'
          density = 'proton_density'
          temperature = 'proton_temp'

        GSM coordinate system: The origin is defined at the center of the Earth, and is positive towards the Sun
        '''

        # Drop unwanted columns from the DataFrame
        to_drop = ['fp_year', 'fp_doy', 'He4toprotons', 
                   'x_dot_GSE', 'y_dot_GSE', 'z_dot_GSE', 
                   'x_dot_RTN', 'y_dot_RTN', 'z_dot_RTN', 
                   'pos_gse_x', 'pos_gse_y', 'pos_gse_z',
                   'pos_gsm_x', 'pos_gsm_y', 'pos_gsm_z',
                   'year', 'day', 'hr', 'min', 'sec']

        plasmaDF = plasmaDF.drop(to_drop, axis = 1)

        # adds a Date Time column, sets it as DataFrame index
        plasmaDF.index = pd.to_datetime(plasmaDF['ACEepoch'], unit = 's', origin = pd.Timestamp('1996-01-01 00:00:00'))
        plasmaDF.index.name = 'Date_UTC'
        
        # Converts invalid values to np.nan
        plasmaDF = plasmaDF_nan_filter(plasmaDF)

        # Appends individual DataFrames of each year, into a list 
        plasmaData.append(plasmaDF)

        print('{0} completed !'.format(path))

    print("Extracting mag data now ...")
    for path in magPath: ### [:1] to speed up processing for debugging purposes
        magDF = HDF_to_DF(path)

        '''
        We want the following data: 
        - By, Bz, Btotal (All in GSM)
        '''
        to_drop = ['fp_year', 'fp_doy', 'SCclock', 
                   'Br', 'Bn', 'Bmag', 'Delta', 'Lambda', 
                   'Bgse_y', 'Bgse_z', 'year', 'day', 'hr','min', 'sec',
                   'dBrms', 'sigma_B','fraction_good', 'N_vectors', 'Quality', 
                   'pos_gse_x', 'pos_gse_y','pos_gse_z', 
                   'pos_gsm_x', 'pos_gsm_y', 'pos_gsm_z']
        
        magDF = magDF.drop(to_drop, axis = 1)

        # adds a Date Time column, sets it as DataFrame index
        magDF.index = pd.to_datetime(magDF['ACEepoch'], unit = 's', origin = pd.Timestamp('1996-01-01 00:00:00'))
        magDF.index.name = 'Date_UTC'

        # Converts invalid values to np.nan
        magDF = magDF_nan_filter(magDF)
    
        # Appends individual DataFrames of each year, into a list 
        magData.append(magDF)

        print('{0} completed !'.format(path))

    plasmaData = pd.concat(plasmaData, axis = 0, ignore_index=False)
    magData = pd.concat(magData, axis = 0, ignore_index=False)

    '''
    The following function
    resamples the ACE data to 1 minute resolution, interploates to the defined limit, 
    and converting column names so they match the column names from the OMNI database.
    '''
    allACEdata = cleanACE(plasmaData, magData) # Comparison with DataFrame obtained from plasmaData implies that this function is working correctly CAA 230425

    return allACEdata
        
def HDF_to_DF(path): 
    '''
        Subsequent 2 lines of code interfaces with HDF data and attributes 
        'SWEPAN_ion' and 'MAG_data_16sec' are the associated dType that are imbued in this data file by ACE
    '''

    if 'swepam' in path: # for extraction of SW plasma data
        hdf = HDF(path) # pyHDF library 
        vd = hdf.vstart().attach('SWEPAM_ion')  
        
        # Convert to Pandas DataFrame
        df = pd.DataFrame(vd[:], columns = vd._fields)
        
        # Closes HDF file interface
        vd.detach()
        hdf.vstart().end()
        hdf.close()

    elif 'mag' in path: # for extraction of SW mag data
        hdf = HDF(path) # pyHDF library 
        vd = hdf.vstart().attach('MAG_data_16sec')

        df = pd.DataFrame(vd[:], columns = vd._fields)

        # Closes HDF file interface
        vd.detach()
        hdf.vstart().end()
        hdf.close()

    return df

def plasmaDF_nan_filter(df):
    df.loc[df['proton_density'] <= -9999, 'proton_density'] = np.nan
    df.loc[df['proton_density'] >= 999, 'proton_density'] = np.nan

    df.loc[df['proton_temp'] <= -9999, 'proton_temp'] = np.nan

    df.loc[df['proton_speed'] <= -9999, 'proton_speed'] = np.nan
    df.loc[df['proton_speed'] >= 999, 'proton_speed'] = np.nan

    df.loc[df['x_dot_GSM'] <= -9999, 'x_dot_GSM'] = np.nan
    df.loc[df['x_dot_GSM'] > 999, 'x_dot_GSM'] = np.nan

    df.loc[df['y_dot_GSM'] <= -9999, 'y_dot_GSM'] = np.nan
    df.loc[df['y_dot_GSM'] > 999, 'y_dot_GSM'] = np.nan

    df.loc[df['z_dot_GSM'] <= -9999, 'z_dot_GSM'] = np.nan
    df.loc[df['z_dot_GSM'] > 999, 'z_dot_GSM'] = np.nan

    return df

def magDF_nan_filter(df): 
    df.loc[df['Bt'] <= -999, 'Bt'] = np.nan
    df.loc[df['Bgse_x'] <= -999, 'Bgse_x'] = np.nan
    df.loc[df['Bgsm_x'] <= -999, 'Bgse_x'] = np.nan
    df.loc[df['Bgsm_y'] <= -999, 'Bgsm_y'] = np.nan
    df.loc[df['Bgsm_z'] <= -999, 'Bgsm_z'] = np.nan

    return df

def cleanACE(plasmaData, magData):
    method = 'linear'
    limit = 15
    
    plasmaData.drop_duplicates(subset='ACEepoch', inplace=True)
    plasmaData = plasmaData.resample('1 min').bfill() # plasmaData is of 64 second resolution, we are "upsampling" to achieve 60 second resolution
    plasmaData = plasmaData.interpolate(method=method, limit=limit)
    
    magData = magData.resample('1 min').mean() # magData is of 16 second resolution, we are "downsampling" to achieve 60 second resolution 
    magData = magData.interpolate(method=method, limit=limit)
    
    allACEdata= pd.DataFrame()
    
    allACEdata['B_Total'] = magData['Bt']
    allACEdata['BY_GSM'] = magData['Bgsm_y']
    allACEdata['BZ_GSM'] = magData['Bgsm_z']
    allACEdata['Vx'] = plasmaData['proton_speed']
    allACEdata['Vy'] = plasmaData['y_dot_GSM']
    allACEdata['Vz'] = plasmaData['z_dot_GSM']
    allACEdata['proton_density'] = plasmaData['proton_density']
    allACEdata['Temp'] = plasmaData['proton_temp']
    
    return allACEdata

def OMNI_DATA_toPd():
    OMNIPath = glob.glob(omni_dir + 'omni_hro_1min_*_v01.cdf')
    all_OMNI = []

    print("Extracting OMNI data now ...")

    for path in OMNIPath: ### Each separate file is for 1 month of OMNI data, at 1 minute resolution // [:12] to speed up processing for debugging purposes
        temp_df = pd.DataFrame()
        
        # The following code interfaces with our CDF files and extracts SYM-H and AE Index only
        cdf = cdflib.CDF(path)
        SYMH = cdf['SYM_H']
        AE = cdf['AE_INDEX']

        # Then, we initialise a DataFrame and insert our extracted data into it // set Date_time as index
        UTC = pd.to_datetime(cdflib.cdfepoch.encode(cdf['Epoch'])) # Gives us an array of all datetime within 1 CDF file (1 month at 1 min intervals)
        temp_df.index = UTC
        temp_df['AE'] = AE
        temp_df['SYMH'] = SYMH

        all_OMNI.append(temp_df)

        print('{0} completed !'.format(path))

    all_OMNI = pd.concat(all_OMNI, axis = 0, ignore_index=False)

    # Replace invalid values with np.nan
    all_OMNI.loc[all_OMNI['AE'] >= 99999, 'AE'] = np.nan
    all_OMNI.loc[all_OMNI['SYMH'] >= 99999, 'SYMH'] = np.nan

    # Interpolate AE index and SYMH
    method = 'linear'
    limit = 15
    all_OMNI['AE'].interpolate(method=method, limit=limit)
    all_OMNI['SYMH'].interpolate(method=method, limit=limit)

    return all_OMNI

def combine_dfs(ACE, OMNI):
    combined = pd.concat([ACE, OMNI], axis = 1, ignore_index=False) # concatenating along axis = 1, adds AE and SYMH as additional columns
    
    print('ACE: ', ACE)
    print('OMNI: ', OMNI)
    print('ACE n OMNI: ', combined)

    return combined

def main(): 
    allACEdata = ACE_DATA_toPd() # Get Pandas DataFrame of all ACE data (Solar Wind + Mag)
    all_OMNI = OMNI_DATA_toPd() # Get Pandas DataFrame of all OMNI data (AE index + SYMH)
    all_ACEnOMNI = combine_dfs(allACEdata, all_OMNI) # Combines ACE and OMNI data into 1 DataFrame

    limit = 15
    method = 'linear'

    # save the files as feathers
    all_OMNI.to_feather(dataDump+'omni_data_{0}_interp.feather'.format(limit))
    allACEdata.to_feather(dataDump+'ace_data_{0}_interp.feather'.format(limit))
    all_ACEnOMNI.to_feather(dataDump+'ace_and_omni_{0}_interp.feather'.format(limit))

if __name__ == '__main__':

	main()

	print('It ran. Good job!')