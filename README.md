## Overall functionality CAA 220525
These collection of python scripts prepares and processes data from ACE, OMNI and INTERMAG for use as TRAINING, VALIDATION and TESTING INPUT and LABEL data.
This is for training of a deep neural network for the purpose of: 
1. Predicting dBHt threshold excursion probabilities
2. Predicting dBHt values directly 

### Downloading of data 
#### ACE: https://izw1.caltech.edu/ACE/ASC/level2/index.html 
MAG and SWEPAM data 

> MAG: 16 second averaged, yearly data files

> SWEPAM: 64 second averaged, yearly data files 


#### OMNI: https://cdaweb.gsfc.nasa.gov/pub/data/omni/
omni/omni_cdaweb/hro_1min data 


#### INTERMAG: https://intermagnet.org/data_download.html 
1. Go to bulk data, download files in IAGA-2002 format using windows batch file download, by year.
2. INTERMAG integrates data of various types and standards to form the best possible continuous case of magnetometer data.
3. There is best available, definitive, semi-definitive, provisional and variation data.
4. Downloading "best available" data will download a stream of data of various file types (because "best available" concatenates definitive, semi-definitive, provisional and variation data to form a continuous stream of data, depending on what is available at a current point in time).

#### Our current code works with "Definitive data" only.

### u_preparing_SW_data_MKII.py and u_preparing_mag_data_INTERd.py 

These scripts are the first to be run in the series. They take in downloaded ACE, OMNI and INTERMAG data for the indicated timeframe (1997 to 2017) and concatenates them into 1 singular dataframe, before saving them as 
interp feather files.

### u_processing_all_data_MKII.py and u_processing_test_data.py 

These scripts take our prepared Solar Wind and Magnetometer data and converts them into data files for MODEL TRAINING, EVALUATION AND TESTING.

Our data is processed and split into INPUT-LABEL pairs.

#### wt = window (how many minutes of weather are we predicting?) 

#### ft = forecast (how many minutes in advanced are we starting prediction ?)

#### th = time history (how many minutes of input data are we taking in ?)


> If our last label data ends at t = t , Then, our last input data must end at t-wt-ft

> If our input data starts at T = T , Our label data must start from T = T+th+ft, and end at T+th+ft+wt

CURRENTLY, WE ARE WORKING WITH WT = 30, FT = 30, TH = 30

HENCE, INPUT-LABEL pairs are each 30 minutes long

### u_processing_all_data_dBHt.py and u_processing_test_data_dBHt.py 

These carry out the same function as above

While the above produces TRAINING-EVAL-TEST INPUT-LABEL files for dBHt > threshold probability predictions model training,

These set of scripts produces the same files for dBHt predictions model training.

> #### These are the set of scripts currently in use !!


### u_train_model_dBHt.py 

Legacy RNN model used for training of model to predict dBHt values

This script provides the model architecture, data calling, training and testing loops

### u_train_model_dBHt_CrossAtt.py

Current Cross Attention Transformer model used for training to predict dBHt values

The script provides the model architecture, data calling, training and testing loops.

Checkpoint for saving of model parameters implemented, in case training loop is cut due to lack of memory.

### u_train_model_dBHt_debug_ver.py 

Copy of u_train_model_dBHt.py that is non functional and used for debugging of training and testing loops

### stormlist.csv

Work derived from Coughlan et.al 

Time periods where SYM-H < -50nT were identified, and the corresponding storm times were extricated 

This was done in an attempt to reduce data bias (due to low dBHt values during storm quiet times)

### groundtruth.pt

File is updated whenever train model scripts are run. 

Contains torch tensor of ground truth dBHt values. 

### u_train_model_crossing_CNN.py 

Legacy CNN model, train and test loop for training of model to predict dBHt > threshold probabilities

### u_train_model_crossing_RNN.py

Legacy RNN model, train and test loop for training of model to predict dBHt > threshold probabilities

### u_train_model_dBHt_CrossAttMini.py

Cross Attention Transformer model used for training to predict dBHt values, with mini mini batch backpropagation + mini batch gradient descent implemented

Checkpoint for saving of model parameters implemented, in case training loop is cut due to lack of memory.
