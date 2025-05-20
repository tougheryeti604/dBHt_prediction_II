'''
DEBUGGING CAA 110525 
DATASET ISSUES
1. Check that input data are in the correct format (dimensionality of tensor) // DONE CAA 120525
2. Check dataloader (that it is not corrupting our input data) // DONE CAA 120525
3. Check dataloader (we are not repeatedly inputting the same batch) // DONE CAA 120525
4. Check that batches do not have repeated labels // DONE CAA 120525
5. Check for noise within our datasets 
6. Class imbalance ? 
7. Ensure that data set is properly shuffled  // DONE CAA 120525

DATA AUGMENTATION ISSUE 
1. Are we normalising our data properly 
2. Too much augmentation (drop out layers, ) //DONE CAA 130525 NEVER IMPLEMENT DROP OUT LAYERS

LOSS IMPLEMENTATION
1. Check that correct loss function is used
2. Adjust loss weights // DONE CAA 130525 --> i am only using one loss function
If your loss is composed of several smaller loss functions, make sure their magnitude relative to each is correct.

3. Check for “frozen” layers or variables
Check if you unintentionally disabled gradient updates for some layers/variables that should be learnable.

MODEL IMPLEMENTATION
1. Increase network size
Maybe the expressive power of your network is not enough to capture the target function. Try adding more layers or more hidden units in fully connected layers.

2. Overfit a small subset of the data and make sure it works
3. Change your hyperparameters
4. Ensure that model is in test mode when running on unseen data
5. Be on the lookout for layer activations with a mean much larger than 0. Try Batch Norm or ELUs.
6. Try a different optimiser
7. Check layer updates, as very large values can indicate exploding gradients. Gradient clipping may help.
8. Increase / Decrease learning rate
9. Overcome np.nan values
10. Check weights initialisation
'''
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt # For data viz
import pandas as pd
import numpy as np
import sys
import os
import csv

from sklearn.preprocessing import MinMaxScaler

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class SpaceWeatherDataset(Dataset): 
    def __init__(self, dirs, transform = None): 
        self.data = dirs
        self.transform = transform

    def __len__(self): 
        return len(self.data_dirs)
    
    def __normaliser(self, x): # Data normalisation 
        scaler = MinMaxScaler() 
        scaled_data = scaler.fit_transform(x)
        scaled_df = pd.DataFrame(scaled_data, 
                         columns=x.columns)
        
        return scaled_df
    
    def __getitem__(self, file_name): # Get data of one individual file
        x = pd.read_csv(file_name)
        dBHt = x['dBHt'] 
        
        to_drop = ['SYMH', 'Z', 'F', 'UTC', 'Label dBHt', 'Label UTC', 'Unnamed: 0', 'dBHt']
        x = x.drop(to_drop, axis = 1)     
                  
        x = self.__normaliser(x) # Normalise our data via min-max scaling
        x = pd.concat((x, dBHt), axis = 1, ignore_index=False) # Adds back unnormalised dBHt
        x = torch.Tensor(x.values) # Input data

        y = pd.read_csv(file_name)[['Label dBHt']]
        y = torch.Tensor(y.values) # Label data (probabilities)

        z = pd.read_csv(file_name)[['Label UTC']] # extract dates of data for sorting later

        return x, y, z
    
class dBHtPredictor (nn.Module): 
    def __init__(self, num_classes = 1): 
        super(dBHtPredictor, self).__init__()

        self.rnn = nn.RNN(13, 30, 2)
        self.linear1 = nn.Linear(30,128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 1)
        self.LeakyReLU = nn.LeakyReLU()
        self.ReLU = nn.ReLU()

    def forward(self, x): 
        out = self.rnn(x)
        out = self.linear1(out[0])
        out = self.ReLU(out)
        out = self.linear2(out)
        out = self.ReLU(out)
        out = self.linear3(out)
        out = self.ReLU(out)

        return out

def optimise(train_Dataset, train_dirs):   
    ''' Epoch - 1 run throughout the entire dataset (We can limit our dataset)
    Batch - OUr entire dataset is split into smaller batches for separate loading into the model for trainigng 
    - Gradient descent is carried out per batch
    - optimisers and batch loss should be refreshed at each batch
    - Back propagation is carried out for loss across an entire batch. Must sum up loss for each individual datase '''
   
    # Initialise some variables 
    train_Dataset = train_Dataset
    train_dirs = train_dirs

    # Storing some values for plotting & tracking
    epoch_loss_tracker = []
    all_outputs = []
    all_labels = []
    all_inputs = []

    # Set some model training parameters 
    train_model = dBHtPredictor() # Initialise NN object
    num_epoch = 3 # It is the number of times the model trains over the whole dataset
    batch_size = 5
    dataset_size = 20
    num_batches = dataset_size / batch_size

    lossFunction = nn.MSELoss()
    optimiser = optim.Adam(train_model.parameters(), lr=1e-3)
    dataloader = DataLoader(train_dirs[:dataset_size], batch_size=batch_size, shuffle=True) # Puts all our data files into batches 

    train_model.to(device='cuda')

    for epoch in range(num_epoch): # Train our model for the given of epochs 
        train_model.train()
        total_batch_loss = 0.0
        
        for batch in dataloader: # For every batch of data in our DataLoader 
            total_pair_loss = 0.0
            optimiser.zero_grad() # Clears all previous gradients

            for file in batch: # For each dataset 
                Input, Label, Dates = train_Dataset.__getitem__(file) # Get the Input and Label tensors
                Input = Input.to(device='cuda')
                Label = Label.to(device='cuda')

                outputs = train_model.forward(Input) # Forward pass

                all_outputs.append(outputs) # Storing of outputs // to aid in debugging
                all_labels.append(Label)
                all_inputs.append(Input)

                '''
                Outputs and Label shape are the size of 30x2 --> logits for each class (2 classes), for each of the 30 minutes we are forecasting 
                We are carrying out binary classification with loss calculated with CrossEntropy
                First column of Outputs & Label tensor --> percentage probability exceeded 
                Second column of Outputs & Label tensor --> percentage probability Non exceeded

                nn.CrossEntropyLoss(Input, Target)
                
                The input is expected to contain the unnormalized logits for each class, for each INPUT-LABEL pair (Probability assigned by our model for each class, for each INPUT-LABEL pair)
                The target that this criterion expects should contain : Class indices in the range [0,C) where C is the number of classes ([c1, c2] Label probability for each class)
                '''
    
                loss = lossFunction(outputs, Label) # Calculates loss per data file 
                loss = torch.sqrt(loss) # RMSE
                total_pair_loss += loss # Calculates total pair loss
           
            # Batch gradient descent
            # Batch loss = Average loss for each LABEL-INPUT pair = Total pair loss / number of pairs (batch size)
            batch_loss = total_pair_loss / batch_size 
            batch_loss.backward() 
            optimiser.step()

            total_batch_loss += batch_loss
        
        # Epoch loss = Average batch loss = Total batch loss / number of batches
        epoch_loss = total_batch_loss / num_batches
        epoch_loss_tracker.append(epoch_loss.item())

        print("Epoch: {0}, Loss: {1}".format(epoch+1, epoch_loss))  
      
    return epoch_loss_tracker, train_model.state_dict(), all_outputs, all_labels, all_inputs

def evaluate(val_Dataset, val_dirs): 
    val_Dataset = val_Dataset
    val_dirs = val_dirs

    # Initialising some performance trackers
    loss_tracker = []
    ct = 0

    # Load the model's weights and biases, initialise some evaluation parameters
    eval_model = dBHtPredictor()
    eval_model_params = torch.load('C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/assemble_RNN/model_v1', weights_only=False)
    eval_model.load_state_dict(eval_model_params)
    eval_model.to(device='cuda')
    eval_model.eval() 
    num_data = 100
    accuracyFunction = nn.CrossEntropyLoss()

    for file in val_dirs[:num_data]: 
        ct+=1
        Input, Label, Dates = val_Dataset.__getitem__(file) # Get the Input and Label tensors
        Input = Input.to(device='cuda')
        Label = Label.to(device='cuda') # First column --> Exceeded // Second column --> Non exceeded

        outputs = eval_model.forward(Input)

        # My inputs are working properly. No tensors with np.nan values here

        # Getting loss values for each of our forecasted datasets
        loss = accuracyFunction(outputs.type(torch.float64), Label.type(torch.float64))
        loss = loss.cpu().data.numpy().argmax()
        loss_tracker.append(loss)

        # Getting the forecasted dBHt values produced by our model, and comparing to ground truth data (for one INPUT-LABEL pair, not very USEFUL !)
        if ct == 1: 
            forecasted_dBHt = outputs 
            ground_truth_dBHt = Label
            allDates = Dates

        else: 
            forecasted_dBHt = torch.cat((forecasted_dBHt, outputs[28:29])) # Only the last value of the next predicted data set is appended (previous 29 overlaps)
            ground_truth_dBHt = torch.cat((ground_truth_dBHt, Label[28:29]))
            newDate = Dates.loc[29]['Label UTC']
            newDate = pd.DataFrame([newDate], columns = ['Label UTC'])
            allDates = pd.concat((allDates, newDate), axis = 0, ignore_index=True)

    return loss_tracker, forecasted_dBHt, ground_truth_dBHt

def test(test_Dataset, test_dirs): 
    test_Dataset = test_Dataset
    test_dirs = test_dirs

    # Initialising some performance trackers
    ct = 0
    loss = 0
    total_pair_loss = 0

    # Load the model's weights and biases, initialise some evaluation parameters
    lossFunction = nn.MSELoss()
    test_model = dBHtPredictor()
    test_model_params = torch.load('C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/assemble_RNN/model_v1', weights_only=False)
    test_model.load_state_dict(test_model_params)
    test_model.to(device='cuda')
    test_model.eval() 

    for file in test_dirs: 
        ct+=1

        print("Processing file number: {0}".format(ct))

        Input, Label, Dates = test_Dataset.__getitem__(file) # Get the Input and Label tensors
        Input = Input.to(device='cuda')
        Label = Label.to(device='cuda') # First column --> Exceeded // Second column --> Non exceeded

        outputs = test_model.forward(Input)

        # Getting the forecasted dBHt values produced by our model, and comparing to ground truth data 

        if ct == 1: 
            forecasted_dBHt = outputs 
            ground_truth_dBHt = Label
            allDates = Dates

        else: 
            forecasted_dBHt = torch.cat((forecasted_dBHt, outputs[29:])) # Only the last value of the next predicted data set is appended (previous 29 overlaps)
            ground_truth_dBHt = torch.cat((ground_truth_dBHt, Label[29:]))
            newDate = Dates.loc[29]['Label UTC']
            newDate = pd.DataFrame([newDate], columns = ['Label UTC'])
            allDates = pd.concat((allDates, newDate), axis = 0, ignore_index=True)

    return forecasted_dBHt, ground_truth_dBHt, allDates

def train(train_Dataset, train_dirs): 
    train_loss, model_params, all_outputs, all_labels, all_inputs = optimise(train_Dataset, train_dirs) # optimise() is our training loop
       
    # Save weights and biases 
    torch.save(model_params, 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/assemble_RNN/config_4/model_v1')

    # Save loss values for plotting later
    loss_file = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/assemble_RNN/config_4/loss.csv'
    
    with open (loss_file, 'w') as outF: 
        write = csv.writer(outF)
        write.writerow(train_loss)

    print(all_inputs[4::20], '\n')

    return loss_file

def testnsave(allTest_Dataset, all_test):
    # Test our model's performance on test storm(s)
    forecasted_dBHt, ground_truth_dBHt, allDates = test(allTest_Dataset, all_test)
    forecasted_dBHt = forecasted_dBHt.cpu().detach().numpy()
    ground_truth_dBHt = ground_truth_dBHt.cpu().detach().numpy()

    # Save test values for plotting later
    forecast_file = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/assemble_RNN/config_4/testforecast.pt'
    ground_truth_file = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/assemble_RNN/config_4/groundtruth.pt'
    allDates_file = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/assemble_RNN/config_4/testdates.csv'

    torch.save(forecasted_dBHt, forecast_file)
    torch.save(ground_truth_dBHt, ground_truth_file)
    allDates.to_csv(allDates_file)

    return forecast_file, ground_truth_file, allDates_file

def plot(loss_file, forecast_file , ground_truth_file, allDates_file): 
    print("\n Plotting...")

    time = pd.read_csv(allDates_file)['Label UTC'].values
    forecasted_dBHt = torch.load(forecast_file, weights_only = False)
    ground_truth_dBHt = torch.load(ground_truth_file, weights_only = False)

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    
    
    ax.step(time, forecasted_dBHt)
    x_ticks = time[::2500]
    ax.set_xticks(x_ticks)

    ax2.step(time, ground_truth_dBHt)
    x2_ticks = time[::2500]
    ax2.set_xticks(x2_ticks)
    
    loss =  pd.DataFrame(pd.read_csv(loss_file).columns.values, columns = ['loss'])
    ax3.plot(loss.index.values, loss['loss'].values)
    ax3.invert_yaxis()

    plt.show()

def main(): 
    train_dirs = glob.glob('C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/train/*_PredictingFrom_*_limit15_.csv')
    val_dirs = glob.glob('C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/val/*_PredictingFrom_*_limit15_.csv')
    all_test = glob.glob('C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/all_test/ * PredictingFrom_*_limit15.csv')
    non_na = glob.glob('C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/filtered_test/ * PredictingFrom_*_limit15.csv')

    all_test = sorted(all_test, key=lambda x: int(x.split()[1])) # Ensure that all INPUT-LABEL pairs are taken in by sequential order when testing

    val_Dataset = SpaceWeatherDataset(val_dirs) # Validation data object
    train_Dataset = SpaceWeatherDataset(train_dirs) # Train data object
    allTest_Dataset = SpaceWeatherDataset(all_test)
    nonNATest_Dataset = SpaceWeatherDataset(non_na)

    # Start Training // sets some model training parameters // Returns our loss values and our trained model parameters  
    loss_file = train(train_Dataset, train_dirs)

    # Evaluate on an INPUT-LABEL pair basis
    #@evaluate(val_Dataset, val_dirs)

    # Test our model and save model performance files for plotting
    forecast_file, ground_truth_file, allDates_file = testnsave(allTest_Dataset, all_test)

    loss_file = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/assemble_RNN/config_4/loss.csv'
    forecast_file = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/assemble_RNN/config_4/testforecast.pt'
    ground_truth_file = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/assemble_RNN/config_4/groundtruth.pt'
    allDates_file = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/assemble_RNN/config_4/testdates.csv'

    # Plot training and test data
    plot(loss_file, forecast_file, ground_truth_file, allDates_file)

    return

if __name__ == '__main__':

	main() 