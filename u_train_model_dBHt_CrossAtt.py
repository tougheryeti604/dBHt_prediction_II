import glob
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt # For data viz
import pandas as pd
import numpy as np
import sys
import os
import csv

from sklearn.preprocessing import MinMaxScaler

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn.functional as F

class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, inner_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out
    
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, num_features):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(seq_len, num_features)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_features, 2).float() * -(math.log(10000.0) / num_features))

        # Even indexed input dimensions have sine applied, odd indexed input dimensions have cosine applied
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:,0:-1] # Additional line needed as our input dimension size is an odd number
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(0)].to(device='cuda')

class SingleHeadAttention(nn.Module): # ATTENTION + FEED FORWARD + NORMALISATION + RESIDUAL LAYERS, WITH WHICH OUR ENCODERS AND DECODERS CAN BE BUILT
    def __init__(self, seq_len, num_features, attention_type = "self"):
        super(SingleHeadAttention, self).__init__()
        
        self.attention_type = attention_type

        # Linear layers for Q, K, V for all heads
        self.query = nn.Linear(num_features, num_features)
        self.key = nn.Linear(num_features, num_features)
        self.value = nn.Linear(num_features, num_features)

        # Layers for normalisation and residual connections
        self.norm = nn.LayerNorm(num_features)
        self.dropout = nn.Dropout(0.1)
        
        # Output linear layer
        self.fc_out = nn.Linear(num_features, num_features)

    def attention(self, Q, K, V, mask=None): # CALCULATES OUR ATTENTION SCORES 
        # Compute the dot products between Q and K, then scale
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # Softmax to normalize scores and get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum of values
        output = torch.matmul(attention_weights, V)  # Final attention score 
        
        return output, attention_weights        

    def forward(self, target, source=None, mask=None):
        seq_len, num_features = target.shape 
        
        if self.attention_type == "self":
            Q = self.query(target) # If only one set of Query + Key + Value is provided (self attention), this set will be assigned to the variable Target
            K = self.key(target)
            V = self.value(target)
            
        elif self.attention_type == "cross":
            assert source is not None, "Source input required for cross-attention"
            Q = self.query(target)
            K = self.key(source)
            V = self.value(source)
        
        # Perform scaled dot-product attention and concatenate heads
        out, _ = self.attention(Q, K, V, mask)

        # Add a position wise feed forward layer
        FF = PositionWiseFeedForward(13,30).to(device='cuda')
        out = FF.forward(out)
        
        # Add residual connection and layer normalization
        out = self.norm(target + self.dropout(out))
        
        # Final linear transformation
        return self.fc_out(out)
    
class Transformer(nn.Module): # PUTS OUR ENTIRE TRANSFORMER MODEL TOGETHER --> EMBEDDING + ENCODERS + DECODERS + OUTPUT LAYERS + ACTIVATION FUNCTIONS 
    def __init__(self):
        super(Transformer, self).__init__()

        # Initalise our ENCODERS and DECODERS
        self.encoder_self_attention = SingleHeadAttention(30, 13, attention_type="self")
        self.decoder_self_attention = SingleHeadAttention(30, 13, attention_type="self")
        self.decoder_cross_attention = SingleHeadAttention(30, 13, attention_type="cross")

        # Initialise our OUTPUT layer
        self.Output = nn.Linear(13, 1)
        self.ReLU = nn.ReLU()

    def forward(self, target, source = None, mask = None): 
        # Source and Target should have the shape [1 x 30 x 13] or [30 x 13]
        source = PositionalEncoding(source.shape[0], source.shape[1]).forward(source).squeeze(0)
        target = PositionalEncoding(target.shape[0], target.shape[1]).forward(source).squeeze(0)

        # Coding a Transformer is like coding forward passes (through our ENCODERS & DECODERS) within a forward pass (through our "whole" model)
        source = self.encoder_self_attention.forward(source, None, mask)
        target = self.decoder_self_attention.forward(target, None,  mask)
        target = self.decoder_cross_attention.forward(target, source, mask)

        target = self.Output(target)
        target = self.ReLU(target)

        return target

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
        y = torch.Tensor(y.values) # Label data (probabilities)'

        z = pd.read_csv(file_name)[['Label UTC']] # extract dates of data for sorting later

        return x, y, z
    
def optimise(train_Dataset, train_dirs, model_params = None):   
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
    ct = 0

    # Set some model training parameters 
    train_model = Transformer() # Initialise our AEAEAEAE OPTIMUS PRIME (Transformer Neural Network)
    num_epoch = 1 # It is the number of times the model trains over the whole dataset
    batch_size = 50
    dataset_size = 50000
    num_batches = dataset_size / batch_size

    lossFunction = nn.MSELoss()
    optimiser = optim.Adam(train_model.parameters(), lr=1e-3)
    dataloader = DataLoader(train_dirs[:dataset_size], batch_size=batch_size, shuffle=True) # Puts all our data files into batches 

    train_model.to(device='cuda')

    if model_params != None: 
        print("Loading from checkpoint ...")
        params = torch.load(model_params)
        train_model.load_state_dict(params)

    for epoch in range(num_epoch): # Train our model for the given of epochs 
        train_model.train()
        total_batch_loss = 0.0
        
        for batch in dataloader: # For every batch of data in our DataLoader 
            ct += 1
            total_pair_loss = 0.0
            optimiser.zero_grad() # Clears all previous gradients

            for file in batch: # For each dataset 
                Input, Label, Dates = train_Dataset.__getitem__(file) # Get the Input and Label tensors
                Input = Input.to(device='cuda')
                Label = Label.to(device='cuda')
                
                outputs = train_model.forward(Label, Input) # Forward pass

                all_outputs.append(outputs) # Storing of outputs // to aid in debugging
                all_labels.append(Label)
                all_inputs.append(Input)

                loss = lossFunction(outputs, Label) # Calculates loss per data file 
                loss = torch.sqrt(loss)
                total_pair_loss += loss # Calculates total pair loss
           
            # Batch gradient descent
            # Batch loss = Average loss for each LABEL-INPUT pair = Total pair loss / number of pairs (batch size)
            batch_loss = total_pair_loss / batch_size 
            batch_loss.backward() 
            optimiser.step()

            total_batch_loss += batch_loss

            print("Batch no. {0} completed".format(ct))
                    
        # Epoch loss = Average batch loss = Total batch loss / number of batches
        epoch_loss = total_batch_loss / num_batches
        epoch_loss_tracker.append(epoch_loss.item())

        torch.save(train_model.state_dict(), 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/assemble_Transformer/config_1/checkpoint2')

        print("Epoch: {0}, Loss: {1}".format(epoch+1, epoch_loss))  
      
    return epoch_loss_tracker, train_model.state_dict(), all_outputs, all_labels, all_inputs
    
def train_loop(train_Dataset, train_dirs): # KICK STARTS TRAINING L
    checkpoint = "C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/assemble_Transformer/config_1/checkpoint"

    train_loss, model_params, all_outputs, all_labels, all_inputs = optimise(train_Dataset, train_dirs, checkpoint) # optimise() is our training loop
       
    # Save weights and biases 
    torch.save(model_params, 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/assemble_Transformer/config_1/model_v1')

    # Save loss values for plotting later
    loss_file = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/assemble_Transformer/config_1/loss.csv'
    
    with open (loss_file, 'w') as outF: 
        write = csv.writer(outF)
        write.writerow(train_loss)

    return loss_file

def test(test_Dataset, test_dirs):
    test_Dataset = test_Dataset
    test_dirs = test_dirs

    # Initialising some performance trackers
    ct = 0
    loss = 0
    total_pair_loss = 0

    # Load the model's weights and biases, initialise some evaluation parameters
    lossFunction = nn.MSELoss()
    test_model = Transformer()
    test_model_params = torch.load('C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/assemble_Transformer/config_1/model_v1', weights_only=False)
    test_model.load_state_dict(test_model_params)
    test_model.to(device='cuda')         

    for file in test_dirs: 
        test_model.eval()
        ct+=1

        print("Processing file number: {0}".format(ct))

        Input, Label, Dates = test_Dataset.__getitem__(file) # Get the Input and Label tensors
        Input = Input.to(device='cuda')
        Label = Label.to(device='cuda') # First column --> Exceeded // Second column --> Non exceeded

        mask = torch.tril(torch.ones(30, 30)).to(device='cuda')

        outputs = test_model.forward(Label, Input, mask)

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

def testnsave(allTest_Dataset, all_test):
    # Test our model's performance on test storm(s)
    forecasted_dBHt, ground_truth_dBHt, allDates = test(allTest_Dataset, all_test)
    forecasted_dBHt = forecasted_dBHt.cpu().detach().numpy()
    ground_truth_dBHt = ground_truth_dBHt.cpu().detach().numpy()

    # Save test values for plotting later
    forecast_file = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/assemble_Transformer/config_1/testforecast.pt'
    ground_truth_file = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/assemble_Transformer/config_1/groundtruth.pt'
    allDates_file = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/assemble_Transformer/config_1/testdates.csv'

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

def main(): # TIES TRAINING LOOP, EVALUATION AND TEST TOGETHER 
    train_dirs = glob.glob('C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/train/*_PredictingFrom_*_limit15_.csv')
    val_dirs = glob.glob('C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/val/*_PredictingFrom_*_limit15_.csv')
    all_test = glob.glob('C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/all_test/ * PredictingFrom_*_limit15.csv')
    non_na = glob.glob('C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/filtered_test/ * PredictingFrom_*_limit15.csv')

    all_test = sorted(all_test, key=lambda x: int(x.split()[1])) # Ensure that all INPUT-LABEL pairs are taken in by sequential order when testing

    val_Dataset = SpaceWeatherDataset(val_dirs) # Validation data object
    train_Dataset = SpaceWeatherDataset(train_dirs) # Train data object
    allTest_Dataset = SpaceWeatherDataset(all_test)
    nonNATest_Dataset = SpaceWeatherDataset(non_na)

    # Train model
    #@loss_file = train_loop(train_Dataset, train_dirs)

    # Test our model and save model performance files for plotting
    forecast_file, ground_truth_file, allDates_file = testnsave(allTest_Dataset, all_test)

    loss_file = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/assemble_Transformer/config_1/loss.csv'
    forecast_file = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/assemble_Transformer/config_1/testforecast.pt'
    ground_truth_file = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/assemble_Transformer/config_1/groundtruth.pt'
    allDates_file = 'C:/Users/UserAdmin/Documents/PythonWork/dbdt_MKII_u/Input_Label_dBHt/assemble_Transformer/config_1/testdates.csv'

    # Plot training and test data
    plot(loss_file, forecast_file, ground_truth_file, allDates_file)

    return

if __name__ == '__main__':

	main() 


