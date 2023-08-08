from __future__ import print_function

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import seaborn as sns
import warnings
import random
import math
from IPython.display import display
import time 
warnings.filterwarnings('ignore')

import keras
from keras.models import Model
from keras.layers import Dense, Input
from keras.datasets import mnist
from keras.regularizers import l1
from keras.optimizers import Adam
import sys
import os
    
def read_shuffled_chunks(filepath: str, chunk_size: int, file_lenght: int, has_header=False):

    header = 0 if has_header else None
    first_data_idx = 1 if has_header else 0
    # create index list
    index_list = list(range(first_data_idx,file_lenght))

    # shuffle the list in place
    random.shuffle(index_list)

    # iterate through the chunks and read them
    n_chunks = math.ceil(file_lenght/chunk_size)
   
    for i in range(n_chunks):
        rows_to_keep = index_list[(i*chunk_size):((i+1)*chunk_size)]
        if has_header:
            rows_to_keep += [0] # include the index row
        # print("Keep Rows: ",len(rows_to_keep))
        # get the inverse selection
        rows_to_skip = list(set(index_list) - set(rows_to_keep)) 
        # print("Skip Rows: ",len(rows_to_skip))
        yield pd.read_csv(filepath,skiprows=rows_to_skip)       
    
if __name__ == "__main__":
	
    if len(sys.argv) < 1:
        print("Error: you need to specify output directory name")
        print("Correct syntax: python3 encoder_test.py <output directory name>")
        exit()
    start_time = time.time()

    batch_size = 32
    num_dataset_groups = 1
    num_dataset_files_per_group = 4
    num_dataset_files = int(num_dataset_groups*num_dataset_files_per_group)
    num_epochs = 50
    chunk_size = 50000
    max_size = 500000
    max_num_chunks = int(max_size / chunk_size)
    # max_num_chunks = 1
    output_filepath = "/Users/Affan/Documents/2022 Research/Encoder-Apr1st/Training/"
    output_directory = output_filepath+sys.argv[1]+'/'

    if os.path.exists(output_directory):
        print("Error: specify new directory name")
        exit()

    os.mkdir(output_directory)

    #initialize Auto-Encoder Neural Network
    print("Initiaziling Autoencoder Neural Network")
    input_size = 537
    hidden_size_1 = 512
    # hidden_size_2 = 384
    hidden_size_3 = 256
    hidden_size_4 = 128
    code_size = 64 
    input_state = Input(shape=(input_size,))
    encoder_1 = Dense(hidden_size_1, activation='relu')(input_state)
    # encoder_2 = Dense(hidden_size_2, activation ='relu')(encoder_1)
    encoder_3 = Dense(hidden_size_3, activation ='relu')(encoder_1)
    encoder_4 = Dense(hidden_size_4, activation ='relu')(encoder_3)
    code = Dense(code_size, activation='relu')(encoder_4)
    decoder_4 = Dense(hidden_size_4, activation='relu')(code)
    decoder_3 = Dense(hidden_size_3, activation='relu')(decoder_4)
    # decoder_2 = Dense(hidden_size_2, activation='relu')(decoder_3)
    decoder_1 = Dense(hidden_size_1, activation='relu')(decoder_3)
    output = Dense(input_size, activation='relu')(decoder_1)

    autoencoder = Model(input_state, output)
    learning_rate = 1e-04
    optimizer_func = keras.optimizers.SGD(lr = learning_rate)
    loss_func = 'mae'
    autoencoder.compile(optimizer=optimizer_func, loss=loss_func, metrics =['acc'])

    mini_batch_generator = dict()
    batch = pd.DataFrame()
    x_train = np.ndarray([])
    loss = []
    val_loss = []
    sum_loss_array = []
    sum_val_loss_array = [] 
    acc = []
    val_acc = []
    min_val_loss = 1000000

    #Save system parameters to output log file
    system_params = output_directory + "system_params.txt"
    f = open(system_params,'a')

    with open(system_params,'a') as fh:
    # Pass the file handle in as a lambda function to make it callable
        autoencoder.summary(print_fn=lambda x: fh.write(x + '\n'))

    f.write("Loss Function: ")
    f.write(str(loss_func))
    f.write('\n')

    f.write("Optimizer: ")
    f.write(str(optimizer_func))
    f.write('\n')
    f.write("Learning Rate: ")
    f.write(str(learning_rate))
    f.write('\n')

    f.write("Batch Size: ")
    f.write(str(batch_size))
    f.write('\n')

    f.write("Epochs: ")
    f.write(str(num_epochs))
    f.write('\n')

    f.write("Number of Data subsets: ")
    f.write(str(max_num_chunks))
    f.write('\n')

    f.close()
    patience = 0
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print("------ Epoch ", epoch+1,"/",num_epochs," ------------")
        sum_loss = 0
        sum_val_loss = 0 
        print("Creating Data Generators to load small subsets from large dataset")
        count = 0
        for i in range(num_dataset_groups):
            for j in range(num_dataset_files_per_group):
                dataset_filepath ="/Users/Affan/Documents/2022 Research/Encoder-Apr1st/dataset/"
                filename = 'states_dataset_EDF_BESTCQI_35_11_5_'+str(j+1)+'_160_80.csv'
                
                input_file = dataset_filepath+filename
                with open(input_file) as f:
                    file_length = sum(1 for line in f)

                mini_batch_generator[count] = read_shuffled_chunks(input_file,chunk_size,file_length)
                count += 1
        
        for num in range(max_num_chunks):    
            for i in range(num_dataset_files):
                mini_batch = next(mini_batch_generator[i])
                mini_batch = mini_batch.to_numpy()
                if i == 0:
                    x_train = mini_batch
                else:
                    x_train = np.concatenate((x_train,mini_batch))
            print("Loaded ",x_train.shape[0]," data samples for Data Subset ",num+1,"/",max_num_chunks)
            
            history = autoencoder.fit(x_train, x_train,  epochs=1, batch_size = 32, validation_split = 0.8)
            sum_loss += history.history['loss'][0]
            sum_val_loss += history.history['val_loss'][0]

        epoch_end = time.time()

        epoch_training_time = epoch_end - epoch_start
        epoch_training_time = time.strftime("%H:%M:%S", time.gmtime(epoch_training_time))
        print("Epoch Training Time: ",epoch_training_time)           
        loss.append(history.history['loss'][0])
        val_loss.append(history.history['val_loss'][0])

        acc.append(history.history['acc'])
        val_acc.append(history.history['val_acc'])
        sum_loss_array.append(sum_loss)
        sum_val_loss_array.append(sum_val_loss)
    

        #Early Stopping
        if np.abs(sum_val_loss - min_val_loss) <= 0.01*min_val_loss and sum_val_loss != min_val_loss:
            patience += 1
        else:
            patience = 0

        if patience == 5:
            print("Stopping Early because Validation Loss not Improving")
            f = open(system_params,'a')
            f.write("Stopping Early because Validation Loss not Improving")
            f.write('\n')
            f.close()
            break

        if sum_val_loss < min_val_loss:
            min_val_loss = sum_val_loss
        if sum_val_loss > 1.05 * min_val_loss:
            print("Stopping Early due to Over-fitting")
            f = open(system_params,'a')
            f.write("Stopping Early due to Over-fitting")
            f.write('\n')
            f.close()
            break

        #save model after every epoch
        autoencoder.save(output_directory)

        #save updated figures after every epoch
        plt.figure()
        plt.plot(loss, 'b:', label='Training loss')
        plt.plot(val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        fig1_name = output_directory + 'loss.png'
        plt.savefig(fig1_name)

        plt.figure()
        plt.plot(sum_loss_array, 'b:', label='Sum Training loss')
        plt.plot(sum_val_loss_array, 'b', label='Sum Validation loss')
        plt.title('Sum Training and validation loss')
        plt.legend()
        fig2_name = output_directory + 'sum_loss.png'
        plt.savefig(fig2_name)

        plt.figure()
        plt.plot(acc, 'b:', label='Training Accuracy')
        plt.plot(val_acc, 'b', label='Validation Accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        fig3_name = output_directory + 'accuracy.png'
        plt.savefig(fig3_name)

    end_time = time.time()
    total_training_time = end_time - start_time
    total_training_time = time.strftime("%H:%M:%S", time.gmtime(total_training_time))
    print("Total Training time: ",total_training_time)
    
    f = open(system_params,'a')
    f.write("Training Time: ")
    f.write(str(total_training_time))
    f.write('\n')
    f.close()

    filename=output_directory + "stats_loss.csv"
    with open(filename, "w") as f:
        np.savetxt(f,np.column_stack([np.array(loss),np.array(val_loss),np.array(sum_loss_array),np.array(sum_val_loss_array),np.array(acc), np.array(val_acc)]),delimiter=',')
    



