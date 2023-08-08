import tensorflow as tf
import numpy as np
from tensorflow import keras

from collections import deque
from scipy.stats import entropy
import time
import random
import datetime


RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class Predictor_Neural_Network():
    def __init__(self,state_shape,action_shape):
        self.learning_rate = 5e-4
        self.optimizer = keras.optimizers.SGD(learning_rate = self.learning_rate)
        self.loss_fcn = 'Huber'

        self.discount_factor = 0.99
        self.MIN_REPLAY_SIZE = 10000
        self.batch_size = 32
        self.steps_to_update_target_model = 0

        self.epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
        self.max_epsilon = 1 # You can't explore more than 100% of the time
        self.min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
        self.decay = 0.01
        self.hidden_layer_size = 1000
        self.replay_memory = deque(maxlen=self.MIN_REPLAY_SIZE) 

        self.init = tf.keras.initializers.HeUniform()
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(self.hidden_layer_size, input_dim= state_shape,  activation='relu', kernel_initializer=self.init))   #kernel_regularizer='l2'
        self.model.add(keras.layers.Dense(self.hidden_layer_size, activation='relu',  kernel_initializer=self.init))
        self.model.add(keras.layers.Dense(action_shape, activation='linear',  kernel_initializer=self.init))
        self.model.compile(loss= self.loss_fcn, optimizer=self.optimizer)
        self.model.summary()
        self.history = []
        self.train_loss = []
        self.train_acc = []
        self.test_loss = []        

    def train(self,target_model):
        if len(self.replay_memory) < self.MIN_REPLAY_SIZE:
            if len(self.replay_memory) % 1000 == 0:
                print(len(self.replay_memory), " samples stores in Replay Memory")
            return 
        # print("Training Proposal")
        mini_batch = random.sample(self.replay_memory, self.batch_size)
       
        # print("Training Predictor")
        start = time.time()
        
        #Stored Experience Format: [state, action, reward, next_state, action_subset, next_action_subset]
        current_states = np.array([transition[0] for transition in mini_batch])
        current_qs_list = self.model.predict(current_states, verbose = 0)
        
        next_states = np.array([transition[3] for transition in mini_batch])
        future_qs_list = target_model.model.predict(next_states, verbose = 0)         

        X = []
        Y = []
        for index, (observation, action, reward, new_observation) in enumerate(mini_batch):
            current_qs = current_qs_list[index]
            future_qs = future_qs_list[index]   

            max_future_q = reward + self.discount_factor * np.max(future_qs)
            
            #do an update on current Q-values           
            current_qs = (1 - self.learning_rate) * current_qs + self.learning_rate * max_future_q
            
            X.append(observation)
            Y.append(current_qs)
        
        self.history = self.model.fit(np.array(X), np.array(Y), validation_split = 0.2, epochs = 1, batch_size=self.batch_size, verbose=1, shuffle=True)
        
        self.train_loss.append(self.history.history['loss'])
        self.test_loss.append(self.history.history['val_loss'])
        end = time.time()
        



