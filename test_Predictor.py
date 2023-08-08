#!/usr/bin/python3

import numpy as np 
import tensorflow as tf
import random
from environment_new import Environment
from environment_new import BS
from environment_new import UE
import matplotlib.pyplot as plt
import math 
import sys
import scipy 
import csv
import pandas as pd
from copy import deepcopy
import time
from tensorflow import keras
from collections import deque

from keras.layers import Dense, Activation
from keras import Sequential
from keras.models import load_model
from keras.models import save_model
from tensorflow.keras.optimizers import Adam

from tensorflow.python.keras import backend as kerasBackend
from Predictor_NN import Predictor_Neural_Network
import time as time
# from theano import function
import os
import itertools

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


import tensorflow as tf

np.random.seed()

if __name__ == "__main__":
	np.set_printoptions(precision=3)
	list_of_schedulers = ["EDF","RR","PF","BESTCQI"]
	list_of_predictors = ["RANDOM","BESTCQI","DQN"]
	
	num_episodes = 1      #re-initialized instances of environment
	T=2		#simulation time in seconds, per epsiode.
	scheduling_interval = 125*(10**(-6))
	T = int(T/scheduling_interval) 

	num_training_steps = 0

	if len(sys.argv) < 3:
		print("Error: you need to specify Scheduler and Predictor")
		print("Correct syntax: python3 test_schedulers.py <SCHEDULER> <PREDICTOR>")
		print("List of Schedulers: ",list_of_schedulers)
		print("List of Predictors: ",list_of_predictors)
		print("Example: python3 test_schedulers.py EDF Random")
		exit()

	scheduler = sys.argv[1]
	predictor = sys.argv[2]
	print("Scheduler: ", scheduler)
	print("Predictor: ",predictor)
	
	if scheduler.strip().upper() not in list_of_schedulers:
		print("ERROR - Incorrect scheduler")
		print("Correct syntax: python3 test_schedulers.py <SCHEDULER> <PREDICTOR>")
		print("List of Schedulers: ",list_of_schedulers)
		print("List of Predictors: ",list_of_predictors)
		print("Example: python3 test_schedulers.py EDF Random")
		exit()

	if predictor.strip().upper() not in list_of_predictors:
		print("ERROR - Incorrect predictor")
		print("Correct syntax: python3 test_schedulers.py <SCHEDULER> <PREDICTOR>")
		print("List of Schedulers: ",list_of_schedulers)
		print("List of Predictors: ",list_of_predictors)
		print("Example: python3 test_schedulers.py EDF Random")
		exit()

	program_start_time = time.time()
	
	N = int(sys.argv[3])
	M = int(sys.argv[4])
	K = int(sys.argv[5])
	L = int(sys.argv[6])
	
	#load state autoencoder
	encoder_directory = "/Users/Affan/Documents/NYU/2022 Research/Encoder-Apr1st/Training/Try8"
	predictor_directory = "/Users/Affan/Documents/NYU/2022 Research/Predictor-Apr2nd/Training/Try12"
	autoencoder = keras.models.load_model(encoder_directory)

	#load only the Encoder part of the saved autoencoder
	layer_name = 'dense_3'
	encoder_only_model = keras.Model(inputs = autoencoder.input, outputs = autoencoder.get_layer(layer_name).output)

	#initialize environment with M gNBS, N UEs, degree of multiconnectivity K, degree of replication L and simulation run time of T ms (1  ms = 1 time slot)
	env= Environment(N,M,K,L,T)      #environment for Predictor Test
	env.scheduling_interval = scheduling_interval
	env.connectivity_handover_delay = int((int(sys.argv[7])*10**-3)/env.scheduling_interval)
	env.replication_handover_delay = int((int(sys.argv[8])*10**-3)/env.scheduling_interval)
	env.predictor_algo = predictor
	env.scheduler_alog = scheduler

	env_bestcqi = Environment(N,M,K,L,T)       #Environment for BestCQI Test

	#Store names of output/logging variables in output file as headers
	filename = f'{scheduler}_{env.predictor_algo}_{N}_{M}_{K}_{L}_{env.connectivity_handover_delay}_{env.replication_handover_delay}.csv'
	filepath ="/Users/Affan/Documents/2022 Research/Predictor-Apr2nd/Test/"
	output_predictor_model_directory = filepath+sys.argv[9]+'/'
	output_file = output_predictor_model_directory + filename

	# if os.path.exists(output_predictor_model_directory):
	# 	print("Error: specify new directory name")
	# 	exit()

	# os.mkdir(output_predictor_model_directory)
	env.initialize_system()

	env_bestcqi = deepcopy(env)
	env_bestcqi.predictor_algo = "BESTCQI"
	UE_states = []

	if predictor == "DQN":
		## For the Predictor Neural Network
		for i in range(env.UEs):
			state = env.state_generator_predictor(i)
			state = np.reshape(state, [1, len(state)])
			encoded_state = encoder_only_model(state).numpy()[0]
			UE_states.append(encoded_state)
		
		predictor_state_size = encoded_state.shape[0]
		predictor_action_size = M

		#Load Predictor DQN
		predictor_agent = Predictor_Neural_Network(predictor_state_size, predictor_action_size)
		loaded_model = keras.models.load_model(predictor_directory)
		predictor_agent.model.set_weights(loaded_model.get_weights())
		predictor_agent.epsilon = predictor_agent.min_epsilon
		env.predictor_agent = predictor_agent
		
	traffic_in_system = 0
	for episode in range(num_episodes):
		if episode > 0:
			env.initialize_system()
			env_bestcqi = env

		total_training_rewards = 0
		
		num_errors = np.zeros(env.UEs, dtype = int)

		cp = time.time()
		#Now running simulation for T time-slot
		loss = []			#training loss for Predictor
		val_loss = []		#validation loss for Predictor
		sum_reward = 0
		sum_reward_bestcqi = 0
		sum_ideal_reward = 0
		sum_ideal_reward_bestcqi = 0
		for i in range(T):

			if i % 100 == 0:
				print("--------------T = ",i," -----------------------------------")
			env.current_time = i
			cp = time.time()
			#check if there are any new arrivals, and if so, insert them into buffers
			env.insert_arrivals(i)
			env_bestcqi.insert_arrivals(i)
			# cp1 = time.time()
			# print("Time to Insert Arrivals: ", cp1 - cp)
			#if no traffic in entire system, continue to next time-slot
			total_system_occupancy = 0
			for j in range(env.UEs):
				current_UE = env.list_of_UEs[j]
				total_system_occupancy += current_UE.current_buffer_occupancy
			if total_system_occupancy == 0:
				continue
			else:
				traffic_in_system += 1

			# cp2 = time.time()
			#Serve UEs according to the scheduling action, and then update traffic matrices and deadlines - observe the REWARD
			if scheduler.strip().lower() == 'edf':
				action = env.Earliest_Deadline_First()
				action_bestcqi = env_bestcqi.Earliest_Deadline_First()
			elif scheduler.strip().lower() == 'pf':
				alpha = 1 
				beta = 1
				action = env.Other_Scheduling_Algorithms(alpha, beta)
				action_bestcqi = env_bestcqi.Other_Scheduling_Algorithms(alpha, beta)
			elif scheduler.strip().lower() == 'rr':
				alpha = 0 
				beta = 1
				action = env.Other_Scheduling_Algorithms(alpha, beta)
				action_bestcqi = env_bestcqi.Other_Scheduling_Algorithms(alpha, beta)
			elif scheduler.strip().lower() == 'bestcqi':
				alpha = 1 
				beta = 0
				action = env.Other_Scheduling_Algorithms(alpha, beta)
				action_bestcqi = env_bestcqi.Other_Scheduling_Algorithms(alpha, beta)
			# cp3 = time.time()
			# print("Time to take Scheduling Action: ", cp3 - cp2)
			
			#Serve UEs according to the scheduling action, and then update traffic matrices and deadlines - observe the REWARD
			reshaped_action = np.reshape(action, (M,N))
			
			#Find ideal reward for DQN, if maximum reward demands are served irrespective of network connectivity and conditions
			frame_sizes = []
			for j in range(env.UEs):
				if env.list_of_UEs[j].current_buffer_frame_initial_size[0] != 0:
					fractional_frame_size = env.list_of_UEs[j].current_buffer_frame_size[0] / env.list_of_UEs[j].current_buffer_frame_initial_size[0]
				else:
					fractional_frame_size = 0
				frame_sizes.append(fractional_frame_size)
			frame_sizes = np.array(frame_sizes)

			max_ind = np.argpartition(frame_sizes, -M)[-M:]
			max_ind_values = frame_sizes[max_ind]
			ideal_reward = np.sum(max_ind_values)
			
			#Find ideal reward for BESTCQI, if maximum reward demands are served irrespective of network connectivity and conditions
			frame_sizes = []
			for j in range(env.UEs):
				if env_bestcqi.list_of_UEs[j].current_buffer_frame_initial_size[0] != 0:
					fractional_frame_size = env_bestcqi.list_of_UEs[j].current_buffer_frame_size[0] / env_bestcqi.list_of_UEs[j].current_buffer_frame_initial_size[0]
				else:
					fractional_frame_size = 0
				frame_sizes.append(fractional_frame_size)
			frame_sizes = np.array(frame_sizes)

			max_ind = np.argpartition(frame_sizes, -M)[-M:]
			max_ind_values = frame_sizes[max_ind]
			ideal_reward_bestcqi = np.sum(max_ind_values)

			### Run Predictor DQN ###

			#find actual reward observed in environment by taking the chosen action
			# cp4 = time.time()
			reward = env.traffic_transitions(reshaped_action)    #equivalent to taking a step
			# print("DQN Reward: ",reward)
			current_predictor_Q_values = []
			encoded_state_UEs = []
			
			c0 = time.time()
			for j in range(env.UEs):	
				#update channel states for the next time slot
				encoded_state = np.reshape(encoded_state, [1, predictor_state_size])
				c0a = time.time()
				Q_values = predictor_agent.model(encoded_state).numpy().flatten()
				c0b = time.time()
				print("Time for 1 prediction: ",c0b-c0a)
				
				encoded_state_UEs.append(encoded_state)
				current_predictor_Q_values.append(Q_values)
			c1 =time.time()
			env.channel_transitions(current_predictor_Q_values)
			
			#generate next state based on the traffic and channel transitions

			c2=time.time()
			next_state_UEs = []
			for j in range(env.UEs):
				next_state = env.state_generator_predictor(j)
				next_state = np.reshape(next_state, [1, len(next_state)])
				next_state = encoder_only_model(next_state).numpy()[0]
				next_state = np.reshape(next_state, [1, len(next_state)])
				next_state_UEs.append(next_state)
				
			encoded_state = next_state
			c3=time.time()
			print("Prediction Time: ",c1-c0)
			print("Encoding Time: ",c3-c2)
			print("Total Time: ",c1-c0+c3-c2)
			# exit()
			### Run BESTCQI Predictor ###
			reward_bestcqi = env_bestcqi.traffic_transitions(action_bestcqi)    #equivalent to taking a step
			# print("BESTCQI Reward: ",reward_bestcqi)
			# print("")
			c4 = time.time()
			for j in range(env.UEs):
				env_bestcqi.list_of_UEs[j].current_link_capacity = deepcopy(env.list_of_UEs[j].current_link_capacity)
			env_bestcqi.channel_transitions(np.zeros(env.UEs))
			c5 = time.time()
			print("BESTCQI Time:", c5-c4)
			exit()
			sum_reward += reward 
			sum_ideal_reward += ideal_reward
			sum_reward_bestcqi += reward_bestcqi	
			sum_ideal_reward_bestcqi += ideal_reward_bestcqi

			if i % 80 == 0:
				env.reward_array.append(sum_reward)
				env.ideal_reward_array.append(sum_ideal_reward)
				env_bestcqi.reward_array.append(sum_reward_bestcqi)
				env_bestcqi.ideal_reward_array.append(sum_ideal_reward_bestcqi)
				sum_reward = 0
				sum_ideal_reward = 0	
				sum_reward_bestcqi = 0
				sum_ideal_reward_bestcqi = 0
			
			#save stats every 1,000 timesteps
			if len(env.reward_array) == 10:
				filename=output_predictor_model_directory + "test_stats_loss.csv"
				with open(filename, "a") as f:
					np.savetxt(f,np.column_stack([np.array(env.reward_array),np.array(env.ideal_reward_array),np.array(env_bestcqi.reward_array), np.array(env_bestcqi.ideal_reward_array)]),delimiter=',')
				env.reward_array = []
				env.ideal_reward_array = []
				env_bestcqi.reward_array = []
				env_bestcqi.ideal_reward_array = []
				
		print("Traffic in system ",100*traffic_in_system / (T * (episode + 1))," percent of the time")	
		## STATS CALCULATIONS ##

		#count time spent in blockage for all BSs
		UE_hol_deadlines = []

		for k in range(env.UEs):
			current_UE = env.list_of_UEs[k]
			
			UE_hol_deadlines.append(current_UE.current_deadlines[0])
			if np.sum(current_UE.BS_Replication_Set) < 1:
				current_UE.stats_number_of_timeslots_with_no_coverage += 1

			for j in range(env.BSs):
				if current_UE.blockage_timer[j] > 0 or current_UE.Distance_Matrix[j] > env.connection_threshold:#j not in current_UE.BS_Candidate_Set:
					current_UE.time_in_blockage[j] += 1

				if current_UE.BS_Candidate_Set[j] == 1:
					current_UE.stats_number_of_timeslots_in_candidate_set[j] += 1
				if current_UE.BS_Connected_Set[j] == 1:
					current_UE.stats_number_of_timeslots_in_connectivity_set[j] += 1
				if current_UE.BS_Replication_Set[j] == 1:
					current_UE.stats_number_of_timeslots_in_replication_set[j] += 1

				if current_UE.BS_Replication_Set[j] == 1 and current_UE.BS_Connected_Set[j] == 0:
					num_errors[k] += 1
				
				#vacancy in Connectivity despite enough Candidates available
				if np.sum(current_UE.BS_Connected_Set) < env.degree_of_multiconnectivity and np.sum(current_UE.BS_Candidate_Set) >= env.degree_of_multiconnectivity and current_UE.BS_Candidate_Set[j] == 1:
					env.stats_wrong_vacancy_in_connectivity_set[k]+= 1

			#time spent with vacancy in Replication Set due to blockage
			if np.sum(current_UE.BS_Replication_Set) < env.degree_of_replication:
				env.stats_vacancy_in_replication_set[k] += 1
				
			#time spent with vacancy in Connectivity Set due to blockage
			if np.sum(current_UE.BS_Connected_Set) < env.degree_of_multiconnectivity:
				env.stats_vacancy_in_connectivity_set[k] += 1	

			#time spent with Empty Replication Set i.e. zero conenctivity; data plane interruption = delta_L
			if np.sum(current_UE.BS_Replication_Set) == 0:
				env.stats_empty_replication_set[k] += 1

			#time spent with Empty Connectivity Set i.e. zero connectivity; data plane interruption = delta_L + delta_K
			if np.sum(current_UE.BS_Connected_Set) == 0:
				env.stats_empty_connectivity_set[k] += 1			

			env.stats_actual_average_candidacy[k] += np.sum(current_UE.BS_Candidate_Set)
			env.stats_actual_average_connectivity[k] += np.sum(current_UE.BS_Connected_Set)
			env.stats_actual_average_replication[k] += np.sum(current_UE.BS_Replication_Set)

			cp5 = time.time()
			
			#calculate the average outage duration for each UE
			if env.stats_outage_counter[k] == 0 and np.sum(current_UE.BS_Replication_Set) == 0:
				env.stats_outage_counter[k] = 1 
				env.stats_number_of_outages[k] += 1

			elif env.stats_outage_counter[k] != 0 and np.sum(current_UE.BS_Replication_Set) == 0:
				env.stats_outage_counter[k] += 1 
			
			elif env.stats_outage_counter[k] != 0 and np.sum(current_UE.BS_Replication_Set) != 0:
				env.stats_average_outage_duration[k] += env.stats_outage_counter[k]
				env.stats_outage_counter[k] = 0
	
	#--------------------------------------------------##
	# Experiment over; Calculating and Printing Stats ##
	#--------------------------------------------------##

	env.stats_average_outage_duration = env.stats_average_outage_duration / env.stats_number_of_outages
	avg_blockage_duration = env.stats_system_average_blockage_duration / env.stats_system_total_num_blockages   #in num of timeslots
	avg_blockage_duration = avg_blockage_duration * env.scheduling_interval   #in seconds
	print("Avg Blockage Duration: ",avg_blockage_duration," seconds")
	print("Number of Outages: ",env.stats_number_of_outages)
	print("Average Outage Duration: ",env.stats_average_outage_duration)
	print("Empty Replication: ",env.stats_empty_replication_set)
	print("Vacancy Replication: ",env.stats_vacancy_in_replication_set)
	
	env.stats_actual_average_candidacy = env.stats_actual_average_candidacy / T 
	env.stats_actual_average_connectivity = env.stats_actual_average_connectivity / T 
	env.stats_actual_average_replication = env.stats_actual_average_replication / T 

	print("Average Candidacy: ",env.stats_actual_average_candidacy)
	print("Average Connectivity: ",env.stats_actual_average_connectivity)
	print("Average Replication: ",env.stats_actual_average_replication)

	print("Execution Time: ",time.time()-cp)
	print("Errors: ",num_errors)
	
	total_served=0
	total_expired=0
	served_array=np.zeros(env.UEs,dtype=int)
	expired_array=np.zeros(env.UEs,dtype=int)
	
	avg_number_of_handovers_in_connectivity_set=0
	avg_number_of_handovers_in_replication_set=0
	BS_load=np.zeros(env.BSs,dtype=float)
	avg_time_spent_in_buffer_system=0

	UEs_avg_throughput = np.zeros(env.UEs, dtype = float)
	UEs_num_frames_served = np.zeros(env.UEs, dtype = int)
	UEs_num_frames_expired = np.zeros(env.UEs, dtype = int)
	UEs_avg_time_spent_in_buffer = np.zeros(env.UEs, dtype = int)
	UEs_avg_fraction_served_of_expired_frames = np.zeros(env.UEs, dtype = float)
	UEs_percent_packets_served = np.zeros(env.UEs, dtype = float)
	UEs_percentage_service = np.zeros(env.UEs, dtype = float) #percentage of a UE's traffic that is delivered within deadline- target is 99%

	total_blockage = 0;
	
	for i in range(env.UEs):
		current_UE=env.list_of_UEs[i]

		served_array[i]=current_UE.num_frames_served
		expired_array[i]=current_UE.num_frames_expired
		total_served=total_served+current_UE.num_frames_served
		total_expired=total_expired+current_UE.num_frames_expired

		UEs_num_frames_served[i] = current_UE.num_frames_served
		UEs_num_frames_expired[i] = current_UE.num_frames_expired
		print("------------------------------")
		print("UE ",i," ; Traffic Type: ",current_UE.traffic_labels[current_UE.UE_traffic_type]," ;Traffic Deadline: ",current_UE.packet_delay_budget[current_UE.UE_traffic_type])
		print("Served: ",current_UE.num_frames_served, " ,Expired: ",current_UE.num_frames_expired," ,Percentage Served: ",100*current_UE.num_frames_served/(current_UE.num_frames_served+current_UE.num_frames_expired),"%")
		
		if current_UE.num_frames_expired > 0:
			UEs_avg_fraction_served_of_expired_frames[i] = current_UE.stats_fraction_served_of_expired_frames/current_UE.num_frames_expired
		else:
			UEs_avg_fraction_served_of_expired_frames[i] =  float("nan")
		#avg_link_capacity=current_UE.stats_avg_link_capacity*12    #in Mbps
		#print("Average Link Capacity: ",avg_link_capacity)
		print("Number of Handovers in Connectivity Set: ",current_UE.stats_number_of_handovers_in_connectivity_set)
		print("Number of Timeslots spent in Connectivity Set: ",current_UE.stats_number_of_timeslots_in_connectivity_set)
		print("Number of Handovers in Replication Set: ",current_UE.stats_number_of_handovers_in_replication_set)
		print("Number of Timeslots spent in Replication Set: ",current_UE.stats_number_of_timeslots_in_replication_set)
		avg_number_of_handovers_in_connectivity_set=avg_number_of_handovers_in_connectivity_set+current_UE.stats_number_of_handovers_in_connectivity_set
		avg_number_of_handovers_in_replication_set=avg_number_of_handovers_in_replication_set+current_UE.stats_number_of_handovers_in_replication_set
		BS_load=BS_load+current_UE.stats_service_from_BS
		current_UE.stats_service_from_BS=100*current_UE.stats_service_from_BS/np.sum(current_UE.stats_service_from_BS)
		print("Percentage Service from BSs: ",current_UE.stats_service_from_BS)
		current_UE.stats_avg_number_of_gNBs_in_Candidate_Set = current_UE.stats_avg_number_of_gNBs_in_Candidate_Set/T
		print("Avg Number of gNBs in Candidate Set: ", current_UE.stats_avg_number_of_gNBs_in_Candidate_Set)
		current_UE.stats_avg_number_of_gNBs_in_Connectivity_Set = current_UE.stats_avg_number_of_gNBs_in_Connectivity_Set/T
		print("Avg Number of gNBs in Connected Set: ",current_UE.stats_avg_number_of_gNBs_in_Connectivity_Set)
		current_UE.stats_avg_number_of_gNBs_in_Replication_Set = current_UE.stats_avg_number_of_gNBs_in_Replication_Set/T
		print("Avg Number of gNBs in Replication Set: ",current_UE.stats_avg_number_of_gNBs_in_Replication_Set)

		#current_UE.stats_avg_link_capacity=current_UE.stats_avg_link_capacity/current_UE.stats_number_of_timeslots_in_connectivity_set     #in packets per timeslot
		#current_UE.stats_avg_link_capacity=current_UE.stats_avg_link_capacity*12   			#in Mbps
		print("Distance to BSs: ",current_UE.Distance_Matrix)
		print("Average Link Capacity: ",current_UE.stats_avg_link_capacity/T)
		print("Max Link Capacity: ",current_UE.stats_max_link_capacity)   #in Mbps
		print("Min Link Capacity: ",current_UE.stats_min_link_capacity)	  #in Mbps
		print("Percentage Time Spent in Blockage: ",100*current_UE.stats_in_blockage/T)
		print("Blockage Arrivals: ",current_UE.stats_blockage_arrivals)
		print("Blockage Durations: ",current_UE.stats_blockage_durations)
		UEs_avg_throughput[i] = (current_UE.stats_total_bytes_served / (T*env.scheduling_interval))*8/(10**6)   #in Mbps
		print("Average Throughput: ",UEs_avg_throughput[i])
		print("Total Bytes Served: ",current_UE.stats_total_bytes_served)
		total_blockage += current_UE.stats_blockage_count

		#avg time spent in buffer for delivered frames
		avg_time_spent_in_buffer=0
		avg_time_spent_in_buffer_for_expired_frames=0
		count_served=0
		count_expired=0
	
		for j in range(current_UE.stats_num_total_arrivals):
			if current_UE.frames_time_spent_in_buffer[j] < current_UE.packet_delay_budget[current_UE.UE_traffic_type]:
				avg_time_spent_in_buffer=avg_time_spent_in_buffer+current_UE.frames_time_spent_in_buffer[j]
				count_served = count_served + 1
			elif current_UE.frames_time_spent_in_buffer[j] >= current_UE.packet_delay_budget[current_UE.UE_traffic_type]:
				avg_time_spent_in_buffer_for_expired_frames=avg_time_spent_in_buffer_for_expired_frames+current_UE.frames_time_spent_in_buffer[j]
				count_expired=count_expired+1
		if count_served != 0:
			avg_time_spent_in_buffer=avg_time_spent_in_buffer/count_served		
			print("Avg Time Spent in Buffer for",count_served ,"successfully delivered frames: ",avg_time_spent_in_buffer)
		UEs_avg_time_spent_in_buffer[i] = avg_time_spent_in_buffer
		avg_time_spent_in_buffer_system=avg_time_spent_in_buffer_system+avg_time_spent_in_buffer
		if count_expired != 0:
			avg_time_spent_in_buffer_for_expired_frames=avg_time_spent_in_buffer_for_expired_frames/count_expired
			print("Avg Time Spent in Buffer for",count_expired ,"expired frames: ",avg_time_spent_in_buffer_for_expired_frames)
		print("Fraction Served of Expired Frames: ", current_UE.stats_fraction_served_of_expired_frames)
		current_UE.stats_avg_time_spent_in_buffer=avg_time_spent_in_buffer
		#print(current_UE.frames_time_spent_in_buffer)

		UEs_percent_packets_served[i] = 100*(current_UE.stats_fraction_served_of_expired_frames + current_UE.num_frames_served)/(current_UE.num_frames_served+current_UE.num_frames_expired)

	BS_load=100*BS_load/np.sum(BS_load)
	print("BS Load Distribution (Percentage):  ",BS_load)
	print("Total Served: ",total_served)
	print("Total Expired: ",total_expired)
	print("Total Percentage Served: ",100*total_served/(total_expired+total_served),"%")
	

	avg_number_of_handovers_in_connectivity_set=avg_number_of_handovers_in_connectivity_set/env.UEs
	print("Avg Number of Handovers in Connectivity Set: ",avg_number_of_handovers_in_connectivity_set)
	avg_number_of_handovers_in_replication_set=avg_number_of_handovers_in_replication_set/env.UEs
	print("Avg Number of Handovers in Replication Set: ",avg_number_of_handovers_in_replication_set)
	avg_time_spent_in_buffer_system=avg_time_spent_in_buffer_system/env.UEs
	print("Avg Time Spent in Buffer for Successfully Delivered Frames in Entire System: ",avg_time_spent_in_buffer_system)
	interarrival_time=math.ceil(1/60*1000)    #in ms - temporarily hardcoded with 60 fps
	num_total_arrivals=math.floor(T/interarrival_time)*env.UEs
	#mean_incoming_size=np.sum(env.packet_size)/num_total_arrivals
	avg_system_throughput = np.mean(UEs_avg_throughput)
	print("Average Throughput across all UEs: ",avg_system_throughput)
	print("Total Blockage: ",total_blockage)
	total_time_without_coverage_across_all_UEs = 0
	for i in range(env.UEs):
		current_UE = env.list_of_UEs[i]
		print("UE: ",i)
		print("Distances:                      ",current_UE.Distance_Matrix)
		print("Time Spent in Blockage:         ",current_UE.time_in_blockage)
		print("Time Spent in Candidate Set:    ",current_UE.stats_number_of_timeslots_in_candidate_set)
		print("Time Spent in Connectivity Set: ", current_UE.stats_number_of_timeslots_in_connectivity_set)
		print("Time Spent in Replication Set:  ", current_UE.stats_number_of_timeslots_in_replication_set)
		print("Total Time in Replication: ",np.sum(current_UE.stats_number_of_timeslots_in_replication_set))
		print("Time without any coverage:      ",current_UE.stats_number_of_timeslots_with_no_coverage)
		total_time_without_coverage_across_all_UEs += current_UE.stats_number_of_timeslots_with_no_coverage

	print("Total Service Time: ")
	print(env.stats_total_service_time)
	print("Total Service Time - BSs: ", np.sum(env.stats_total_service_time, axis =1))
	print("Total Service Time - UEs: ", np.sum(env.stats_total_service_time, axis =0))
	print("Total time without coverage across all UEs: ",total_time_without_coverage_across_all_UEs )
	
	print("Time with Vacancy in Replication Set: ", env.stats_vacancy_in_replication_set)
	print("Time with Vacancy in Connectivity Set: ", env.stats_vacancy_in_connectivity_set)
	print("Time with Empty Replication Set: ", env.stats_empty_replication_set)
	print("Time with Empty Connectivity Set: ", env.stats_empty_connectivity_set)
	avg_replication_vacancy_time = np.average(env.stats_vacancy_in_replication_set)
	percentage_replication_vacancy_time = 100*avg_replication_vacancy_time/T
	print("Percentage of Time with Vacancy in Replication Set: ",percentage_replication_vacancy_time)
	avg_connectivity_vacancy_time = np.average(env.stats_vacancy_in_connectivity_set)
	percentage_connectivity_vacancy_time = 100*avg_connectivity_vacancy_time/T
	print("Percentage of Time with Vacancy in Connectivity Set: ",percentage_connectivity_vacancy_time)

	avg_replication_empty_time = np.average(env.stats_empty_replication_set)
	percentage_empty_replication_time = 100*avg_replication_empty_time/T
	print("Percentage of Time with Empty Replication Set: ", percentage_empty_replication_time)
	avg_connectivity_empty_time = np.average(env.stats_empty_connectivity_set)
	percentage_connectivity_empty_time = 100*avg_connectivity_empty_time/T
	print("Percentage of Time with Empty Connectivity Set: ",percentage_connectivity_empty_time)
	print("Wrong Vacancy in Connectivity Set: ")
	print(env.stats_wrong_vacancy_in_connectivity_set)

	UEs_percentage_service = 100*UEs_num_frames_served / (UEs_num_frames_served+UEs_num_frames_expired)

	# output = np.array([total_served,total_expired,avg_time_spent_in_buffer_system])
	output = np.concatenate((UEs_avg_throughput,UEs_num_frames_served, UEs_num_frames_expired, UEs_avg_time_spent_in_buffer, UEs_percent_packets_served, env.stats_vacancy_in_replication_set, env.stats_vacancy_in_connectivity_set, env.stats_empty_replication_set, env.stats_empty_connectivity_set, env.stats_actual_average_candidacy, env.stats_actual_average_connectivity, env.stats_actual_average_replication, env.stats_average_outage_duration,env.stats_number_of_outages))
	with open(output_file, "a") as f:
		np.savetxt(f,np.column_stack(output),delimiter=',')



	

	
	# 