#!/usr/bin/python3

#This script codes the environment for the multi-user scheduling problem
#Last Updated: Nov 1,2022

import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial import distance
from itertools import permutations
from copy import deepcopy
import scipy as scipy
from scipy import signal
import matplotlib.pyplot as plt
import math as math
from random import randint
from array import array
import random
import base64
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import networkx as nx

class Environment:
	def __init__(self,N,M,K,L,simulation_time=10,num_topologies=1):
		self.UEs=int(N)	#number of users
		self.BSs=int(M)		#number of BSs
		self.scheduling_interval=125*(10**(-6))		#125 micro-second scheduling interval/timeslot

		self.list_of_UEs=[]
		self.list_of_gNBs=[]
		self.degree_of_multiconnectivity=int(K)
		self.degree_of_replication=int(L)
		self.connectivity_handover_delay = int((10*10**-3)/self.scheduling_interval) #10 ms, in no. of scheduling time slots
		self.replication_handover_delay = int((2*10**-3)/self.scheduling_interval) #2 ms, in no. of scheduling time slots
		self.dynamic_blocker_density = 0.01

		self.current_time=0
		self.bitwise_replication_set=np.zeros([M,N],dtype=int)		

		self.BS_UE_connected_set=np.zeros([self.UEs,self.degree_of_multiconnectivity],dtype=int)
		

		self.update_interval_larger_timescale = 1   #1 s update interval for parameters that need not be updated at microsecond granularity: e.g mobility model, channel updates
		
		self.simulation_time=int(simulation_time)  #in seconds
		self.Algorithm=""    #scheduling algorithm chosen

		self.connection_threshold=300  	#in meters - if UE-BS distance is greater than this, link is always in outage
		self.inter_BS_dist=100 #in meters
		self.MapLength=400    # side of square for total simulation area
		self.Distance_Matrix=np.zeros([self.BSs,self.UEs])

		self.UEs_posX_mobility=np.zeros([self.UEs,int(self.simulation_time / self.update_interval_larger_timescale)])
		self.UEs_posY_mobility=np.zeros([self.UEs,int(self.simulation_time / self.update_interval_larger_timescale)])
		
		# self.BS_posX=[0, -86.6,0,86.6,86.6,0,-86.6];
		# self.BS_posY=[0,50,100,50,-50,-100,-50];
		self.BS_posX=[0, -86.6,0,86.6,86.6,0,-86.6, -50, 50, 0, 0];
		self.BS_posY=[0,50,100,50,-50,-100,-50, 0, 0, 50, -50];
		
		fps = 60
		inter_arrival_time_frames = 1 / fps
		self.generation_buffer_size = int(np.ceil((self.simulation_time / self.scheduling_interval)/(inter_arrival_time_frames / self.scheduling_interval)))
		
		self.buffer_size=3
		self.packet_size = 1500  #IP packets of 1500 bytes each
		self.global_frame_size_min = 0    #will be updated later
		self.global_frame_size_max = 0    #will be updated later

				
		self.current_channel_states=np.ones([self.degree_of_multiconnectivity,self.UEs],dtype=int)
		self.current_spectral_efficiency=np.zeros([self.degree_of_multiconnectivity,self.UEs],dtype=float)
		#The following two variables together define the current state of the system
		self.current_channel_capacity=np.ones([self.degree_of_multiconnectivity,self.UEs],dtype=int)			#the C matrix
		self.max_channel_capacity = 0   #packets per scheduling slot; will be updated later
		
		#Stats variables: only deal with gathering some stats about the system
		self.stats_initial_frame_sizes=np.zeros([self.UEs,self.generation_buffer_size],dtype=int)
		self.stats_percentage_frame_served_of_dropped_frames=[]      #captures percentage of frame that had been served when it was dropped
		self.stats_remaining_size_of_expired_frames=array('i')
		self.stats_total_service_time = np.zeros([self.BSs,self.UEs],dtype = int)	
		self.stats_empty_replication_set = np.zeros(self.UEs, dtype = int)
		self.stats_empty_connectivity_set = np.zeros(self.UEs, dtype = int)	
		self.stats_vacancy_in_replication_set = np.zeros(self.UEs, dtype = int)
		self.stats_vacancy_in_connectivity_set = np.zeros(self.UEs, dtype = int)
		self.stats_actual_average_candidacy = np.zeros(self.UEs, dtype = float)
		self.stats_actual_average_connectivity = np.zeros(self.UEs, dtype = float)
		self.stats_actual_average_replication = np.zeros(self.UEs, dtype = float)
		self.stats_wrong_vacancy_in_connectivity_set = np.zeros(self.UEs, dtype = float)
		self.stats_average_outage_duration = np.zeros(self.UEs, dtype = float)
		self.stats_outage_counter = np.zeros(self.UEs, dtype = int)
		self.stats_number_of_outages = np.zeros(self.UEs, dtype = int)
		self.stats_system_total_num_blockages = 0
		self.stats_system_average_blockage_duration = 0

		#Reward Tracking Variables
		self.rew_frame_sizes_at_start_of_timeslot = np.zeros([self.UEs,self.buffer_size],dtype=int)
		self.rew_frame_sizes_at_end_of_timeslot = np.zeros([self.UEs,self.buffer_size],dtype=int)
		self.rew_penalty_for_expired_frames=0
		self.current_frame_IDs = -1*np.ones([self.UEs,self.buffer_size],dtype=int)

		self.data_logging_period = 100        #number of iterations after which to write all stored stats variables to output file
		

		
		
	def Wraparound_Distance(self,x1,y1,x2,y2):
		dx=np.abs(x2-x1)
		dy=np.abs(y2-y1)

		if dx > 0.5*self.MapLength:
			dx=1*self.MapLength - dx 

		if dy > 0.5*self.MapLength:
			dy= 1*self.MapLength- dy 

		return np.sqrt(dx*dx + dy*dy)

	def Generate_Mobility_Trace_MultipleUEs(self):
		#Time = int(self.simulation_time/self.num_topologies)
		
		Time= int(self.simulation_time)  	#UEs only change their position once every second 
		
		MapLength=self.MapLength

		UEs_posX=np.zeros([self.UEs,Time])
		UEs_posY=np.zeros([self.UEs,Time])
		
		for i in range(self.UEs):
			UEs_posX[i,:],UEs_posY[i,:]=self.Generate_Mobility_Trace_SingleUE()
		
		self.UEs_posX_mobility, self.UEs_posY_mobility=UEs_posX,UEs_posY
		
	

	#takes in coordinates of two points, m and n, and applies an exponential filter based on correlation distance and euclidean distance between the points.
	def exponential_filter_2D(self,m,n):
		h=np.exp(-scipy.spatial.distance.euclidean(m,n)/self.CorrelationDistance)
		return h

	def Generate_Mobility_Trace_SingleUE(self):

		MapLength=int(self.MapLength)

		Time= int(self.simulation_time)
		
		while 1:
			#random initial starting point for UE within the grid:
			x_src=np.random.randint(0,MapLength) - MapLength / 2
			y_src=np.random.randint(0,MapLength) - MapLength / 2

			count=0
			#find Distance between source point and all BSs; if out of range of all BSs, generate source point again
			for j in range(self.BSs):
				distance=self.Wraparound_Distance(x_src,y_src,self.BS_posX[j],self.BS_posY[j])
				if distance < self.connection_threshold:
					count=count+1

			if count > 1:
				break	
			# if count > self.degree_of_multiconnectivity:
			# 	break		

		#random destination for UE within the grid:
		x_dst=np.random.randint(0,MapLength) - MapLength / 2
		y_dst=np.random.randint(0,MapLength) - MapLength / 2
		# print("Dest-  X: ",x_dst," ,Y: ",y_dst)
		#choose random velocity for UE between 0 and 3 kmph (0.833 m/s)
		vel_min=0.1
		vel_max=0.833
		vel=np.random.uniform(vel_min,vel_max)
		
		count=0

		x_out=np.zeros(Time)
		y_out=np.zeros(Time)
		x_out[count]=x_src
		y_out[count]=y_src
		count+=1
		direction=np.arctan((np.absolute(y_dst-y_out[count-1]))/(np.absolute(x_dst-x_out[count-1])))
		
		while 1:
			if count == Time:
				break	

			#if vel is 0, UE stays where he is
			# if vel ==0:
			# 	x_out[count]=x_out[count-1]
			# 	y_out[count]=y_out[count-1]
			# 	vel=np.random.uniform(vel_min,vel_max)
	
			#increments for step according to velocity
			y_increment = np.sin(direction)*vel
			x_increment = vel*(np.cos(direction))

			if x_dst < x_out[count-1]:
				x_out[count]=x_out[count-1]-x_increment
			else:
				x_out[count]=x_out[count-1]+x_increment

			if y_dst < y_out[count-1]:	
				y_out[count]=y_out[count-1]-y_increment
			else:
				y_out[count]=y_out[count-1]+y_increment

			# if x_out[count] == x_dst and y_out[count] == y_dst:
			if np.abs(x_out[count] - x_dst) <= x_increment and np.abs(y_out[count] - y_dst) <= y_increment:
				#pick new destinationprint("Current Pos: ")
				# print("Time: ",count)
				# print("Picked New Destination")
				# print("Current Pos- X: ",x_out[count]," ;Y: ",y_out[count])
				# print("Current Dest-  X: ",x_dst," ,Y: ",y_dst)
				x_dst=np.random.randint(0,MapLength) - MapLength / 2
				y_dst=np.random.randint(0,MapLength) - MapLength / 2
				# print("New Dest-  X: ",x_dst," ,Y: ",y_dst)
				#pick new velocity
				vel = np.random.uniform(vel_min,vel_max)
				# direction = np.tan((np.absolute(y_dst-y_out[count-1]))/(np.absolute(x_dst-x_out[count-1])))
				direction = np.arctan((np.absolute(y_dst-y_out[count]))/(np.absolute(x_dst-x_out[count])))
				# print("Direction: ",direction)

			count += 1	
			
		return x_out,y_out

	
	#initialize system
	def initialize_system(self):
		M = self.BSs 
		N = self.UEs
		T = self.simulation_time

		#initialize gNBs
		list_of_gNBs = []
		for i in range(M):
			print("Initializing gNB ",i)
			single_gNB = BS(self,i)
			print("Generating Spatially Correlated LoS/NLoS Map for BS ",i)
			
			single_gNB.Generate_Correlated_LOS_Map()
			#single_gNB.Generate_Correlated_LOS_Map()
			list_of_gNBs.append(single_gNB)		

	
		#initialize UEs
		list_of_UEs=[]
		#Generate Random Walk mobility trace where UEs only change position once every second
		print("Generating Mobility Trace for all UEs for ", T, "seconds, with UEs updating their position once every second")
		self.Generate_Mobility_Trace_MultipleUEs()

		for i in range(N):
			print("Initializing UE ",i)
			single_UE=UE(self,i) 				

			#find out Candidate BSs based on starting position of the UE; Candidate BSs Set will be updated every Tau time-slots
			print("UE ",i)
			single_UE.find_BS_Candidate_Set(0,self)
			print("Candidate BS Set: ",single_UE.BS_Candidate_Set)
		
			#generate arrival slots, packet sizes and packet delay budgets according to 3gpp model for this UE
			single_UE.traffic_model_3gpp(self)
			single_UE.stats_arrival_times=deepcopy(single_UE.arrival_slots_with_jitter)

			#initialize channels of this UE to all Candidate BSs; non-Candidate BSs have channel link capacity set to 0
			for j in range(M):
				single_UE.find_current_link_capacity(0,list_of_gNBs[j],self)
	
			#find BS connected and replicated sets:
			single_UE.find_BS_Connectivity_Set(self)
			print("Connected BS Set: ",single_UE.BS_Connected_Set)
			rs = RandomState(MT19937(SeedSequence(123456789)))
			single_UE.find_BS_Replication_Set(self,np.random.rand(1,self.BSs*self.UEs))
			print("Replicated BS Set: ",single_UE.BS_Replication_Set)

			list_of_UEs.append(single_UE)	

		self.list_of_UEs = list_of_UEs
		self.list_of_gNBs = list_of_gNBs
	

	def insert_arrivals(self,current_time):
		for j in range(self.UEs):			
			current_UE=self.list_of_UEs[j]
			#check for new arrival and append new arrival frame into UE buffer
			if current_UE.arrival_slots_with_jitter[0] == current_time: #means a frame for this UE arrived in this timeslot
				if current_UE.current_buffer_occupancy == 0: 		#buffer for this UE is currently empty
					current_UE.current_buffer_frame_size[0]=current_UE.frame_size[0]
					current_UE.current_buffer_frame_initial_size[0]=current_UE.frame_size[0]
					current_UE.current_deadlines[0]=current_UE.frame_deadlines[0]
					current_UE.current_buffer_occupancy=current_UE.current_buffer_occupancy+1

				else:			#buffer for this UE is non-empty
					index=current_UE.current_buffer_occupancy
					current_UE.current_buffer_frame_size[index]=current_UE.frame_size[0]
					current_UE.current_buffer_frame_initial_size[index]=current_UE.frame_size[0]
					current_UE.current_deadlines[index]=current_UE.frame_deadlines[0]
					current_UE.current_buffer_occupancy=current_UE.current_buffer_occupancy+1
				
				#remove first item in array and left-shift array by 1 slot so 2nd place frame now becomes next incoming frame
				current_UE.frame_size[0]=0
				current_UE.frame_size=np.roll(current_UE.frame_size,-1)
				current_UE.frame_deadlines[0]=0
				current_UE.frame_deadlines=np.roll(current_UE.frame_deadlines,-1)
				current_UE.arrival_slots_with_jitter[0]=0
				current_UE.arrival_slots_with_jitter=np.roll(current_UE.arrival_slots_with_jitter,-1)
			
	#generate new channel states, check for blockages and update the channel states accordingly
	def channel_transitions(self,predictor):
		for j in range(self.UEs):
			current_UE=self.list_of_UEs[j]
			current_UE.connectivity_blockage_started_this_slot = 0
			current_UE.replication_blockage_started_this_slot = 0
			old_channels=deepcopy(current_UE.current_link_capacity)
			#check for blockages
			current_UE.dynamic_blockage_model(self)

			for k in range(self.BSs):
				current_BS=self.list_of_gNBs[k]
				
				if current_UE.blockage_timer[k] > 0:    #this link is in blockage
					# print ("UE ",j,"-BS ",k," link is currently blocked for another ",current_UE.blockage_timer[k],' ms')
					current_UE.stats_in_blockage[k] += 1
					if current_UE.BS_Candidate_Set[k] == 1:   
						current_UE.connectivity_num_handovers_in_progress += 1

					current_UE.blockage_timer[k] -= 1
					current_UE.current_link_capacity[k]=-1

					#if there is a blockage, immediately drop the BS from the Connectivity set and initiate handover
					if current_UE.BS_Connected_Set[k] == 1:
						current_UE.connectivity_blockage_started_this_slot = 1
						current_UE.replication_blockage_started_this_slot = 1
						# print("UE ",j," and BS ",k," blockage started this slot")
						current_UE.BS_Candidate_Set[k] = 0
						current_UE.BS_Connected_Set[k] = 0
						current_UE.BS_Replication_Set[k] = 0	
						# current_UE.find_BS_Candidate_Set(self.current_time,self)
						# current_UE.find_BS_Connectivity_Set(self)
						# current_UE.find_BS_Replication_Set(self,predictor)	

				#only update CQI every Larger Timescale Update Interval (set to 1 ms for now, instead of 125 us scheduling interval)
				if self.current_time % (1/self.update_interval_larger_timescale) == 0: 
					current_UE.find_current_link_capacity(self.current_time,current_BS,self)

				if current_UE.current_link_capacity[k] > 0:
					current_UE.stats_avg_link_capacity[k]=(old_channels[k]+current_UE.current_link_capacity[k])

			#update candidate,connected and replication sets
			if (self.current_time > current_UE.BS_connectivity_handover_delay): # and (self.current_time*self.scheduling_interval) % 1 == 0) or np.sum(current_UE.BS_Replication_Set) == 0:
				current_UE.find_BS_Candidate_Set(self.current_time,self)
				current_UE.find_BS_Connectivity_Set(self)
				current_UE.find_BS_Replication_Set(self,predictor)

	#Given an action, this function executes it by serving the appropriate packets, dropping expired packets, and then filling the buffers again with new packets	
	#Returns the matrix of new deadlines after the action has been executed
	def traffic_transitions(self,action):

		current_action=action
		reward = 0
		#print("Current Packet Sizes: ",self.UE_current_buffer_packet_size)
		
		#Serving packets according to the current action and channel conditions
		for i in range(self.BSs):
			for j in range(self.UEs):
				current_UE=self.list_of_UEs[j]
				current_BS_index=i

				if current_UE.BS_Replication_Set[current_BS_index] == 0:
					continue

				if current_action[current_BS_index,j] == 1:

					if current_BS_index == -1:
						#print("BS not in Replication Set chosen in Action")
						continue
						
					if current_UE.current_buffer_occupancy==0:
						#print("UE ",j," currently has an empty buffer ")
						continue

					max_servable=current_UE.current_link_capacity[current_BS_index]						   #max number of packets that can be served during this timeslot on this UE-BS link 
					#print("Max number of packets that can be served: ",max_servable)
					if current_UE.current_buffer_occupancy > 0:   #if there is atleast one frame in buffer with some packets still left to send
						while 1:
							
							if max_servable == 0 or current_UE.current_buffer_occupancy == 0:
								break

							num_packets_in_this_frame= math.ceil(current_UE.current_buffer_frame_size[0]/self.packet_size)       #number of remaining packets in this frame
							# print("BS ",current_BS_index," is serving UE ",j,"; total capacity is ",max_servable," packets")
							if max_servable <= num_packets_in_this_frame:     #frame can only be partially delivered during this timeslot
								#print("       ",max_servable," packets from frame ",current_UE.frame_currently_being_served," will be served")
								current_UE.current_buffer_frame_size[0]=current_UE.current_buffer_frame_size[0]-(max_servable*self.packet_size)			#deliver part of packet, and update remaining size
								current_UE.frame_status[0]=2											#mark packet as partially delivered
								current_UE.stats_service_from_BS[current_BS_index]=current_UE.stats_service_from_BS[current_BS_index]+max_servable
								current_UE.current_frames_fraction_served += (max_servable*self.packet_size)/current_UE.current_buffer_frame_initial_size[0]  
								reward += current_UE.current_frames_fraction_served
								current_UE.stats_total_bytes_served += max_servable*self.packet_size	
								max_servable = 0	
							elif max_servable > num_packets_in_this_frame:	  #frame will be fully served, and will have to move on to partially serve next frame as well	
								#print("       ",num_packets_in_this_frame," packets from frame ",current_UE.frame_currently_being_served," will be served")
								max_servable=max_servable-num_packets_in_this_frame
								current_UE.stats_total_bytes_served += num_packets_in_this_frame * self.packet_size
								current_UE.num_frames_served=current_UE.num_frames_served+1
								current_UE.frame_currently_being_served=current_UE.frame_currently_being_served+1
								current_UE.time_served_or_expired[current_UE.frame_currently_being_served]=self.current_time

								current_UE.current_frames_fraction_served += (num_packets_in_this_frame*self.packet_size)/current_UE.current_buffer_frame_initial_size[0]  
								reward += current_UE.current_frames_fraction_served
								current_UE.stats_service_from_BS[current_BS_index]=current_UE.stats_service_from_BS[current_BS_index]+num_packets_in_this_frame
								# print("Frame of UE ",j," fully served; fraction served: ",current_UE.current_frames_fraction_served)

								#remove first frame from buffer as it is fully delivered
								current_UE.current_buffer_frame_size[0]=0
								current_UE.current_buffer_frame_size=np.roll(current_UE.current_buffer_frame_size,-1)
								current_UE.current_buffer_frame_initial_size[0] = 0
								current_UE.current_buffer_frame_initial_size =  np.roll(current_UE.current_buffer_frame_initial_size, -1)
								current_UE.current_deadlines[0]=10000;
								current_UE.current_deadlines=np.roll(current_UE.current_deadlines,-1)
								current_UE.current_buffer_occupancy=current_UE.current_buffer_occupancy-1
								current_UE.current_frames_fraction_served = 0
								
		#Decrementing deadlines of remaining packets and dropping expired packets
		for i in range(self.UEs):
			current_UE=self.list_of_UEs[i]
			temp_index=current_UE.frame_currently_being_served
			
			for j in range(current_UE.current_buffer_occupancy):
				if current_UE.current_deadlines[j] != 10000:
					#decrementing deadlines of remaining packets
					current_UE.current_deadlines[j]=current_UE.current_deadlines[j]-1
					
					current_UE.frames_time_spent_in_buffer[temp_index+j]=current_UE.frames_time_spent_in_buffer[temp_index+j]+1
					# print("Time Spent in Buffer for Frame ",current_UE.frame_currently_being_served," of UE ",current_UE.index," is ",current_UE.frames_time_spent_in_buffer[current_UE.frame_currently_being_served])
				
				if current_UE.current_deadlines[j]<=0:			#drop expired packets
					
					# print("Frame ",current_UE.frame_currently_being_served," of UE ",current_UE.index," has expired ")
					current_UE.time_served_or_expired[current_UE.frame_currently_being_served]=self.current_time
					current_UE.num_frames_expired=current_UE.num_frames_expired+1
					current_UE.frame_currently_being_served=current_UE.frame_currently_being_served+1
					
					#Stats: store how much of the frame was remaining to be served when expired 
					current_UE.frame_size_at_expiry.append(current_UE.current_buffer_frame_size[0])
					penalty = (1+current_UE.current_frames_fraction_served)
					current_UE.stats_fraction_served_of_expired_frames += current_UE.current_frames_fraction_served
					
					reward -= penalty
					
					#remove frame from the buffer
					current_UE.current_buffer_frame_size[0]=0
					current_UE.current_buffer_frame_size=np.roll(current_UE.current_buffer_frame_size,-1)
					current_UE.current_buffer_frame_initial_size[0] = 0
					current_UE.current_buffer_frame_initial_size =  np.roll(current_UE.current_buffer_frame_initial_size, -1)			
					current_UE.current_deadlines[0]=10000;
					current_UE.current_deadlines=np.roll(current_UE.current_deadlines,-1)
					current_UE.current_buffer_occupancy=current_UE.current_buffer_occupancy-1
					current_UE.current_frames_fraction_served = 0

		return reward

	def state_generator(self,module):				#Embed state variables to create the State for Deep Learning; module can be "Scheduler" or "Predictor"

		state = []
		Replication_Sets = []
		Connectivity_Sets = []
		Channel_Sets = []
		Current_Frame_Sizes = []
		Current_Deadlines = []

		if module.lower() != "scheduler" and module.lower() != "predictor":
			print("ERROR: Please input correct module")
			return 0
		# elif module.lower() == "scheduler":
		# 	print("Generating State Embedding Variable for Scheduler")
		# elif module.lower() == "predictor":
		# 	print("Generating State Embedding Variable for Predictor")


		for i in range(self.UEs):
			current_UE = self.list_of_UEs[i]

			if module.lower() == "scheduler":
				#State contains the BS Replication Set for all UEs
				Replication_Sets = np.concatenate((Replication_Sets,current_UE.BS_Replication_Set),axis=0)
			elif module.lower() == "predictor":
				Connectivity_Sets = np.concatenate((Connectivity_Sets,current_UE.BS_Connected_Set),axis=0)
				

			#State contains all UE-gNB Channel State Information matrices
			Channel_Sets = np.concatenate((Channel_Sets,current_UE.current_link_capacity),axis=0)

			#State conatins all UE traffic information: current sizes of frames in buffer, and their deadlines; since there can be a variable number of frames
			#in buffer, it is padded to length 5. Padded frames have size 0 and deadline 10000
			buffer = 3
			temp_frame_sizes = np.zeros(buffer)
			if current_UE.current_buffer_occupancy > 0:
				temp_frame_sizes[0:current_UE.current_buffer_occupancy]=current_UE.current_buffer_frame_size[0:current_UE.current_buffer_occupancy]
			Current_Frame_Sizes = np.concatenate((Current_Frame_Sizes,temp_frame_sizes),axis=0)

			temp_current_deadlines = 100*np.ones(buffer)
			if current_UE.current_buffer_occupancy > 0:
				temp_current_deadlines[0:current_UE.current_buffer_occupancy] = current_UE.current_deadlines[0:current_UE.current_buffer_occupancy]
			Current_Deadlines = np.concatenate((Current_Deadlines,temp_current_deadlines),axis=0)

		## Normalizing all the features before adding them to the state ##
		
		#Normalizing Connectivity and Replication Sets
		connectivity_min = -1
		connectivity_max = self.BSs - 1
		Connectivity_Sets = [(i - connectivity_min)/(connectivity_max - connectivity_min) for i in Connectivity_Sets]
		Replication_Sets = [(i - connectivity_min)/(connectivity_max - connectivity_min) for i in Replication_Sets]

		#Normalizing Frame Sizes
		normalized_frame_sizes = [(i - self.global_frame_size_min)/(self.global_frame_size_max - self.global_frame_size_min) for i in Current_Frame_Sizes]
		Current_Frame_Sizes = normalized_frame_sizes

		#Normalizing Deadline Matrix
		deadline_max = 100   #artifical padding max 
		deadline_min = 0 
		Current_Deadlines = [(i - deadline_min)/(deadline_max - deadline_min) for i in Current_Deadlines]
	
		#Normalizing Channel States
		channel_min = 0 
		Channel_Sets = [(i - channel_min)/(self.max_channel_capacity - channel_min) for i in Channel_Sets] 
		
		
		if module.lower() == "scheduler":
			state = np.concatenate((state,Replication_Sets,Channel_Sets,Current_Frame_Sizes,Current_Deadlines),axis=0)
		elif module.lower() == "predictor":
			state = np.concatenate((state,Connectivity_Sets,Channel_Sets,Current_Frame_Sizes,Current_Deadlines),axis=0)
		
		return state

	def pre_process_data(self,array):	#standardize the state variable so that it is a standard Gaussian (zero mean, sigma std dev)
		mean=np.mean(array)
		stddev=np.std(array)
		array = (array - mean)/stddev
		return array


	def generate_random_action(self, action_prob_dist):			#Randomly generate a single action that satisfies all the feasibility constraints 
		N = self.UEs
		M = self.BSs 
		action = np.zeros([M,N],dtype=int)
		replication_set = deepcopy(self.bitwise_replication_set)
		BS_order = np.random.permutation(M) #M dimensions - select BS ordering randomly
		
		prob_this_action = 1
		for BS_chosen in BS_order:
			potential_UEs= np.asarray(np.argwhere(replication_set[BS_chosen,:]))
			if len(potential_UEs) == 0:
				continue
			random_selection_index = np.random.randint(len(potential_UEs))
			UE_selected = int(potential_UEs[random_selection_index])
			# print("Bitwise Replication Set:",replication_set[:,UE_selected])
			# print("UE Replication Set:",self.list_of_UEs[UE_selected].BS_Replication_Set)
			# print("")
			action[BS_chosen,UE_selected] = 1
			replication_set[:,UE_selected] = 0

			#find the probability of this action according to the Proposal Distribution
			#the prob dist for this sub-action; comes from the output of Proposal DQN: 1 x N vector for each BS
			prob_dist = action_prob_dist[BS_chosen*N:(BS_chosen+1)*N]
			prob_dist = abs(prob_dist)   #make sure all probability values are positive
			prob_dist = prob_dist / np.sum(prob_dist)   #make sure probabilities sum to 1
			
			#Also assign 0 probability to actions where BS is not in UE Replication Set
			for l in range(self.UEs):
				current_UE = self.list_of_UEs[l]
				if BS_chosen not in current_UE.BS_Replication_Set:
					prob_dist[current_UE.index] = 0 

			#adjust prob dist according to previous chosen sub_actions:
			for k in range(M):	
				previous_action_inverted = ~action[k,:]+2
				if np.sum(previous_action_inverted) == 2 * N or k == BS_chosen or np.amax(action[k,:]) < 1:
					continue						
				zero_index = int(np.asarray(np.where(previous_action_inverted == 0)))
				prob_dist[zero_index]=0

			if np.sum(prob_dist) != 0:
				prob_dist = prob_dist / np.sum(prob_dist)
				prob_dist = prob_dist / np.sum(prob_dist)
				prob_this_action *= prob_dist[UE_selected]

		action = action.flatten()
		return action, prob_this_action

	def given_action_find_probability(self,action,action_prob_dist):
		'''Given an action and the autoregressive probability distributions, find the overall probability of that action by summing up over
		all the probabilities for different orderings that lead to that action '''
		
		M = self.BSs 
		N = self.UEs 
		action = np.reshape(action,(M,N))

		num_orderings = np.math.factorial(M)     #number of orderings that will lead to the same action
		set_of_orderings = list(permutations(range(M)))
		prob_action = 1
		total_prob = 0

		for i in range(num_orderings):
			current_ordering = set_of_orderings[i]
			temp_action_prob_dist = deepcopy(action_prob_dist)
			temp_action_prob_dist = np.reshape(temp_action_prob_dist,(M,N))
			
			for BS_chosen in current_ordering:
				#the prob dist for this sub-action; comes from the output of Proposal DQN: 1 x N vector for each BS
				prob_dist = temp_action_prob_dist[BS_chosen,:]
				prob_dist = abs(prob_dist)   #make sure all probability values are positive
				prob_dist = prob_dist / np.sum(prob_dist)   #make sure probabilities sum to 1

				#Also assign 0 probability to actions where BS is not in UE Replication Set
				for l in range(N):
					current_UE = self.list_of_UEs[l]
					if BS_chosen not in current_UE.BS_Replication_Set:
						prob_dist[current_UE.index] = 0 


				#Also assign 0 prob to every row and column where this Action is 1
				for l in range(N):
					if action[BS_chosen,l] == 1:
						current_ind = np.asarray(np.where(np.array(current_ordering) == BS_chosen))[0]
						current_ind = int(current_ind)
						for k in range(M):
							if k > current_ind:
								BS = current_ordering[k]
								temp_action_prob_dist[BS,l] = 0	

				
				if np.sum(prob_dist) != 0:
						prob_dist = prob_dist / np.sum(prob_dist)
						for l in range(N):
							if action[BS_chosen,l] == 1:
								prob_action *= prob_dist[l]

				else:
					prob_action = 0
			total_prob += prob_action
		
		return total_prob
	
	def check_strong_feasibility_action(self,action):
		if np.sum(np.bitwise_and(action,self.bitwise_replication_set)) <= self.BSs:
			return 1
		else:
			return 0

	
	def proposed_actions(self,A,B, action_prob_dist):   		 #draw A random samples, and B samples from the proposal distribution given by Proposal DQN, total A+B actions 	
		M=self.BSs 
		N=self.UEs

		action=[]					#action =[A random actions, B proposed actions] 
		action_subset_probs =[]		#probabilities of choosing the actions in the action subset according to the Proposal Distribution
		# action_indices = []			#Indices of the random actions
		
		
		#draw A random action samples
		for i in range(A):	
			
			while 1:
				#print(i, "Random Actions Generated")
				random_action, prob_this_action=self.generate_random_action(action_prob_dist)
			
				duplicate = 0
				for comp in action:
					if np.array_equal(random_action,comp):
						#print("Duplicate Action Generated!!")
						duplicate = 1
						break

				if duplicate == 0:
					action.append(random_action)
					action_subset_probs.append(prob_this_action)
					break
		
		action_prob = np.ones(B, dtype = float)
		#sample B action samples from the proposal distribution
		for i in range(B):
			

			while 1:
				#print(i," Actions drawn from Proposal")
				temp_action = np.zeros([M,N],dtype=int)
				BS_order = np.random.permutation(M) #M dimensions - select BS ordering randomly
				
				for BS_chosen in BS_order:
					# prob_dist = np.ones(N) / N    #assume for now that each probability distribution is uniform in the beginning
					BS = self.list_of_gNBs[BS_chosen]
					
					#the prob dist for this sub-action; comes from the output of Proposal DQN: 1 x N vector for each BS
					
					prob_dist = action_prob_dist[BS_chosen*N:(BS_chosen+1)*N]
					prob_dist = abs(prob_dist)   #make sure all probability values are positive
					prob_dist = prob_dist / np.sum(prob_dist)   #make sure probabilities sum to 1
					
					#Also assign 0 probability to actions where BS is not in UE Replication Set
					for l in range(N):
						current_UE = self.list_of_UEs[l]
						if BS_chosen not in current_UE.BS_Replication_Set:
							prob_dist[current_UE.index] = 0 
				
					#adjust prob dist according to previous chosen sub_actions:
					for k in range(M):	
						previous_action_inverted = ~temp_action[k,:]+2
						if np.sum(previous_action_inverted) == 2 * N or k == BS_chosen or np.amax(temp_action[k,:]) < 1:
							continue						
						zero_index = int(np.asarray(np.where(previous_action_inverted == 0)))
						prob_dist[zero_index]=0

					# print("Prob Dist: ",prob_dist)
					if np.sum(prob_dist) != 0:
						prob_dist = prob_dist / np.sum(prob_dist)

						#randomly sample from this BS's proposal distribution, given we know the samples of the previous BSs in this iteration
						choices = np.arange(0,N,1,dtype=int)
						random_sample = np.random.choice(choices,p = prob_dist)
						temp_action[BS_chosen,:] = 0
						temp_action[BS_chosen,random_sample] = 1
						action_prob[i] *= prob_dist[random_sample]
					else:
						temp_action[BS_chosen,:] = 0


				duplicate = 0
				#discard this action if its already in the action proposal
				for comp in action:
					if np.array_equal(temp_action,comp):
						#print("Duplicate Proposed Action!!")
						duplicate = 1
						action_prob[i] = 1
						break
				
				if duplicate == 1:
					continue

				if self.check_strong_feasibility_action(temp_action) == 1:
					temp_action = temp_action.flatten()
					action.append(temp_action)
					action_subset_probs.append(action_prob[i])
					break
				
		return action, action_subset_probs    #returns the subset of A+B actions and their probabilities

		

	#--------------------------------------------------------------------------------------#
	####        AMORTIZED DEEP Q-LEARNING (ADQL) SCHEDULING ALGORITHM                   ####
	#--------------------------------------------------------------------------------------#
	def Amortized_Deep_Q_Learning(self, state, proposal_agent, scheduler_agent , A, B):
		M = self.BSs 
		N = self.UEs 

		proposal_state_size=len(state)
		state = np.reshape(state, [1, proposal_state_size])
		
		proposal_agent.steps_to_update_target_model += 1
		
		#The output of the neural network gives the probability distribution of the autoregressive proposed actions
		# action_prob_dist = proposal_agent.model.predict(state).flatten()
		action_prob_dist = proposal_agent.model(state).numpy().flatten()
		
		action_subset, proposal_action_probs = self.proposed_actions(A,B,action_prob_dist)
		action_subset = np.array(action_subset).flatten()

		# #create the Scheduler State = Proposal State + Proposal Action Subset
		scheduler_state = state
		scheduler_state = np.append(scheduler_state,action_subset)
		
		scheduler_state_size = len(scheduler_state)
		scheduler_state_with_proposed_action_subset = np.reshape(scheduler_state,[1, scheduler_state_size])

		#This subset of "proposed" actions will now be passed on to Schedular Neural Network, which will pick the best action from this subset
		random_number = np.random.rand()
		
		predicted = scheduler_agent.model(scheduler_state_with_proposed_action_subset).numpy().flatten()
		max_ind = np.argmax(predicted)
		if random_number <= scheduler_agent.epsilon:
            # Explore
			# print("Exploring")
			ind = np.random.randint(len(action_subset)/(M*N))
			chosen_action = action_subset[ind*M*N:(ind+1)*M*N]
		else:
            # Exploit best known action	
			# print("Exploiting")
			chosen_action = action_subset[max_ind*M*N:(max_ind+1)*M*N]

		#proposal_agent.proposal_loss_func(action_subset[max_ind],self)

		return chosen_action, action_subset, max_ind, proposal_action_probs, action_prob_dist

	## Maximal Weight Matching (MWM) heuristic scheduling 
	def MWM_Heuristic(self,beta):
		M = self.BSs
		N = self.UEs
		L = self.degree_of_replication
		action=np.zeros([self.BSs,self.UEs],dtype=int)

		deadline_max = int((15*10**-3)/self.scheduling_interval)
		max_channel_capacity = self.max_channel_capacity

		G = nx.Graph()
		UEs = ["n"+str(i) for i in range(N)]
		gNBs = ["m"+str(i) for i in range(M)]
		G.add_nodes_from(UEs, bipartite = 0)
		G.add_nodes_from(gNBs, bipartite = 1)

		top_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
		bottom_nodes = set(G) - top_nodes
		edges = []
		for i in range(M):
			for j in range(N):
				current_UE = self.list_of_UEs[j]
				if current_UE.BS_Replication_Set[i] == 0 or current_UE.current_deadlines[0] == 10000:
					w = 0
					edges.append((gNBs[i],UEs[j],{'weight': w}))
					continue
				else: 
					# print("UE: ",j," ; BS: ",i)
					L_eff = np.sum(current_UE.BS_Replication_Set)
					# print("L: ",L_eff)
					# print("Deadline: ",current_UE.current_deadlines[0])
					# print("Channel: ", current_UE.current_link_capacity[i])
					# print("Max Channel: ",max_channel_capacity)
					w = (beta / current_UE.current_deadlines[0] ) + (1 - beta) * (current_UE.current_link_capacity[i] / L_eff)  #weight1
					w = (1 / L_eff) * ((current_UE.current_link_capacity[i] ** (1 - beta)) / (current_UE.current_deadlines[0] ** beta))   #weight2

					# print("Weight: ", w)
					# print("")
					edges.append((gNBs[i],UEs[j],{'weight': w}))
		G.add_edges_from(edges)
		
		maximal_match = sorted(nx.max_weight_matching(G))
		for i in range(len(maximal_match)):
			if maximal_match[i][0][0] == 'm':
				BS_ind = 0
				UE_ind = 1 
			else: 
				BS_ind = 1
				UE_ind = 0
			BS_chosen = int(maximal_match[i][BS_ind][1:])
			UE_chosen = int(maximal_match[i][UE_ind][1:])
			if self.list_of_UEs[UE_chosen].BS_Replication_Set[BS_chosen] == 0:
				print("ERROR")
				continue
			action[BS_chosen, UE_chosen] = 1
	
		return action

	#--------------------------------------------------------------------------------------#
	####        EARLIEST DEADLINE FIRST (EDF) SCHEDULING ALGORITHM                      ####
	#--------------------------------------------------------------------------------------#
	#Return an action that corresponds to serving the UE with earliest deadline from its best BS. Each BS can serve only one UE.
	def Earliest_Deadline_First(self):	
		action=np.zeros([self.BSs,self.UEs],dtype=int)
		BS_busy=np.zeros(self.BSs,dtype=int)
		UE_busy=np.zeros(self.UEs,dtype=int)
		
		current_deadlines=np.zeros([self.UEs,self.buffer_size],dtype=int)
		current_chan=-1*np.ones([self.BSs,self.UEs],dtype=int)

		for i in range(self.UEs):
			current_UE=self.list_of_UEs[i]
			#print("Current link capacity of this UE: ",current_UE.current_link_capacity)
			current_deadlines[i,:]=current_UE.current_deadlines
			for j in range(self.BSs):
				if current_UE.BS_Replication_Set[j] == 1:
					current_chan[j,i]=current_UE.current_link_capacity[j]

		while 1:	
			if current_deadlines.min() == 10000:
				# print("No remaining UE to serve feasibly")
				#print(current_deadlines)
				break


			#break loop if all BSs busy or all UEs busy
			if np.sum(BS_busy) == self.BSs or np.sum(UE_busy)==self.UEs:
				break

			#break loop if all channels are in outage
			if np.amax(current_chan) == 0:
				# print("All Channels are in Outage")
				break
			#print("Current Deadlines: \n",current_deadlines)
			
			min_ind=np.where(current_deadlines==np.min(current_deadlines))  #can be multiple indices where minimum occurs
			min_ind=np.asarray(min_ind)
			min_ind=min_ind[0,:]
			
			#if multiple indices, choose one UE randomly
			if len(min_ind)>1:
				ind=np.random.randint(0,len(min_ind))
				UE_chosen=min_ind[ind]
			else:
				UE_chosen=min_ind

			UE_chosen=int(UE_chosen)
			if UE_busy[UE_chosen]==1 or np.sum(current_chan[:,UE_chosen])==0:
				current_deadlines[UE_chosen,:]=10000  #indicates this UE has been chosen before already or that it has no channel available to any BS
				# print("This UE has already been chosen before or doesnt have a channel available to any free BS")
				continue
			
			max_BS=np.where(current_chan[:,UE_chosen]==np.max(current_chan[:,UE_chosen]))
			max_BS=np.asarray(max_BS)
			max_BS=max_BS[0,:]
			
			if len(max_BS)>1:
				ind=np.random.randint(0,len(max_BS))
				BS_chosen=max_BS[ind]
			else:
				BS_chosen=max_BS
			
			BS_chosen=int(BS_chosen)

			current_UE=self.list_of_UEs[UE_chosen]
			
			BS_index=BS_chosen

			if BS_busy[BS_index]==1:
				current_chan[BS_chosen,UE_chosen]= -1
				#print("Chosen BS is already busy serving another UE")
				continue

			if current_chan[BS_chosen,UE_chosen]==0:
				current_deadlines[UE_chosen]=10000  
				# print("Chosen BS has no link with the Chosen UE")
				continue

			action[BS_index,UE_chosen]=1	
			BS_busy[BS_index]=1
			UE_busy[UE_chosen]=1
			current_deadlines[UE_chosen,:]=10000

		# print("EDF Action: ")
		# print(action)
		# print("")
		return action
	

	def Other_Scheduling_Algorithms(self, alpha, beta):	
		action=np.zeros([self.BSs,self.UEs],dtype=int)
		BS_busy=np.zeros(self.BSs,dtype=int)
		UE_busy=np.zeros(self.UEs,dtype=int)
		
		current_deadlines=np.zeros([self.UEs,self.buffer_size],dtype=int)
		current_chan=-1*np.ones([self.BSs,self.UEs],dtype=int)

		P = np.zeros([self.UEs,self.BSs],dtype=float)
		UEs_previous_avg_throughput = np.zeros(self.UEs, dtype = float)

		for i in range(self.UEs):
			current_UE = self.list_of_UEs[i]
			if self.current_time == 0 :
				UEs_previous_avg_throughput[i] = 0
			else:
				UEs_previous_avg_throughput[i] = (current_UE.stats_total_bytes_served / (self.current_time*self.scheduling_interval))*8/(10**6)   #in Mbps

			#if buffer empty, skip this UE
			if current_UE.current_buffer_occupancy == 0:
				P[i,:] = 0
				continue 

			for j in range(self.BSs):
				if current_UE.BS_Replication_Set[j] == 0 or current_UE.current_buffer_occupancy == 0:
					P[i,j] = 0
					continue

				if current_UE.current_link_capacity[j] > 0:
					current_link_capacity = current_UE.current_link_capacity[j]*self.packet_size*(1/self.scheduling_interval)*8/(10**6)    #in Mbps
				else:
					current_link_capacity = 0
				
				if UEs_previous_avg_throughput[i] == 0:
					P[i,j] = 100000
				else:
					P[i,j] = (current_link_capacity**alpha) / (UEs_previous_avg_throughput[i]**beta)

		
		while 1:
			if np.amax(P) == 0:
				break

			max_ind = np.asarray(np.where(P == np.max(P)))
			prospective_UE = max_ind[0]
			prospective_BS = max_ind[1]

			if len(prospective_UE)>1:
				ind=np.random.randint(0,len(prospective_UE))
				chosen_UE=prospective_UE[ind]
				chosen_BS=prospective_BS[ind]
			else:
				chosen_UE = prospective_UE 
				chosen_BS = prospective_BS
			
			action[chosen_BS, chosen_UE] = 1
			P[chosen_UE,:] = 0 
			P[:,chosen_BS] = 0

		return action



class UE(Environment):
	def __init__(self, env, index):
		self.index=index
		self.BS_Candidate_Set=-1*np.ones(env.BSs,dtype=int)					#C: the set of BSs in range of this UE
		self.BS_Connected_Set=np.zeros(env.BSs,dtype=int) 					#K: the set of BSs connected to this UE; K is a subset of C
		self.BS_Replication_Set=np.zeros(env.BSs,dtype=int) 					#L:	the set of BSs who have this UE's data buffered and ready for servin; L is a subset of K
		self.generation_buffer_size=env.generation_buffer_size 			#size of generation buffer in number of frames
		self.buffer_size=env.buffer_size 					#size of actual UE buffers
		self.Distance_Matrix=np.zeros(env.BSs)		#current distance between this UE and all BSs; updated whenever BS_Candidate_Set is updated
		self.current_link_capacity=np.zeros(env.BSs)		#capacity of the links between this UE and all BSs
		self.UE_height=1.5 			#height of UE in meters
		
		self.BS_connectivity_handover_delay=env.connectivity_handover_delay   #ms
		self.BS_replication_handover_delay=env.replication_handover_delay 	#ms
		self.dynamic_blocker_density = env.dynamic_blocker_density
		
		self.connectivity_handover_timer_countdown = -1*np.ones(env.BSs,dtype=int)
		self.connectivity_num_handovers_in_progress = 0
		self.connectivity_blockage_started_this_slot = 0
		self.connectivity_handovers_to = -1*np.ones(env.degree_of_multiconnectivity, dtype = int)
		self.connectivity_handovers_from = -1*np.ones(env.degree_of_multiconnectivity, dtype = int)

		self.replication_handover_timer_countdown = -1*np.ones(env.BSs,dtype=int)
		self.replication_num_handovers_in_progress = 0
		self.replication_blockage_started_this_slot = 0
		self.replication_handovers_to = -1*np.ones(env.degree_of_replication, dtype = int)
		self.replication_handovers_from = -1*np.ones(env.degree_of_replication, dtype = int)
		self.time_in_blockage = np.zeros(env.BSs, dtype = int)

		

		#blockage model variables
		self.time_of_last_event=np.zeros(env.BSs,dtype=int)
		self.wait_till_next_event=np.zeros(env.BSs,dtype=int)
		self.time_till_next_event=np.zeros(env.BSs,dtype=int)
		self.blockage_duration=np.zeros(env.BSs,dtype=int)
		self.blockage_time=np.zeros(env.BSs,dtype=int)	  #denotes the time when link when next be in blockage

		self.blockage_timer=np.zeros(env.BSs,dtype=int)   #denotes remaining time a link will be in blockage; if 0, means link is un-blocked
		self.Theta=np.zeros(env.BSs,dtype=float)
		self.alpha=np.zeros(env.BSs,dtype=float)
		

		#3GPP Traffic Model Parameters
		# [Virtual Reality, Augmented Reality, Cloud Gaming]
		self.traffic_labels=["VR","AR","CG"]
		self.data_rate=[45, 45, 30]      #in Mbps
		self.frame_rate=[60, 60, 60]      #in fps
		self.packet_delay_budget=[10, 10, 15]     #in ms

		#assigning random traffic type to each UE: 0 = VR, 1= AR, 2 = CG
		self.UE_traffic_type=randint(0,2)   
		
		self.arrival_slots_without_jitter=np.zeros(self.generation_buffer_size,dtype=int)
		self.arrival_slots_with_jitter=np.zeros(self.generation_buffer_size,dtype=int)
		self.frame_size=np.zeros(self.generation_buffer_size,dtype=int)
		self.frame_deadlines=np.zeros(self.generation_buffer_size,dtype=int)
		self.frame_currently_being_served=0   #ID of frame currently being served, incremented after a frame is served completely or dropped at expiry

		#traffic type determines PDB for the UE's traffic
		self.UE_traffic_PDB=0
		if self.UE_traffic_type==0:
			self.UE_traffic_PDB=int((self.packet_delay_budget[0]*10**-3)/env.scheduling_interval)   #in no. of scheduling time slots
		elif self.UE_traffic_type==1:
			self.UE_traffic_PDB = int((self.packet_delay_budget[1]*10**-3)/env.scheduling_interval)   #in no. of scheduling time slots
		elif self.UE_traffic_type==2:
			self.UE_traffic_PDB = int((self.packet_delay_budget[2]*10**-3)/env.scheduling_interval)   #in no. of scheduling time slots
		else:
			print("Traffic Type not assigned to UE")

		#these are the variables which track the current state of this UE's traffic
		self.current_buffer_occupancy=0			#denotes the number of frames currently in the UEs' buffer
		self.current_buffer_frame_size=np.zeros(self.buffer_size,dtype=int)
		self.current_buffer_frame_initial_size = np.zeros(self.buffer_size,dtype=int)
		self.current_frames_fraction_served = 0
		self.current_deadlines=10000*np.ones(self.buffer_size,dtype=int)		#the D matrix


		# interarrival_time=math.ceil(1/60*1000)    #in ms - temporarily hardcoded with 60 fps
		# num_total_arrivals=math.floor(env.simulation_time/interarrival_time)*env.UEs
		# mean_incoming_size=np.sum(self.frame_size)/num_total_arrivals


		#stats tracking variables
		self.stats_num_total_arrivals=0
		self.frames_time_spent_in_buffer=np.zeros(self.generation_buffer_size,dtype=int)   #if equal to frame deadline, it means it expired
		self.frame_status=np.zeros(self.generation_buffer_size,dtype=int)		#1: unserved, 2: partially served, 3: fully served
		self.frame_size_at_expiry=[]  #if 0, it means the frame was served and didnt expire
		self.num_frames_served=0
		self.num_frames_expired=0
		self.stats_total_bytes_served = 0
		self.time_served_or_expired=np.zeros(self.generation_buffer_size,dtype=int)
		self.stats_arrival_times=[]
		self.stats_avg_link_capacity=np.zeros(env.BSs,dtype=int)
		self.stats_max_link_capacity=np.zeros(env.BSs,dtype=int)
		self.stats_min_link_capacity=10000*np.ones(env.BSs,dtype=int)
		self.stats_number_of_handovers_in_connectivity_set=0
		self.stats_number_of_handovers_in_replication_set=0
		self.stats_avg_time_spent_in_buffer=0
		self.stats_number_of_timeslots_in_candidate_set = np.zeros(env.BSs,dtype=int)
		self.stats_number_of_timeslots_in_connectivity_set=np.zeros(env.BSs,dtype=int)
		self.stats_number_of_timeslots_in_replication_set=np.zeros(env.BSs,dtype=int)
		self.stats_number_of_timeslots_with_no_coverage = 0

		self.stats_service_from_BS=np.zeros(env.BSs,dtype=float)		#tracks the amount of service (in number of packets) received from each BS
		self.stats_in_blockage=np.zeros(env.BSs,dtype=int)
		self.stats_avg_number_of_gNBs_in_Candidate_Set=0
		self.stats_avg_number_of_gNBs_in_Connectivity_Set=0
		self.stats_avg_number_of_gNBs_in_Replication_Set=0
		self.stats_blockage_arrivals=[]
		self.stats_blockage_durations=[]
		self.stats_fraction_served_of_expired_frames = 0
		self.stats_blockage_count = 0
		

	def traffic_model_3gpp(self, env):
		
		duration=int(env.simulation_time/env.scheduling_interval)
		#Packet Arrivals: interarrival times are periodic with a superimposed jitter that follows a truncated Gaussian distribution
		
		#truncated Gaussian Distribution parameters for jitter
		jitter_mean=0           #ms
		jitter_STD=int((2*10**-3)/env.scheduling_interval)            #2 ms, in no. of scheduling time-slots
		jitter_min=int((-4*10**-3)/env.scheduling_interval)            #-4 ms, in no. of scheduling time-slots
		jitter_max=int((4*10**-3)/env.scheduling_interval)            #4 ms, in no. of scheduling time-slots
		
		jitter_a= (jitter_min-jitter_mean)/jitter_STD
		jitter_b= (jitter_max-jitter_mean)/jitter_STD
		
		#Frame Size: truncated Gaussian Distribution (in bytes)
		#parameters
		frame_size_Mean=[math.ceil(self.data_rate[0]*10**6/self.frame_rate[0]/8), math.ceil(self.data_rate[1]*10**6/self.frame_rate[1]/8), math.ceil(self.data_rate[2]*10**6/self.frame_rate[2]/8)]	#bytes
		frame_size_STD=np.multiply(frame_size_Mean,0.105)		#bytes		
		frame_size_min=np.multiply(frame_size_Mean,0.5)		#bytes
		frame_size_max=np.multiply(frame_size_Mean,1.5)		#bytes
		frame_size_a=(frame_size_min-frame_size_Mean)/frame_size_STD
		frame_size_b=(frame_size_max-frame_size_Mean)/frame_size_STD

		env.global_frame_size_min = min(frame_size_min)
		env.global_frame_size_max = max(frame_size_max)
		   
		
		interarrival_time = 1/self.frame_rate[1]*1000    #in ms - temporarily hardcoded with 60 fps
		interarrival_time = math.ceil((interarrival_time*10**-3)/env.scheduling_interval)   #in no.of scheduling time-slots
	
		num_total_arrivals=math.floor(duration/interarrival_time)
		
		self.stats_num_total_arrivals=num_total_arrivals
		traffic_index=self.UE_traffic_type

		for j in range(num_total_arrivals):

			#generating arrival slots
			self.arrival_slots_without_jitter[j]=(j+1)*interarrival_time
				
			#generate jitter for this frame
			jitter=math.ceil(scipy.stats.truncnorm.rvs(jitter_a,jitter_b,jitter_mean,jitter_STD))
			self.arrival_slots_with_jitter[j]=self.arrival_slots_without_jitter[j]+jitter + self.index   #self.index is the UE offset   
				
			#generate frame size
			self.frame_size[j]=math.ceil(scipy.stats.truncnorm.rvs(frame_size_a[traffic_index],frame_size_b[traffic_index],frame_size_Mean[traffic_index],frame_size_STD[traffic_index]))
			
			#populate deadline matrix to show frame deadlines
			self.frame_deadlines[j]=self.UE_traffic_PDB
			
		#Stats: store the initial packet sizes of the generated packets
		env.stats_initial_frame_sizes[self.index,:]=self.frame_size			

	def find_current_link_capacity(self,current_time, BS, env):   #find the current channel capacity between given UE-BS pair
		
		P_TX=24 #dBm, Transmit Power
		G_TX= 10 #dBi, Tx Antenna Gain
		G_RX= 10 #dBi, Rx Antenna Gain
		noise_spectral_density=-174		
		f=73        #GHz, frequency 
		alpha=0.004    #attenuation factor at 73 GHz
		# system_bandwidth=1000  #MHz
		# B=system_bandwidth/env.BSs		#MHz, Bandwidth of one BS 
		B = 400    #400 MHz bandwidth per BS
		noise_figure=8
		loss_factor=3
		rho_max=4.8		#bps/Hz, max spectral efficiency
		packet_size=env.packet_size	#in bytes
		P_N=noise_spectral_density+10*math.log10(B*10**6)+noise_figure
		

		if self.BS_Candidate_Set[BS.index] == 0: 			#this BS is not a feasible candidate for this UE
			self.current_link_capacity[BS.index]=-1
			return 0
		
		T=int(np.floor(current_time*env.scheduling_interval))
		BS_x=BS.GridLength
		BS_y=BS.GridLength
		
		#shifting coordinates so BS is at center and finding relative position of UE to apply the LOS/NLOS map
		x1 = env.UEs_posX_mobility[self.index,T]
		y1 = env.UEs_posY_mobility[self.index,T]
		x2 = env.BS_posX[BS.index]
		y2 = env.BS_posY[BS.index]

		dx=np.abs(x2-x1)
		dy=np.abs(y2-y1)
		if x2-x1 > 0: direction_x=1   # 1 means left 
		else: direction_x= -1		  # -1 means right
		if y2 - y1 > 0: direction_y = 1 	#1 means down
		else: direction_y = -1 				# -1 means up

		if dx > 0.5*env.MapLength:
			dx=1*env.MapLength - dx 
			direction_x=-direction_x

		if dy > 0.5*env.MapLength:
			dy= 1*env.MapLength- dy 
			direction_y=-direction_y

		modified_X_index=int(BS_x+direction_x*dx-1)
		modified_Y_index=int(BS_y+direction_y*dy-1)


		LOSprob=BS.LosMap[modified_X_index,modified_Y_index]

		if LOSprob == 1 : #LOS
			n=2    #path-loss exponent
			sigma= 4
		else:    #NLOS
			n=3.2 
			sigma=7

		distance_3D=np.sqrt(self.Distance_Matrix[BS.index]**2+(BS.BS_height-self.UE_height)**2)

		# PL=32.4 + 20*np.log10(f) + 10*n*np.log10(self.Distance_Matrix[BS.index]) + alpha*np.log10(self.Distance_Matrix[BS.index]) +np.random.normal(0,sigma)
		PL=32.4 + 20*np.log10(f) + 10*n*np.log10(distance_3D) + alpha*distance_3D +np.random.normal(0,sigma)
		
		P_RX=P_TX-PL+G_TX+G_RX
		SNR=P_RX-P_N	
		rho= np.log2(1+10**(0.1*(SNR-loss_factor)))
		rho=np.min([rho,rho_max])
				
		data_rate=rho*B*(10**6)/8  #in bytes per second	
		env.max_channel_capacity = rho_max*B*(10**6)/8   #in bytes per second
		env.max_channel_capacity = math.floor(env.max_channel_capacity/packet_size) #in packets per second
		env.max_channel_capacity = math.floor(env.max_channel_capacity*env.scheduling_interval) #in packets per scheduling slot

		packet_rate=math.floor(data_rate/packet_size)   #in packets per second
		packet_rate=math.floor(packet_rate*env.scheduling_interval) #in packets per scheduling slot

		if data_rate*8/10**6 > self.stats_max_link_capacity[BS.index]:
			self.stats_max_link_capacity[BS.index]=data_rate*8/10**6
		if data_rate*8/10**6 < self.stats_min_link_capacity[BS.index]:
			self.stats_min_link_capacity[BS.index]=data_rate*8/10**6

		#transition to the new link capacity
		self.current_link_capacity[BS.index]=packet_rate
		#self.current_link_capacity[BS.index] = 1000000
		self.stats_avg_link_capacity[BS.index]=(self.stats_avg_link_capacity[BS.index]+data_rate)/2      #in bytes per second
		self.stats_avg_link_capacity[BS.index]=self.stats_avg_link_capacity[BS.index]*8/10**6			#in Mbps

		return 0

	def dynamic_blockage_model(self,env):
		
		blocker_density= self.dynamic_blocker_density   #blockers per meter squared
		blocker_velocity=1		#meters per second
		blocker_height=1.8 		#meters
		expected_blockage_duration= int((500*(10**-3))/env.scheduling_interval) 	# 500 ms
		min_blockage_duration = int((1*(10**-3))/env.scheduling_interval) #1 ms
		

		for i in range(env.BSs):
			if self.Distance_Matrix[i] > env.connection_threshold:		#ignore BS if it is out of connection range
				continue

			BS = env.list_of_gNBs[i]
			#Generate Blocker Arrival Rate
			self.Theta[i]= (2/np.pi)*blocker_density*blocker_velocity*(blocker_height-self.UE_height)/(BS.BS_height-self.UE_height)
			self.alpha[i]=self.Theta[i]*self.Distance_Matrix[i]        #arrival rate in blockers/second
			self.alpha[i]=self.alpha[i]*env.scheduling_interval 			#arrival rate in blockers/time-slot	
			
			if self.wait_till_next_event[i]==0:
				#generate inter-arrival time till next blockage event
				#self.time_till_next_event[i]=my_generator.exponential(1/self.alpha[i])
				self.time_till_next_event[i]=np.random.exponential(1/self.alpha[i])
				
				#generate duration of next blockage event
				# self.blockage_duration[i]=my_generator.exponential(expected_blockage_duration)
				self.blockage_duration[i]=np.random.exponential(expected_blockage_duration)
				self.blockage_duration[i] = max(self.blockage_duration[i],min_blockage_duration)
				env.stats_system_total_num_blockages += 1
				env.stats_system_average_blockage_duration += self.blockage_duration[i]
				self.blockage_time[i]=self.time_of_last_event[i]+self.time_till_next_event[i]
				# print("UE ",self.index,"-BS ",i, ": Next blockage at ",self.time_till_next_event[i]+env.current_time," for ",self.blockage_duration[i], " ms")
				self.wait_till_next_event[i]=1
				if i == env.BSs-1:
					self.stats_blockage_arrivals.append(self.blockage_time[i])
					self.stats_blockage_durations.append(self.blockage_duration[i])

			if env.current_time ==  self.blockage_time[i]:   #blockage occurs
				self.blockage_timer[i]=int(np.maximum(self.blockage_timer[i],self.blockage_duration[i]))
				self.time_of_last_event[i]=self.blockage_time[i]

				# print("UE ",self.index," channel with BS ",i," blocked for ",self.blockage_timer[i]," ms")
				self.wait_till_next_event[i]=0


	def find_BS_Candidate_Set(self,current_time, env):		#calculate the Candidate Set C for this UE
		current_time=int(np.floor(current_time*env.scheduling_interval))
		UEs_curr_posX=env.UEs_posX_mobility[:,current_time]
		UEs_curr_posY=env.UEs_posY_mobility[:,current_time]
					
		#calculate the Distance Matrix 
		for j in range(env.BSs):
				self.Distance_Matrix[j]=env.Wraparound_Distance(UEs_curr_posX[self.index],UEs_curr_posY[self.index],env.BS_posX[j],env.BS_posY[j])
				#self.Distance_Matrix[j]=distance.euclidean([UEs_curr_posX[self.index],UEs_curr_posY[self.index]],[env.BS_posX[j],env.BS_posY[j]])
					
				#print("Distance: ",self.Distance_Matrix[j])
				if self.Distance_Matrix[j] < env.connection_threshold and self.blockage_timer[j] == 0:
					self.BS_Candidate_Set[j] = 1
				else:
					self.BS_Candidate_Set[j] = 0
		

		self.stats_avg_number_of_gNBs_in_Candidate_Set += np.sum(self.BS_Candidate_Set)

	def find_BS_Connectivity_Set(self,env):		#calculate the Connectivity Set K for this UE
		
		temp_channels=deepcopy(self.current_link_capacity)
		
		if env.current_time != 0:
			old_connectivity_set=deepcopy(self.BS_Connected_Set)

		self.connectivity_num_handovers_in_progress = 0
		
		#count number of handovers already in progress
		for i in range(env.degree_of_multiconnectivity):
			if self.connectivity_handovers_to[i] != -1:
				self.connectivity_num_handovers_in_progress += 1

		#Set channel states to negative (i.e BS unavailable) if link in blockage or already in process of handover
		for i in range(env.BSs):
			if self.blockage_timer[i] > 0:   #link in blockage
				temp_channels[i] = -1

			if self.connectivity_handover_timer_countdown[i] != -1:   #link already in process of handover
				temp_channels[i] = -1

		#K is the number of handovers we can still process if better links are found
		K = env.degree_of_multiconnectivity - int(self.connectivity_num_handovers_in_progress)
		bestCQI_indices = np.argpartition(temp_channels, -K)[-K:]
		bestCQI_indices = np.asarray(bestCQI_indices)
		
		#if all connected stations already in handover, no need to look for other better links
		if K == 0:
			bestCQI_indices = []


		num_blockage_handovers_in_progress = 0
		for i in range(env.degree_of_multiconnectivity):
			if self.connectivity_handovers_to[i] != -1 and self.connectivity_handovers_from[i] == -1:
				num_blockage_handovers_in_progress += 1

		prospective_Connected_Set = np.zeros(env.BSs, dtype = int)
		for i in range(env.BSs):
			if i in bestCQI_indices and temp_channels[i] >= 0 and (np.sum(self.BS_Connected_Set)+num_blockage_handovers_in_progress) < env.degree_of_multiconnectivity:
				prospective_Connected_Set[i] = 1
			else:
				prospective_Connected_Set[i] = 0

			#if link already in handover, preserve its previous state
			if self.connectivity_handover_timer_countdown[i] != -1 or self.replication_handover_timer_countdown[i] != -1: 
				prospective_Connected_Set[i] = self.BS_Connected_Set[i]

		if env.current_time == 0:
			self.BS_Connected_Set = prospective_Connected_Set
			return

		#if blockage occured while handover was in progress, cancel the handover
		for i in range(env.BSs):
			if i in self.connectivity_handovers_from:
				index = np.asarray(np.where(i == self.connectivity_handovers_from)) 
				
			elif i in self.connectivity_handovers_to:
				index = np.asarray(np.where(i == self.connectivity_handovers_to)) 
				
			else:
				continue

			if self.blockage_timer[i] > 0:	
				BS_handover_from = self.connectivity_handovers_from[int(index)]
				BS_handover_to = self.connectivity_handovers_to[int(index)]
				
				self.connectivity_handovers_to = np.delete(self.connectivity_handovers_to,index)
				self.connectivity_handovers_to = np.append(self.connectivity_handovers_to, -1)
				
				self.connectivity_handovers_from = np.delete(self.connectivity_handovers_from,index)
				self.connectivity_handovers_from = np.append(self.connectivity_handovers_from, -1)
				
				self.connectivity_num_handovers_in_progress -= 1
				self.connectivity_handover_timer_countdown[BS_handover_to] = -1
				self.connectivity_handover_timer_countdown[BS_handover_from] = -1
					
		
		#if blockage occured this slot, just execute a handover_to
		while 1:
			for i in range(env.BSs):
				if i in self.connectivity_handovers_from or i in self.connectivity_handovers_to:
					continue 

				if self.connectivity_blockage_started_this_slot == 0 or max(self.blockage_timer) == 0:
					break

				if old_connectivity_set[i] == 0 and self.blockage_timer[i] > 0:
					self.connectivity_blockage_started_this_slot = 0

				if i in self.connectivity_handovers_from or i in self.connectivity_handovers_to:
					continue 

				if old_connectivity_set[i] == 0 and prospective_Connected_Set[i] == 1 and self.connectivity_blockage_started_this_slot == 1:
					self.connectivity_handover_timer_countdown[i] = env.connectivity_handover_delay
					self.connectivity_handovers_to[int(self.connectivity_num_handovers_in_progress)] = i
					self.connectivity_num_handovers_in_progress += 1
					self.connectivity_blockage_started_this_slot = 0
			
			if self.connectivity_blockage_started_this_slot == 0 or max(self.blockage_timer) == 0:
				break


		#if no blockage occured, but there is still a handover it must involve two BSs: handover_to and handover_from
		for i in range(env.BSs):
			if i in self.connectivity_handovers_from or i in self.connectivity_handovers_to:
				continue 

			if old_connectivity_set[i] != prospective_Connected_Set[i] and self.connectivity_num_handovers_in_progress <= env.degree_of_multiconnectivity:
				if old_connectivity_set[i] == 0 and prospective_Connected_Set[i] == 1:
					self.connectivity_handover_timer_countdown[i] = env.connectivity_handover_delay
					self.connectivity_handovers_to[int(self.connectivity_num_handovers_in_progress)] = i
					self.connectivity_num_handovers_in_progress += 0.5

					#find the complimentary handover_from
					for j in range(env.BSs):
						if j in self.connectivity_handovers_from or j in self.connectivity_handovers_to:
							continue 

						if old_connectivity_set[j] == 1 and prospective_Connected_Set[j] == 0:
							self.connectivity_handover_timer_countdown[j] = env.connectivity_handover_delay
							
							self.connectivity_handovers_from[int(self.connectivity_num_handovers_in_progress)] = j
							self.connectivity_num_handovers_in_progress += 0.5
							break


		#decrement handovers
		for i in range(env.BSs):
			if i in self.connectivity_handovers_from or i in self.connectivity_handovers_to:
				self.connectivity_handover_timer_countdown[i] -= 1


		#if handover timer reaches 0, execute the handover
		for i in range(env.BSs):
			if self.connectivity_handover_timer_countdown[i] == 0 and i in self.connectivity_handovers_to:
				index = np.asarray(np.where(i == self.connectivity_handovers_to))

				self.BS_Connected_Set[i] = 1	
				self.connectivity_handover_timer_countdown[i] = -1
				self.connectivity_handovers_to[index] = -1
				
				if self.connectivity_handovers_from[index] != -1:
					BS_handing_off = self.connectivity_handovers_from[index]
					self.BS_Connected_Set[BS_handing_off] = 0
					self.BS_Replication_Set[BS_handing_off] = 0
					self.connectivity_handover_timer_countdown[BS_handing_off] = -1
					self.connectivity_handovers_from[index] = -1

				if self.connectivity_handovers_from[0] == -1 and self.connectivity_handovers_to[0] == -1:
					self.connectivity_handovers_to = np.roll(self.connectivity_handovers_to, -1)
					self.connectivity_handovers_from = np.roll(self.connectivity_handovers_from, -1)
		

	def find_BS_Replication_Set(self,env,predictor):		#calculate the Replication Set L for this UE
		
		temp_channels=deepcopy(self.current_link_capacity)
		
		if env.current_time != 0:
			old_replication_set=deepcopy(self.BS_Replication_Set)

		self.replication_num_handovers_in_progress = 0
		
		#count number of handovers already in progress
		for i in range(env.degree_of_replication):
			if self.replication_handovers_to[i] != -1:
				self.replication_num_handovers_in_progress += 1

		#Set channel states to negative (i.e BS unavailable) if link in blockage or already in process of handover or not in connectivity set
		for i in range(env.BSs):
			if self.blockage_timer[i] > 0:   #link in blockage
				temp_channels[i] = -1

			if self.replication_handover_timer_countdown[i] != -1:   #link already in process of handover
				temp_channels[i] = -1

			if self.BS_Connected_Set[i] == 0:    #link not in connectivity set yet
				temp_channels[i] = -1

		#L is the number of handovers we can still process if better links are found
		L = env.degree_of_replication - int(self.replication_num_handovers_in_progress)
		bestCQI_indices = np.argpartition(temp_channels, -L)[-L:]
		bestCQI_indices = np.asarray(bestCQI_indices)
		
		#if all connected stations already in handover, no need to look for other better links
		if L == 0:
			bestCQI_indices = []
		
		num_blockage_handovers_in_progress = 0
		for i in range(env.degree_of_replication):
			if self.replication_handovers_to[i] != -1 and self.replication_handovers_from[i] == -1:
				num_blockage_handovers_in_progress += 1

		prospective_replication_Set = np.zeros(env.BSs, dtype = int)
		for i in range(env.BSs):
			if i in bestCQI_indices and temp_channels[i] >= 0 and (np.sum(self.BS_Replication_Set)+num_blockage_handovers_in_progress) < env.degree_of_replication:
				prospective_replication_Set[i] = 1
			else:
				prospective_replication_Set[i] = 0

			#if link already in handover, preserve its previous state
			if self.replication_handover_timer_countdown[i] != -1: #or self.connectivity_handover_timer_countdown[i] != -1: 
				prospective_replication_Set[i] = self.BS_Replication_Set[i]

		if env.current_time == 0:
			self.BS_Replication_Set = prospective_replication_Set
			return

		#if blockage occured while handover was in progress, cancel the handover
		for i in range(env.BSs):
			if i in self.replication_handovers_from:
				index = np.asarray(np.where(i == self.replication_handovers_from)) 
				
			elif i in self.replication_handovers_to:
				index = np.asarray(np.where(i == self.replication_handovers_to)) 
				
			else:
				continue

			if self.blockage_timer[i] > 0:	
				BS_handover_from = self.replication_handovers_from[int(index)]
				BS_handover_to = self.replication_handovers_to[int(index)]
				
				self.replication_handovers_to = np.delete(self.replication_handovers_to,index)
				self.replication_handovers_to = np.append(self.replication_handovers_to, -1)
				
				self.replication_handovers_from = np.delete(self.replication_handovers_from,index)
				self.replication_handovers_from = np.append(self.replication_handovers_from, -1)
				
				self.replication_num_handovers_in_progress -= 1
				self.replication_handover_timer_countdown[BS_handover_to] = -1
				self.replication_handover_timer_countdown[BS_handover_from] = -1
					

		#if blockage occured this slot, just execute a handover_to
		while 1:
			for i in range(env.BSs):
				if i in self.replication_handovers_from or i in self.replication_handovers_to:
					continue 

				if self.replication_blockage_started_this_slot == 0 or max(self.blockage_timer) == 0:
					break

				if old_replication_set[i] == 0 and self.blockage_timer[i] > 0:
					self.replication_blockage_started_this_slot = 0

				if i in self.replication_handovers_from or i in self.replication_handovers_to:
					continue 

				if old_replication_set[i] == 0 and prospective_replication_Set[i] == 1 and self.replication_blockage_started_this_slot == 1:
					self.replication_handover_timer_countdown[i] = env.replication_handover_delay
					self.replication_handovers_to[int(self.replication_num_handovers_in_progress)] = i
					self.replication_num_handovers_in_progress += 1
					self.replication_blockage_started_this_slot = 0
			
			if self.replication_blockage_started_this_slot == 0 or max(self.blockage_timer) == 0:
				break

		#if no blockage occured, but there is still a handover it must involve two BSs: handover_to and handover_from
		for i in range(env.BSs):
			if i in self.replication_handovers_from or i in self.replication_handovers_to:
				continue 

			if old_replication_set[i] != prospective_replication_Set[i] and self.replication_num_handovers_in_progress <= env.degree_of_replication:
				if old_replication_set[i] == 0 and prospective_replication_Set[i] == 1:
					self.replication_handover_timer_countdown[i] = env.replication_handover_delay
					self.replication_handovers_to[int(self.replication_num_handovers_in_progress)] = i
					self.replication_num_handovers_in_progress += 0.5

					#find the complimentary handover_from
					for j in range(env.BSs):
						if j in self.replication_handovers_from or j in self.replication_handovers_to:
							continue 

						if old_replication_set[j] == 1 and prospective_replication_Set[j] == 0:
							self.replication_handover_timer_countdown[j] = env.replication_handover_delay
							
							self.replication_handovers_from[int(self.replication_num_handovers_in_progress)] = j
							self.replication_num_handovers_in_progress += 0.5
							break


		#decrement handovers
		for i in range(env.BSs):
			if i in self.replication_handovers_from or i in self.replication_handovers_to:
				self.replication_handover_timer_countdown[i] -= 1


		#if handover timer reaches 0, execute the handover
		for i in range(env.BSs):
			if self.replication_handover_timer_countdown[i] == 0 and i in self.replication_handovers_to:
				index = np.asarray(np.where(i == self.replication_handovers_to))

				self.BS_Replication_Set[i] = 1
				self.replication_handover_timer_countdown[i] = -1
				self.replication_handovers_to[index] = -1

				
				if self.replication_handovers_from[index] != -1:
					BS_handing_off = self.replication_handovers_from[index]
					self.BS_Replication_Set[BS_handing_off] = 0
					self.replication_handover_timer_countdown[BS_handing_off] = -1
					self.replication_handovers_from[index] = -1

				if self.replication_handovers_from[0] == -1 and self.replication_handovers_to[0] == -1:
					self.replication_handovers_to = np.roll(self.replication_handovers_to, -1)
					self.replication_handovers_from = np.roll(self.replication_handovers_from, -1)

		for i in range(env.BSs):
			current_BS = env.list_of_gNBs[i]
			for j in range(env.UEs):
				current_UE = env.list_of_UEs[j]
				if current_UE.BS_Replication_Set[i] == 1:
					current_BS.UE_Replication_Set[j] = 1

	def algo_random_replication_set(self,env):
		print("Using Random Predictor")
		#select L BSs from Connectivity Set randomly
		#print("Random Predictor Chosen")
		if env.degree_of_multiconnectivity != env.degree_of_replication:
			possible_BSs = []
			for i in range(env.BSs):
				if self.BS_Connected_Set[i] == 1:
					possible_BSs.append(i)

			#print("Replication Indices: ",Replication_BSs_indices)
			#selected_BS_indices = np.random.randint(env.degree_of_multiconnectivity, size = env.degree_of_replication)
			li=range(0,env.degree_of_multiconnectivity-1)
			selected_BS_indices=random.sample(li,env.degree_of_replication)
			selected_BSs = [ possible_BSs[i] for i in selected_BS_indices]
			
			for i in range(env.BSs):
				if i in selected_BSs:
					self.BS_Replication_Set[i] = 1
				
		elif env.degree_of_multiconnectivity < env.degree_of_replication:
			print("ERROR: Degree of Multiconnectivity (K) can't be less than Degree of Replication (L)" )
			exit()
		else:
			self.BS_Replication_Set = self.BS_Connected_Set
	
	def algo_nearest_neighbors_replication_set(self,env):
		

	def algo_bestCQI_replication_set(self,env):
		print("Using BestCQI Predictor")
		#select L BSs from Connectivity Set according to best CQIs
		temp_channels=deepcopy(self.current_link_capacity)
		old_replication_set = self.BS_Replication_Set

		L = env.degree_of_replication
		for i in range(env.BSs):
			if self.BS_replication_handover_progress[i] != -1 or self.BS_Connected_Set[i] == 0:
				temp_channels[i] = -1

		print("Num Handovers in Progress: ",self.num_BS_replication_handovers_in_progress)
		L -= int(self.num_BS_replication_handovers_in_progress)
		print("Handover Progress: ",self.BS_replication_handover_progress)
		print("L: ",L)
		if env.degree_of_multiconnectivity != env.degree_of_replication:
			bestCQI_indices = np.argpartition(self.current_link_capacity, -L)[-L:]
			bestCQI_indices = np.asarray(bestCQI_indices)
			print("Best CQI: ",bestCQI_indices)
			for i in range(env.BSs):
				if self.BS_replication_handover_progress[i] != -1:
					self.BS_Replication_Set[i] = 2   #handover in progress
					continue

				if i in bestCQI_indices:
					#if self.BS_replication_handover_progress[i] != -1 and old_replication_set[i] == 0 and self.blockage_timer[i] == 0:
					self.BS_Replication_Set[i] = 1
				else:
					self.BS_Replication_Set[i] = 0
			print("Proposed Replication Set: ",self.BS_Replication_Set)

		elif env.degree_of_multiconnectivity < env.degree_of_replication:
			print("ERROR: Degree of Multiconnectivity (K) can't be less than Degree of Replication (L)" )
			exit()
		else:
			self.BS_Replication_Set = self.BS_Connected_Set
			print("K = L")


	def algo_predictor_NN(self,env,current_predictor_Q_values):
		M = env.BSs 
		N = env.UEs 
		L = env.degree_of_replication
		
		current_UE_Q_values = current_predictor_Q_values[M*self.index:M*(self.index+1)]
		for i in range(M):
			if i not in self.BS_Connected_Set:
				current_UE_Q_values[i] = -1000
		#find the L top Q-value gNBs
		max_ind = np.argpartition(current_UE_Q_values, -L)[-L:]

		self.BS_Replication_Set = np.asarray(max_ind).flatten()



class BS(Environment):
	def __init__(self, env, index):
		self.index= index
		self.UE_Connected_Set=np.zeros(env.UEs,dtype=int) 		#the set of UEs connected to this BS
		self.UE_Replication_Set=np.zeros(env.UEs,dtype=int)  		#the set of UEs whose data is buffered in this BS
		self.GridLength=env.connection_threshold   #max one-sided coverage distance of a single BS
		self.CorrelationDistance=10  #in meters, for UMi scenario
		self.Granularity=1   #granularity of grid in meters
		self.BS_height=5 	#height of BS in meters
		N=int(2*self.GridLength/self.Granularity +1)
		self.LosMap=-1*np.ones([N-1,N-1])

	def Generate_Correlated_LOS_Map(self):

		gridLength=self.GridLength
		d_co=self.CorrelationDistance

		#generate grid
		d_px=self.Granularity 		#granularity of grid in meters
		N=int(2*gridLength/d_px +1)
		delta= d_co/d_px

		x,y=np.meshgrid(range(-gridLength,gridLength,d_px),range(-gridLength,gridLength,d_px))
		Pr_LOS=np.zeros([N-1,N-1])

		d1=22
		d2=100
		d_2d=np.sqrt(np.square(x)+np.square(y))+np.finfo(float).eps

		#Probability of LOS condition - UMi Scenario
		for i in range(1,N):
			for j in range(1,N):
				d=d_2d[i-1,j-1]
				Pr_LOS[i-1,j-1]=np.minimum(d1/d,1)*(1-np.exp(-d/d2))+np.square(np.exp(-d/d2))

		#Filter co-efficient is considered as 0 when distance is beyond 4*d_co
		M=int(8*delta + 1)
		h=np.zeros([M-1,M-1])
		InitMap= np.random.randn(N+M-1,N+M-1)

		#Generate the filter
		for i in range(1,M):
			for j in range(1,M):
				h[i-1,j-1]=np.exp(-np.sqrt(((M+1)/2-i)**2+((M+1)/2-j)**2)/d_co)
		
		CorrMapPad=signal.convolve2d(InitMap,h,'same')
		CorrQ = CorrMapPad[int((M+1)/2):int((M+1)/2+N-1),int((M+1)/2):int((M+1)/2+N-1)];
		CorrK = 1/2*(1+scipy.special.erf(CorrQ/np.sqrt(2)));

		LosMap=-1*np.ones([N-1,N-1])
		for i in range(N-1):
			for j in range(N-1):
				LosMap[i,j]=1 if CorrK[i,j] < Pr_LOS[i,j] else 0

		self.LosMap=LosMap







	
