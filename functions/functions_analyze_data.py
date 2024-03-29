import numpy as np
import scipy.sparse 
from scipy.interpolate import interp1d
import sys 
import os 

# returns a list of unique backup directory file names  
def loadListOfBackupTimesteps( Directory ):
	
	lines=[]
	backupSWithDuplicates=[]

	# load lines from list ob backup times
	with open( Directory+'/listOfBackupTimeSteps.txt','r') as f:
		lines.append( f.read() )
		lines=(lines[0].split('\n'))
		for line in lines:
			backupSWithDuplicates.append(line.split(' '))

		backupS = []
		for i in backupSWithDuplicates:
			if i not in backupS:
				backupS.append(i)
	return backupS
	
# loads complete spike train and weight sequence for order of listed directories
def load_Complete_SpikeTrain_And_Weigth_Trajectories( sortedListOfDirectories ):
	
	import os

	# initialize spikeTrain and weightData
	spikeTrain=[]
	weightData=[]
				
	for Directory in sortedListOfDirectories:

		# load first Trajectory   
		if os.path.isfile( Directory+'/listOfBackupTimeSteps.txt' ): 
			# load short data
			# load long data
			backupS =loadListOfBackupTimesteps( Directory )
	
			# load data
			for fileEndings in range(len(backupS)-1):
				
				# get actual file ending
				fileEndingString=backupS[ fileEndings ][1]
				if fileEndingString!='0_sec':
					
					#print fileEndingString
					spikeTimesFile=Directory+'/spikeTimes_'+str(fileEndingString)+'.npy'
						
					if os.path.isfile( spikeTimesFile ):
						if len(spikeTrain)==0:
							spikeTrain=np.load( spikeTimesFile )
						else:

							# ensure that spike trains dont overlap
							newSpikeTrain = np.load( spikeTimesFile )
							if len(newSpikeTrain) != 0:
								#print len(newSpikeTrain), len(spikeTrain), spikeTrain[-1,1], newSpikeTrain[0,1]
								oldSpikeTrain = spikeTrain[ spikeTrain[:,1]<newSpikeTrain[0,1] ]

								spikeTrain=np.concatenate( ( oldSpikeTrain, newSpikeTrain ), axis=0 )
					
					weightFile=Directory+'/meanWeightTimeSeries_'+str(fileEndingString)+'.npy'
					if os.path.isfile( weightFile ):
						if len(weightData)==0:
							weightData=np.load( weightFile )
						else:
							newWeightData = np.load(weightFile )
							# remove zeros
							newWeightData = newWeightData[ newWeightData[:,0]!=0 ]
							oldWeightData = weightData[ weightData[:,0]<newWeightData[0,0] ]

							weightData=np.concatenate( ( oldWeightData, newWeightData  ), axis=0 )
	
		else:
			print( 'Error: no data found in', Directory )
		

	# returns spike train and weight trajectory. 
	# times are in simulation time steps
	return spikeTrain, weightData        
		
# loads complete spike train and weight sequence for order of listed directories 
# only files with data between tmin and tmax are loaded
def load_Complete_SpikeTrain_And_Weigth_Trajectories_Tmin_Tmax( sortedListOfDirectories, tmin, tmax ):
	
	import os

	# initialize spikeTrain and weightData
	spikeTrain=[]
	weightData=[]
				
	for Directory in sortedListOfDirectories:

		# load first Trajectory       
		if os.path.isfile( Directory+'/listOfBackupTimeSteps.txt' ): 
			# load short data
			# load long data
			backupS =loadListOfBackupTimesteps( Directory )
	
			# load data
			for fileEndings in range(len(backupS)-1):
				
				# get actual file ending
				fileEndingString=backupS[ fileEndings ][1]
				if fileEndingString!='0_sec':

					# check whether times between tmin and tmax
					fileEndingString.split("_")
					try:
						tFile = fileEndingString.split("_")[0]
						
						# check whether file data are in interval of interest
						if ( int( tFile ) >= tmin - 10 ) and ( int( tFile ) <= tmax + 30 ):
							# print(tFile)
							#print fileEndingString
							spikeTimesFile=Directory+'/spikeTimes_'+str(fileEndingString)+'.npy'
								
							if os.path.isfile( spikeTimesFile ):
								if len(spikeTrain)==0:
									spikeTrain=np.load( spikeTimesFile )
								else:

									# ensure that spike trains dont overlap
									newSpikeTrain = np.load( spikeTimesFile )
									if len(newSpikeTrain) != 0:
										#print len(newSpikeTrain), len(spikeTrain), spikeTrain[-1,1], newSpikeTrain[0,1]
										oldSpikeTrain = spikeTrain[ spikeTrain[:,1]<newSpikeTrain[0,1] ]

										spikeTrain=np.concatenate( ( oldSpikeTrain, newSpikeTrain ), axis=0 )
							
							weightFile=Directory+'/meanWeightTimeSeries_'+str(fileEndingString)+'.npy'
							if os.path.isfile( weightFile ):
								if len(weightData)==0:
									weightData=np.load( weightFile )
								else:
									newWeightData = np.load(weightFile )
									# remove zeros
									newWeightData = newWeightData[ newWeightData[:,0]!=0 ]
									oldWeightData = weightData[ weightData[:,0]<newWeightData[0,0] ]

									weightData=np.concatenate( ( oldWeightData, newWeightData  ), axis=0 )

					except:
						print("Warning: couldn't read time points from ", fileEndingString )

					
	
		else:
			print( 'ERROR: no data found in', Directory )
		

	# returns spike train and weight trajectory. 
	# times are in simulation time steps
	return spikeTrain, weightData        
		
# cacluates the Kuramoto order parameter for arrayOfNeuronIndixes from spikeTimes
def piece_wise_calcKuramotoOrderParameter( spikeTimes, tmin, tmax, resolution, arrayOfNeuronIndixes, outputFilename ):

	# delete empty entries
	spikeTimes=spikeTimes[ spikeTimes[:,1]!= 0 ]

	################################################################################
	######## the following two lines were added later ##############################
	populationSize=len(arrayOfNeuronIndixes)
	
	# number of grid points for which Kuramoto order parameter is evaluated at once
	NinterPolSteps = int( 1000.0*( tmax-tmin )/float(resolution) )
	processAtOnce_NinterPolSteps = 100000

	# if too many gridpoints are given a piece-wise calculation is performed
	if NinterPolSteps > processAtOnce_NinterPolSteps:
		
		KuramotoOutArray = []
		
		currentInterPolSteps = 0
		lengthOfTimeIntervals = resolution * processAtOnce_NinterPolSteps # ms
		# in order to exclude boundary effects when combining arrays, we consider an overlap of 2000 ms
		TimeStepsOfOverlap = int( 2000.0/resolution ) # time steps

		if 2*TimeStepsOfOverlap > processAtOnce_NinterPolSteps:
			print( 'ERROR: overlap for piece-wise Kuramoto order parameter calculation too long compared to processAtOnce_NinterPolSteps!' )
			return 0
		
		current_Tmin = 1000.0*tmin # ms
		current_Tmax = current_Tmin + lengthOfTimeIntervals # ms

		while current_Tmax < 1000*tmax:

			# initialize phases at fixed points
			phases=np.zeros( (populationSize, processAtOnce_NinterPolSteps) )
			arrayOfGridPoints=np.arange( current_Tmin, current_Tmax , resolution)

			# consider only spikes that are between tmin and tmax
			processedSpikeTimes=spikeTimes[ np.logical_and( spikeTimes[:,1]>=current_Tmin , spikeTimes[:,1]<= current_Tmax )  ]        

			# calculate phases in current interval
			krecPhases=0

			for kNeuron in arrayOfNeuronIndixes:

				# get spike train of corresponding neuron
				spikeTrainTemp = processedSpikeTimes[processedSpikeTimes[:,0].astype(int)== kNeuron ][:,1]

				# calc phase function
				if len(spikeTrainTemp) != 0:
					phaseNPiCrossings=np.concatenate( ( np.full( 1, 1000.0*(2*tmin-tmax) ) , spikeTrainTemp, np.full( 1, 1000.0*(2*tmax-tmin) ) ), axis=0 )
					PhaseValues=np.linspace(0,len(phaseNPiCrossings)-1, len(phaseNPiCrossings))
				else:
					phaseNPiCrossings=np.array([-np.inf, np.inf])
					PhaseValues=np.array([0, 1])

				# linear interpolate phaseNPiCrossings
				phaseFunctionKNeuron=interp1d(phaseNPiCrossings,2*np.pi*PhaseValues)

				phases[krecPhases,:]=phaseFunctionKNeuron(arrayOfGridPoints) 
				krecPhases+=1

			# calc Kuramoto order parameter
			TotalArrayOfKuramotoOrderParameterAtGridPoints=1/float(populationSize)*np.absolute(np.sum( np.exp( 1j*phases ), axis=0 ))

			current_KuramotoOutArray=np.array( [arrayOfGridPoints, TotalArrayOfKuramotoOrderParameterAtGridPoints] )

			current_KuramotoOutArray=np.transpose( current_KuramotoOutArray )

			if len(KuramotoOutArray) == 0:
				KuramotoOutArray = current_KuramotoOutArray[:-TimeStepsOfOverlap]
			else:
				# add to previous Kuramoto array
				KuramotoOutArray = np.concatenate( ( KuramotoOutArray, current_KuramotoOutArray[ TimeStepsOfOverlap:-TimeStepsOfOverlap ] ), axis = 0 )

			# prepare boundaries of next interval 
			current_Tmin = current_Tmax - resolution * 2*TimeStepsOfOverlap # ms
			current_Tmax = current_Tmin + lengthOfTimeIntervals # ms
	 
			print( 'current interval (left boundary)' , current_Tmin * 0.001 )
				
				
	# calculate in one single run    
	else:
		phases=np.zeros( (populationSize, NinterPolSteps) )
		arrayOfGridPoints=np.linspace(1000.0*tmin,1000.0*tmax,NinterPolSteps)

		processedSpikeTimes=spikeTimes[ np.logical_and( spikeTimes[:,1]>=1000.0*tmin , spikeTimes[:,1]<= 1000.0*tmax )  ]

			
		krecPhases=0
		for kNeuron in arrayOfNeuronIndixes:

			# get spike train of corresponding neuron
			spikeTrainTemp=processedSpikeTimes[processedSpikeTimes[:,0].astype(int)== kNeuron ][:,1]
			# calc phase function
			if len(spikeTrainTemp) != 0:
				phaseNPiCrossings=np.concatenate( ( np.full( 1, 1000.0*(2*tmin-tmax) ) , spikeTrainTemp, np.full( 1, 1000.0*(2*tmax-tmin) ) ), axis=0 )
				PhaseValues=np.linspace(0,len(phaseNPiCrossings)-1, len(phaseNPiCrossings))
			else:
				phaseNPiCrossings=np.array([-np.inf, np.inf])
				PhaseValues=np.array([0, 1])

			# linear interpolate phaseNPiCrossings
			phaseFunctionKNeuron=interp1d(phaseNPiCrossings,2*np.pi*PhaseValues)

			phases[krecPhases,:]=phaseFunctionKNeuron(arrayOfGridPoints) 
			krecPhases+=1

		# calc Kuramoto order parameter
		TotalArrayOfKuramotoOrderParameterAtGridPoints=1/float(populationSize)*np.absolute(np.sum( np.exp( 1j*phases ), axis=0 ))
		KuramotoOutArray=np.array( [arrayOfGridPoints, TotalArrayOfKuramotoOrderParameterAtGridPoints] )
		
		KuramotoOutArray=np.transpose( KuramotoOutArray )

		
	if outputFilename!='':
	   np.save( outputFilename+'.npy' , KuramotoOutArray )
	
	return KuramotoOutArray

# The following functions calculates the trajectory of the Kuramoto order parameter and the mean
# synaptic weight for simulation data stored in sortedListOfDirectories.
def get_Kuramoto_trajectory_and_mean_weight( sortedListOfDirectories, tmin, tmax ):

	# load spike train and weight data
	spikeTrain, weightData = load_Complete_SpikeTrain_And_Weigth_Trajectories_Tmin_Tmax( sortedListOfDirectories, tmin, tmax )

	# print( weightData )
	# load adjacency matrix 
	path_to_first_sim_result = sortedListOfDirectories[0] + "/0_sec/synConnections.npz"
	adj = scipy.sparse.load_npz( path_to_first_sim_result )

	resolution = 246.0 # ms (one data point every "resolution" ms)
	arrayOfNeuronIndixes = np.arange( 1000 ).astype( int )
	print("calculating Kuramoto order parameter")
	KuramotoTrajectory = piece_wise_calcKuramotoOrderParameter( spikeTrain*[1,0.1], tmin, tmax, resolution, arrayOfNeuronIndixes, "" )

	weightData = weightData[ weightData[:,0] != 0 ] 

	weightData[:,0] = 0.0001*weightData[:,0] # sec
	weightData[:,1] = weightData[:,1]/np.mean( adj[:1000,:1000] ) # sec

	return weightData , KuramotoTrajectory

def load_weight_data( directory, seed , d, outputDirectory ):

	tmin = 0    # sec
	tmax = 10020 # sec

	# mean weight 0.0
	mw_init = 0.0
	sortedListOfDirectories = [ directory+"/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_"+str(mw_init) ]

	w1, rho1 = get_Kuramoto_trajectory_and_mean_weight( sortedListOfDirectories, tmin, tmax )
	rho1[:,0] = 0.001*rho1[:,0] # transform to seconds

	print(mw_init, "done")

	# mean weight 0.15
	mw_init = 0.15
	sortedListOfDirectories = [ directory+"/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_"+str(mw_init) ]

	w2, rho2 = get_Kuramoto_trajectory_and_mean_weight( sortedListOfDirectories, tmin, tmax )
	rho2[:,0] = 0.001*rho2[:,0] # transform to seconds

	print(mw_init, "done")

	# mean weight 0.3
	mw_init = 0.3
	sortedListOfDirectories = [ directory+"/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_"+str(mw_init) ]

	w3, rho3 = get_Kuramoto_trajectory_and_mean_weight( sortedListOfDirectories, tmin, tmax )
	rho3[:,0] = 0.001*rho3[:,0] # transform to seconds

	print(mw_init, "done")

	# mean weight 0.45
	mw_init = 0.45
	sortedListOfDirectories = [ directory+"/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_"+str(mw_init) ]

	w4, rho4 = get_Kuramoto_trajectory_and_mean_weight( sortedListOfDirectories, tmin, tmax )
	rho4[:,0] = 0.001*rho4[:,0] # transform to seconds

	print(mw_init, "done")

	# # mean weight 0.6
	# mw_init = 0.6
	# sortedListOfDirectories = [ directory+"/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_"+str(mw_init) ]

	# w5, rho5 = get_Kuramoto_trajectory_and_mean_weight( sortedListOfDirectories, tmin, tmax )
	# rho5[:,0] = 0.001*rho5[:,0] # transform to seconds

	# print(mw_init, "done")

	# # mean weight 0.75
	# mw_init = 0.75
	# sortedListOfDirectories = [ directory+"/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_"+str(mw_init) ]

	# w6, rho6 = get_Kuramoto_trajectory_and_mean_weight( sortedListOfDirectories, tmin, tmax )
	# rho6[:,0] = 0.001*rho6[:,0] # transform to seconds

	# print(mw_init, "done")

	# # mean weight 0.9
	# mw_init = 0.9
	# sortedListOfDirectories = [ directory+"/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_"+str(mw_init) ]

	# w7, rho7 = get_Kuramoto_trajectory_and_mean_weight( sortedListOfDirectories, tmin, tmax )
	# rho7[:,0] = 0.001*rho7[:,0] # transform to seconds

	# print(mw_init, "done")

	# save data
	np.savez( outputDirectory+"/data_initial_network_seed_"+str(seed)+"_d_"+str(d)+".npz", w1=w1, rho1=rho1, w2=w2, rho2=rho2, w3=w3, rho3=rho3, w4=w4, rho4=rho4 )

	# np.savez( outputDirectory+"/data_initial_network_seed_"+str(seed)+"_d_"+str(d)+".npz", w1=w1, rho1=rho1, w2=w2, rho2=rho2, w3=w3, rho3=rho3, w4=w4, rho4=rho4, w5=w5, rho5=rho5, w6=w6, rho6=rho6, w7=w7, rho7=rho7 )

def loadData( seed, d, directory ):
	filename = directory + "/data_initial_network_seed_"+str(seed)+"_d_"+str(d)+".npz"
	print(filename)
	data = np.load( filename )

	w1 = data['w1']
	w2 = data['w2']
	w3 = data['w3']
	w4 = data['w4']
	# w5 = data['w5']
	# w6 = data['w6']
	# w7 = data['w7']

	w = [w1, w2, w3, w4] #, w5, w6, w7]
	
	rho1 = data['rho1']
	rho2 = data['rho2']
	rho3 = data['rho3']
	rho4 = data['rho4']
	# rho5 = data['rho5']
	# rho6 = data['rho6']
	# rho7 = data['rho7']
	
	rho = [rho1, rho2, rho3, rho4] #, rho5, rho6, rho7]
	
	return w, rho

########## the following function generates Figure 2.
def genFigureMultistabiltiy( directory, seed ):
	labelFontsize = 15
	tickFontsize = 12
	
	import matplotlib.pyplot as plt 

	fig = plt.figure()

	axRhoD1 = fig.add_subplot(2,3,1)
	axRhoD2 = fig.add_subplot(2,3,2)
	axRhoD3 = fig.add_subplot(2,3,3)

	axesRho = [axRhoD1,axRhoD2,axRhoD3]

	axwD1 = fig.add_subplot(2,3,4)
	axwD2 = fig.add_subplot(2,3,5)
	axwD3 = fig.add_subplot(2,3,6)

	axesW = [axwD1,axwD2,axwD3]
	colors = ["0.8","0.6","0.4","0.2","0.0"]

	d_values = [0.4, 2.0, 10.0] # values of synaptic length scale

	xticks = [0, 1000, 2000, 3000, 4000, 5000]
	xticklabels = ["$0$", "", "$2000$", "", "$4000$", ""]

	xticks      = [0,     1000, 2000,   3000, 4000,   5000, 6000,   7000, 8000, 9000, 10000]
	xticklabels = ["$0$", "", "", "", "$4000$", "", "", "", "$8000$", "", ""]

	yticksRho = [0, 0.25, 0.5, 0.75, 1]
	ytickRholabels = ["$0$", "", "", "", "$1$"]

	yticksW = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
	ytickWlabels = ["$0$", "", "$0.2$", "", "$0.4$",""]


	for kd in range( len(d_values) ):
		d = d_values[kd]
		w, rho = loadData( seed, d, directory )

		for kwinit in range(4):

			rhoSingle = rho[kwinit]
			axesRho[kd].plot( rhoSingle[:,0] , rhoSingle[:,1], color = colors[kwinit] )

			wSingle = w[kwinit]
			axesW[kd].plot( wSingle[:,0] , wSingle[:,1], color = colors[kwinit] )

	for ax in np.concatenate( ( axesRho, axesW ), axis = 0 ):
		ax.set_xticks( xticks )
		ax.set_xticklabels( ["" for x in xticks], fontsize = tickFontsize )

		ax.set_xlim(0,10000)

	for ax in axesRho:

		ax.set_yticks( yticksRho )
		ax.set_yticklabels( ["" for x in yticksRho], fontsize = tickFontsize )

		ax.set_ylim(0, 1.05)

	axesRho[0].set_yticklabels( ytickRholabels, fontsize = tickFontsize )
	axesRho[0].set_ylabel(r"$\rho$", fontsize = labelFontsize )

	for ax in axesW:
		ax.set_xticklabels( xticklabels, fontsize = tickFontsize )

		ax.set_yticks( yticksW )
		ax.set_yticklabels( ["" for x in yticksW], fontsize = tickFontsize )

		ax.set_ylim(0,0.5)

		ax.set_xlabel(r"$t$ (sec)", fontsize = labelFontsize )

	axesW[0].set_yticklabels( ytickWlabels, fontsize = tickFontsize )
	axesW[0].set_ylabel( r"$\langle w \rangle$", fontsize = labelFontsize )

	# axesRho[0].text(-1500, 1.05, "A", fontsize = 1.2*labelFontsize )
	# axesRho[1].text(-800, 1.05, "B", fontsize = 1.2*labelFontsize )
	# axesRho[2].text(-800, 1.05, "C", fontsize = 1.2*labelFontsize )

	# axesW[0].text(-1500, 0.5, "A'", fontsize = 1.2*labelFontsize )
	# axesW[1].text(-800, 0.5, "B'", fontsize = 1.2*labelFontsize )
	# axesW[2].text(-800, 0.5, "C'", fontsize = 1.2*labelFontsize )

	axesRho[0].text(-3000, 1.05, "A", fontsize = 1.2*labelFontsize )
	axesRho[1].text(-1600, 1.05, "B", fontsize = 1.2*labelFontsize )
	axesRho[2].text(-1600, 1.05, "C", fontsize = 1.2*labelFontsize )

	axesW[0].text(-3000, 0.5, "A'", fontsize = 1.2*labelFontsize )
	axesW[1].text(-1600, 0.5, "B'", fontsize = 1.2*labelFontsize )
	axesW[2].text(-1600, 0.5, "C'", fontsize = 1.2*labelFontsize )

	return fig

def load_adj_data( seed, directory, outputDirectory ):

	for d in [0.4, 2.0, 10.0]:
		winit = 0.45

		dataDirectory = directory + "/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_"+str(winit)+"/0_sec/"
			
		adj = scipy.sparse.load_npz( dataDirectory + "synConnections.npz" )
		neuron_locations = np.load( dataDirectory + "STNCenter.npy" )

		np.savez( outputDirectory+"/data_adj_seed_"+str(seed)+"_d_"+str(d)+"_winit_"+str(winit)+".npz", adj=adj.A, neuron_locations=neuron_locations )

## the following function generates Figure 1
def generateFigureConnectivityMatrix( directory ):
	
	import matplotlib.pyplot as plt 
	import matplotlib.gridspec as gridspec


	labelFontsize = 15
	tickFontsize = 12

	seed = 10
	winit = 0.45

	fig = plt.figure()

	gsP1 = gridspec.GridSpec(1, 1)
	gsP1.update( left=0.05, right=0.33, top=0.95, bottom=0.75 )
	axP1 = fig.add_subplot(gsP1[0,0])

	gsP2 = gridspec.GridSpec(1, 1)
	gsP2.update( left=0.36, right=0.64, top=0.95, bottom=0.75 )
	axP2 = fig.add_subplot(gsP2[0,0])

	gsP3 = gridspec.GridSpec(1, 1)
	gsP3.update( left=0.67, right=0.95, top=0.95, bottom=0.75 )
	axP3 = fig.add_subplot(gsP3[0,0])

	L = 5.0 # mm

	# function: conProbability
	def conProbability(d, Cd):
		# connection probability decays exponentially
		p=1/Cd*np.exp(-d/Cd )

		return p

	x = np.linspace(0,L,1000)
	y = conProbability(x, 0.08*L)
	labelP1="$0.08$ L"
	#axP1.plot( x , y, color = "black" )
	axP1.fill_between( x , y1=y, y2=0, color = "black", label="$0.08$ L" )
	# print(np.sum( y*(x[1]-x[0]) ))

	x = np.linspace(0,L,1000)
	y = conProbability(x, 0.4*L)
	labelP2="$0.4$ L"
	#axP2.plot( x , y, color = "black" )
	axP2.fill_between( x , y1=y, y2=0, color = "black", label="$0.4$ L" )

	# print(np.sum( y*(x[1]-x[0]) ))

	x = np.linspace(0,L,1000)
	y = conProbability(x, 2*L)
	labelP3="$2$ L"
	# axP3.plot( x , y, color = "black" )
	axP3.fill_between( x , y1=y, y2=0, color = "black" )

	# print(np.sum( y*(x[1]-x[0]) ))

	axP1.text(3.2,1.8,labelP1, fontsize = tickFontsize)
	axP2.text(3.4,1.8,labelP2, fontsize = tickFontsize)
	axP3.text(3.6,1.8,labelP3, fontsize = tickFontsize)

	for ax in [axP1, axP2, axP3]:
		
		ax.set_xticks([0,1.25,2.5,3.75,5])
		ax.set_yticks([0,0.4,0.8,1.2,1.6,2.0,2.4])
		ax.set_ylim(0,1/0.4)
		
		ax.set_xticklabels(["$0$","","","","$L$"], fontsize = tickFontsize)
		ax.set_yticklabels(["","","","","","",""], fontsize  = labelFontsize)
		ax.set_xlim(0,5)
		ax.set_xlabel("$d_{ij}$", fontsize  = labelFontsize, labelpad = -10)

	axP1.set_ylabel("$p(d_{ij})$", fontsize  = labelFontsize)
	axP1.set_yticklabels(["$0$","","","","","$10/L$",""], fontsize  = labelFontsize)




	gsM1 = gridspec.GridSpec(1, 1)
	gsM1.update( left=0.05, right=0.33, top=0.75, bottom=0.05 )
	ax1 = fig.add_subplot(gsM1[0,0])

	gsM2 = gridspec.GridSpec(1, 1)
	gsM2.update( left=0.36, right=0.64, top=0.75, bottom=0.05 )
	ax2 = fig.add_subplot(gsM2[0,0])

	gsM3 = gridspec.GridSpec(1, 1)
	gsM3.update( left=0.67, right=0.95, top=0.75, bottom=0.05 )
	ax3 = fig.add_subplot(gsM3[0,0])





	# ax1 = fig.add_subplot(131)
	# ax2 = fig.add_subplot(132)
	# ax3 = fig.add_subplot(133)

	d = 0.4
	data_d1 = np.load( directory + "/data_adj_seed_"+str(seed)+"_d_"+str(d)+"_winit_"+str(winit)+".npz", allow_pickle=True )
	ax1.scatter( data_d1["neuron_locations"][ np.nonzero( data_d1["adj"] )[0] ], data_d1["neuron_locations"][ np.nonzero( data_d1["adj"] )[1] ] , s=0.0001, color="black" )
	ax1.set_aspect(1)

	d = 2.0
	data_d2 = np.load( directory + "/data_adj_seed_"+str(seed)+"_d_"+str(d)+"_winit_"+str(winit)+".npz", allow_pickle=True )
	ax2.scatter( data_d2["neuron_locations"][ np.nonzero( data_d2["adj"] )[0] ], data_d2["neuron_locations"][ np.nonzero( data_d2["adj"] )[1] ] , s=0.0001, color="black" )
	ax2.set_aspect(1)

	d = 10.0
	data_d3 = np.load( directory + "/data_adj_seed_"+str(seed)+"_d_"+str(d)+"_winit_"+str(winit)+".npz", allow_pickle=True )
	ax3.scatter( data_d3["neuron_locations"][ np.nonzero( data_d3["adj"] )[0] ], data_d3["neuron_locations"][ np.nonzero( data_d3["adj"] )[1] ] , s=0.0001, color="black" )
	ax3.set_aspect(1)

	for ax in [ax1, ax2, ax3]:
		ax.set_xticks([-2.5,2.5])
		ax.set_xticklabels(["$0$","$L$"], fontsize = tickFontsize )

		ax.set_yticks([-2.5,2.5])
		ax.set_yticklabels(["",""])
		
		ax.set_xlim([-2.5,2.5])
		ax.set_ylim([-2.5,2.5])

		ax.set_xlabel("$x_{\mathrm{pre}}$", fontsize = labelFontsize, labelpad = -10 )

	ax1.set_yticklabels(["$0$","$L$"], fontsize = tickFontsize )
	ax1.set_ylabel("$x_{\mathrm{post}}$", fontsize = labelFontsize, labelpad = -10 )

	ax1.text(-4.0, 6.7, "A", fontsize = 1.2*labelFontsize )
	ax2.text(-3.0, 6.7, "B", fontsize = 1.2*labelFontsize )
	ax3.text(-3.0, 6.7, "C", fontsize = 1.2*labelFontsize )

	ax1.text(-4.0, 2.7, "A'", fontsize = 1.2*labelFontsize )
	ax2.text(-3.0, 2.7, "B'", fontsize = 1.2*labelFontsize )
	ax3.text(-3.0, 2.7, "C'", fontsize = 1.2*labelFontsize )
	
	return fig

def load_trajectories_Kuramoto_and_weights_nonShuffledCR( outputToInitialNetworks , outputPath_nonShuffledCR , outputPath_relAfterNonShuffledCR , pathToBackupOfKuramotoMeanW_nonShuffledCR, Astim, npb ):
	
	for seq in ["0_1_2_3","0_1_3_2","0_2_1_3","0_2_3_1","0_3_1_2","0_3_2_1"]:
		for d in [0.4, 2.0, 10.0]: 
			for fCR in [4.0, 10.0, 21.0]: # Hz
				seed = 10
				
				# file path to initial network, during CR stimulation, and relaxation after cessation of stimulation
				pathToInitialNetwork = outputToInitialNetworks+"/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45"

				pathToSimulationResults = outputPath_nonShuffledCR+"/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45/CR_stim_seq_"+seq+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_"+str(Astim)+"_Tstim_1020.0_dsites_0.25_erw_0.25"

				
				pathToRelaxationAfterCessationOfStimulation =   outputPath_relAfterNonShuffledCR+"/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45/CR_stim_seq_"+seq+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_"+str(Astim)+"_Tstim_1020.0_dsites_0.25_erw_0.25_TstartRelax_6000_sec"

				# specify output directory (loaded trajectories are saved there)
				backupFilename = pathToBackupOfKuramotoMeanW_nonShuffledCR + "/traj_CR_Astim_"+str(Astim)+"_npb_"+str(npb)+"_seq_"+seq+"_seed_"+str(seed)+"_d_"+str(d)+"_fCR_"+str(fCR)+".npz"
				print(d, fCR, seq)
				if os.path.isfile( backupFilename ) == False:
					
					tmin = 4000.0 # sec
					tmax = 20000.0 # sec
					
					sortedListOfDirectories = [pathToInitialNetwork,pathToSimulationResults,pathToRelaxationAfterCessationOfStimulation]
					
					weightData , KuramotoTrajectory = get_Kuramoto_trajectory_and_mean_weight( sortedListOfDirectories, tmin, tmax )
					KuramotoTrajectory[:,0] = 0.001*KuramotoTrajectory[:,0]
					KuramotoTrajectory[:,1] = KuramotoTrajectory[:,1]
					
					np.savez( backupFilename , mw = weightData , rho=KuramotoTrajectory  )
	
def load_trajectories_Kuramoto_and_weights_shuffledCR( outputToInitialNetworks , outputPath_shuffledCR , outputPath_relAfterShuffledCR , pathToBackupOfKuramotoMeanW_shuffledCR, Astim, npb ):

	for seedSeq in [100, 110, 120, 130, 140]:
		for fCR in [4.0, 10.0, 21.0]: # Hz
			for d in [0.4, 2.0, 10.0]:
				seed = 10
				
				Tshuffle = np.round( 1.0/fCR , 4) # seconds
				
				# file path to initial network, during shuffled CR stimulation, and relaxation after cessation of shuffled CR stimulation
				pathToInitialNetwork = outputToInitialNetworks+"/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45"

				pathToSimulationResults = outputPath_shuffledCR+"/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45/shuffled_CR_stim_Tshuffle_"+str(Tshuffle)+"_seedSeq_"+str(seedSeq)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_"+str(Astim)+"_Tstim_1020.0_seedSeq_"+str(seedSeq)+"_dsites_0.25_erw_0.25"
				pathToRelaxationAfterCessationOfStimulation =   outputPath_relAfterShuffledCR+"/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45/shuffled_CR_stim_Tshuffle_"+str(Tshuffle)+"_seedSeq_"+str(seedSeq)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_"+str(Astim)+"_Tstim_1020.0_seedSeq_"+str(seedSeq)+"_dsites_0.25_erw_0.25_TstartRelax_6000_sec"

				# specify output directory (loaded trajectories are saved there)
				backupFilename = pathToBackupOfKuramotoMeanW_shuffledCR + "/traj_shuffledCR_Astim_"+str(Astim)+"_npb_"+str(npb)+"_Tshuffle_"+str(Tshuffle)+"_seedSeq_"+str(seedSeq)+"_seed_"+str(seed)+"_d_"+str(d)+"_fCR_"+str(fCR)+".npz"
				
				print(d, fCR, Tshuffle, seedSeq)
				print(backupFilename)
				if os.path.isfile( backupFilename ) == False:
					
					tmin = 4000.0 # sec
					tmax = 20000.0 # sec
					
					sortedListOfDirectories = [pathToInitialNetwork,pathToSimulationResults,pathToRelaxationAfterCessationOfStimulation]
					
					weightData , KuramotoTrajectory = get_Kuramoto_trajectory_and_mean_weight( sortedListOfDirectories, tmin, tmax )
					# print("weights")
					# print(weightData)

					KuramotoTrajectory[:,0] = 0.001*KuramotoTrajectory[:,0]
					KuramotoTrajectory[:,1] = KuramotoTrajectory[:,1]
					# print("KuramotoTrajectory")
					# print(KuramotoTrajectory)
					# exit()
					np.savez( backupFilename , mw = weightData , rho=KuramotoTrajectory  )

					

def genPlotCMatrix( ax , par, T_sec ):

	import matplotlib.pyplot as plt 
	
	# print("genPlotCMatrix")
	import scipy.sparse
	
	Astim = par['Astim']
	fCR = par['fCR']
	npb = par['npb']
	d = par['d']
	
	if 'seed' in par:
		seed = par['seed']
	else:
		seed = 10
	
	loadFromBackup = False
	
	if par['type'] == "non-shuffled":
		seq = par['seq']
		
		# construct backup file name
		backupFileName_Cmatrix = "data/data_connectivity_diagrams/backup_Cmatrix_seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45_nonShuffled_CR_stim_seq_"+str(seq)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_"+str(Astim)+"_Tstim_1020.0_dsites_0.25_erw_0.25_"+str(T_sec)+"_sec_cMatrix.npz"
		backupFileName_Loc =     "data/data_connectivity_diagrams/backup_Cmatrix_seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45_nonShuffled_CR_stim_seq_"+str(seq)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_"+str(Astim)+"_Tstim_1020.0_dsites_0.25_erw_0.25_"+str(T_sec)+"_sec_STNCenter.npy"
		backupFileName_adj =     "data/data_connectivity_diagrams/backup_Cmatrix_seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45_nonShuffled_CR_stim_seq_"+str(seq)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_"+str(Astim)+"_Tstim_1020.0_dsites_0.25_erw_0.25_"+str(T_sec)+"_sec_adj.npz"

		if os.path.isfile(backupFileName_Cmatrix) == False:
			print("not found", backupFileName_Cmatrix)
		if os.path.isfile(backupFileName_Loc) == False:
			print("not found", backupFileName_Loc)
		if os.path.isfile(backupFileName_adj) == False:
			print("not found", backupFileName_adj)
			
		# check whether backup file name exists, if it exists load data from backup, otherwise load from simulations and generate backup
		if os.path.isfile( backupFileName_Cmatrix ) and os.path.isfile( backupFileName_Loc ) and os.path.isfile(backupFileName_adj):
			loadFromBackup = True
		else:
			cMatrix_name  = par['pathSimData'] + "/CR_stimulation/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45/CR_stim_seq_"+str(seq)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_"+str(Astim)+"_Tstim_1020.0_dsites_0.25_erw_0.25/"+str(T_sec)+"_sec/cMatrix.npz"
			loc_file_name = par['pathSimData'] + "/CR_stimulation/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45/CR_stim_seq_"+str(seq)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_"+str(Astim)+"_Tstim_1020.0_dsites_0.25_erw_0.25/"+str(T_sec)+"_sec/STNCenter.npy"
			adj_name  = par['pathSimData'] + "/CR_stimulation/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45/CR_stim_seq_"+str(seq)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_"+str(Astim)+"_Tstim_1020.0_dsites_0.25_erw_0.25/"+str(T_sec)+"_sec/synConnections.npz"

	elif par['type'] == "shuffled":
		Tshuffle = np.round( 1.0/fCR , 4) # seconds
		seedSeq = par['seedSeq']
		
		# construct backup file name
		backupFileName_Cmatrix = "data/data_connectivity_diagrams/backup_Cmatrix_seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45_shuffled_CR_stim_Tshuffle_"+str(Tshuffle)+"_seedSeq_"+str(seedSeq)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_"+str(Astim)+"_Tstim_1020.0_dsites_0.25_erw_0.25_"+str(T_sec)+"_sec_cMatrix.npz"
		backupFileName_Loc = "data/data_connectivity_diagrams/backup_Cmatrix_seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45_shuffled_CR_stim_Tshuffle_"+str(Tshuffle)+"_seedSeq_"+str(seedSeq)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_"+str(Astim)+"_Tstim_1020.0_dsites_0.25_erw_0.25_"+str(T_sec)+"_sec_STNCenter.npy"
		backupFileName_adj = "data/data_connectivity_diagrams/backup_Cmatrix_seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45_shuffled_CR_stim_Tshuffle_"+str(Tshuffle)+"_seedSeq_"+str(seedSeq)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_"+str(Astim)+"_Tstim_1020.0_dsites_0.25_erw_0.25_"+str(T_sec)+"_sec_adj.npz"
  
		if os.path.isfile(backupFileName_Cmatrix) == False:
			print("not found", backupFileName_Cmatrix)
		if os.path.isfile(backupFileName_Loc) == False:
			print("not found", backupFileName_Loc)
		if os.path.isfile(backupFileName_adj) == False:
			print("not found", backupFileName_adj)

		# check whether backup file name exists, if it exists load data from backup, otherwise load from simulations and generate backup
		if os.path.isfile( backupFileName_Cmatrix ) and os.path.isfile( backupFileName_Loc ) and os.path.isfile(backupFileName_adj):
			loadFromBackup = True
		else:        
			cMatrix_name  = par['pathSimData'] + "/shuffled_CR_stimulation/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45/shuffled_CR_stim_Tshuffle_"+str(Tshuffle)+"_seedSeq_"+str(seedSeq)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_"+str(Astim)+"_Tstim_1020.0_seedSeq_"+str(seedSeq)+"_dsites_0.25_erw_0.25/"+str(T_sec)+"_sec/cMatrix.npz"
			loc_file_name = par['pathSimData'] + "/shuffled_CR_stimulation/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45/shuffled_CR_stim_Tshuffle_"+str(Tshuffle)+"_seedSeq_"+str(seedSeq)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_"+str(Astim)+"_Tstim_1020.0_seedSeq_"+str(seedSeq)+"_dsites_0.25_erw_0.25/"+str(T_sec)+"_sec/STNCenter.npy"
			adj_name      = par['pathSimData'] + "/shuffled_CR_stimulation/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45/shuffled_CR_stim_Tshuffle_"+str(Tshuffle)+"_seedSeq_"+str(seedSeq)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_"+str(Astim)+"_Tstim_1020.0_seedSeq_"+str(seedSeq)+"_dsites_0.25_erw_0.25/"+str(T_sec)+"_sec/synConnections.npz"

	elif par['type'] == "initial": 

		# construct backup file name
		backupFileName_Cmatrix = "data/data_connectivity_diagrams/backup_Cmatrix_initial_seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45_cMatrix.npz"
		backupFileName_Loc = "data/data_connectivity_diagrams/backup_Cmatrix_initial_seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45_STNCenter.npy"
		backupFileName_adj = "data/data_connectivity_diagrams/backup_Cmatrix_initial_seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45_adj.npz"
 
		if os.path.isfile(backupFileName_Cmatrix) == False:
			print("not found", backupFileName_Cmatrix)
		if os.path.isfile(backupFileName_Loc) == False:
			print("not found", backupFileName_Loc)
		if os.path.isfile(backupFileName_adj) == False:
			print("not found", backupFileName_adj)
			
		# check whether backup file name exists, if it exists load data from backup, otherwise load from simulations and generate backup
		if os.path.isfile( backupFileName_Cmatrix ) and os.path.isfile( backupFileName_Loc ) and os.path.isfile(backupFileName_adj):
			print("load files from backup")
			loadFromBackup = True
		else:
			cMatrix_name  = par['pathSimData'] + "/initial_networks/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45/"+str(T_sec)+"_sec/cMatrix.npz"
			loc_file_name = par['pathSimData'] + "/initial_networks/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45/"+str(T_sec)+"_sec/STNCenter.npy"
			adj_name      = par['pathSimData'] + "/initial_networks/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45/"+str(T_sec)+"_sec/synConnections.npz"

	if loadFromBackup == False:
		STN_center = np.load( loc_file_name )    
		cMatrix = scipy.sparse.load_npz( cMatrix_name )
		adj = scipy.sparse.load_npz( adj_name )
		
		# save backup file
		# print("creating backup file" + backupFileName)
		scipy.sparse.save_npz( backupFileName_Cmatrix , cMatrix )
		scipy.sparse.save_npz( backupFileName_adj , adj )
		np.save( backupFileName_Loc , STN_center )
	else:
		cMatrix    = scipy.sparse.load_npz( backupFileName_Cmatrix  )
		STN_center = np.load( backupFileName_Loc )
		adj = scipy.sparse.load_npz( backupFileName_adj )



	# print( cMatrix.A )
	# print(np.nonzero( cMatrix ) )

	x = STN_center[ np.nonzero( cMatrix )[0] ]
	y = STN_center[ np.nonzero( cMatrix )[1] ]
	z = np.array( cMatrix[ np.nonzero( cMatrix )[0] , np.nonzero( cMatrix )[1] ] )[0]


	ax.scatter( y, x, c = z, cmap="gray_r",  vmin=0, vmax=1, s = 0.001 )
	# plt.scatter( y, x, c = z, cmap="gray_r",  vmin=0, vmax=1, s = 0.001 )
	# exit()
	 
	for loc in [-1.25,0,1.25]:
		ax.axhline( loc, lw=0.5, color="red", ls="--" )
		ax.axvline( loc, lw=0.5, color="red", ls="--" )
	
	# ax.imshow( cMatrix.A, cmap="gray_r",  vmin=0, vmax=1, origin = "lower" )
	
	ax.set_xlim(-2.5,2.5)  
	ax.set_ylim(-2.5,2.5)    

	return np.mean( cMatrix[:1000,:1000] )/np.mean( adj[:1000,:1000] )

def genRasterPlot( ax , par ):
	
	import matplotlib.pyplot as plt 

	Astim = par['Astim']
	fCR = par['fCR']
	npb = par['npb']
	d = par['d']
	
	if 'seed' in par:
		seed = par['seed']
	else:
		seed = 10
	
	if par['type'] == "non-shuffled":
		Tmin = 5999
		seq = par['seq']

		# specify file paths
		neuronCenter_filename = par['pathSimData'] + "/CR_stimulation/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45/CR_stim_seq_"+str(seq)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_"+str(Astim)+"_Tstim_1020.0_dsites_0.25_erw_0.25/6000_sec/STNCenter.npy"
		spk_name     = par['pathSimData'] + "/CR_stimulation/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45/CR_stim_seq_"+str(seq)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_"+str(Astim)+"_Tstim_1020.0_dsites_0.25_erw_0.25/spikeTimes_6000_sec.npy"
		
		backupFilename = "data/data_connectivity_diagrams/backupRaster_seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45_nonShuffled_CR_stim_seq_"+str(seq)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_"+str(Astim)+"_Tstim_1020.0_dsites_0.25_erw_0.25_6000_sec.npy.npz"

	elif par['type'] == "shuffled":
		Tmin = 5999
		Tshuffle = np.round( 1.0/fCR , 4) # seconds
		seedSeq = par['seedSeq']

		# specify file paths
		neuronCenter_filename = par['pathSimData'] + "/shuffled_CR_stimulation/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45/shuffled_CR_stim_Tshuffle_"+str(Tshuffle)+"_seedSeq_"+str(seedSeq)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_"+str(Astim)+"_Tstim_1020.0_seedSeq_"+str(seedSeq)+"_dsites_0.25_erw_0.25/6000_sec/STNCenter.npy"
		spk_name     = par['pathSimData'] + "/shuffled_CR_stimulation/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45/shuffled_CR_stim_Tshuffle_"+str(Tshuffle)+"_seedSeq_"+str(seedSeq)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_"+str(Astim)+"_Tstim_1020.0_seedSeq_"+str(seedSeq)+"_dsites_0.25_erw_0.25/spikeTimes_6000_sec.npy"

		backupFilename = "data/data_connectivity_diagrams/backupRaster_seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45_shuffled_CR_stim_Tshuffle_"+str(Tshuffle)+"_seedSeq_"+str(seedSeq)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_"+str(Astim)+"_Tstim_1020.0_seedSeq_"+str(seedSeq)+"_dsites_0.25_erw_0.25_6000_sec.npy.npz"
		
	elif par['type'] == "initial": 
		Tmin = 4999

		# specify file paths
		neuronCenter_filename = par['pathSimData'] + "/initial_networks/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45/5000_sec/STNCenter.npy"
		spk_name     = par['pathSimData'] + "/initial_networks/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45/spikeTimes_5000_sec.npy"

		backupFilename = "data/data_connectivity_diagrams/backupRaster_initial_seed_"+str(seed)+"_d_"+str(d)+"_mw_init_0.45_5000_sec.npy.npz"
		
	if os.path.isfile( backupFilename ):
		dataSet = np.load( backupFilename )
		plotSpk = dataSet["plotSpk"]
		neuronCenters = dataSet["neuronCenters"]
	else:
		print(backupFilename, "not found")
		# load spike train
		spk = np.load( spk_name )
		# load neuronCenter coordinates
		neuronCenters = np.load( neuronCenter_filename )

		# only consider spikes in plot range
		Tmax = Tmin + 1
		plotSpk = spk[ np.logical_and( 0.0001*spk[:,1] >= Tmin , 0.0001*spk[:,1] < Tmax ) ]
		plotSpk = plotSpk[ plotSpk[:,0]<1000 ]
		np.savez( backupFilename, plotSpk=plotSpk, neuronCenters=neuronCenters )
	
	ax.scatter( 0.0001*plotSpk[:,1] , neuronCenters[plotSpk[:,0]], color="black",  s=par['s'] )
	
	# ax.set_ylim(0,1000)
	ax.set_ylim(-2.5,2.5)

# the following function generates Figure 4
def gen_Figure_trajectories_Kuramoto_weight( backupDirectory ):

	import matplotlib.pyplot as plt 
	
	tickFontsize = 12
	labelFontsize = 15

	fig = plt.figure( figsize = (8,7) )

	tmin = 4500 # sec
	tmax = 20000 # sec
	tmax = 13000 # sec
	# tmax = 7000 # sec

	# axes for plots of Kuramoto
	ax_rho_d1_f1 = fig.add_subplot(631)
	ax_rho_d2_f1 = fig.add_subplot(632)
	ax_rho_d3_f1 = fig.add_subplot(633)

	ax_rho_d1_f2 = fig.add_subplot(6,3,7)
	ax_rho_d2_f2 = fig.add_subplot(6,3,8)
	ax_rho_d3_f2 = fig.add_subplot(6,3,9)

	ax_rho_d1_f3 = fig.add_subplot(6,3,13)
	ax_rho_d2_f3 = fig.add_subplot(6,3,14)
	ax_rho_d3_f3 = fig.add_subplot(6,3,15)


	# axes for plots of mean weight
	ax_mw_d1_f1 = fig.add_subplot(634)
	ax_mw_d2_f1 = fig.add_subplot(635)
	ax_mw_d3_f1 = fig.add_subplot(636)

	ax_mw_d1_f2 = fig.add_subplot(6,3,10)
	ax_mw_d2_f2 = fig.add_subplot(6,3,11)
	ax_mw_d3_f2 = fig.add_subplot(6,3,12)

	ax_mw_d1_f3 = fig.add_subplot(6,3,16)
	ax_mw_d2_f3 = fig.add_subplot(6,3,17)
	ax_mw_d3_f3 = fig.add_subplot(6,3,18)

	axes_rho = [[ ax_rho_d1_f1, ax_rho_d1_f2, ax_rho_d1_f3 ],[ ax_rho_d2_f1, ax_rho_d2_f2, ax_rho_d2_f3 ],[ ax_rho_d3_f1, ax_rho_d3_f2, ax_rho_d3_f3 ]]
	axes_mw  = [[ ax_mw_d1_f1,  ax_mw_d1_f2,  ax_mw_d1_f3  ],[ ax_mw_d2_f1,  ax_mw_d2_f2,  ax_mw_d2_f3  ],[ ax_mw_d3_f1,  ax_mw_d3_f2,  ax_mw_d3_f3 ]]

	seed = 10
	Astim = 1.0
	npb = 3
	tSimOnset = 5000 # sec
	# system length scale 
	L = 5.0 # mm

	# print("Astim="+str(Astim)+" npb="+str(npb)+" seed="+str(seed))

	fCR_array = [4.0, 10.0, 21.0] # Hz
	d_array   = [0.4, 2.0, 10.0]

	seq_array = ["0_1_2_3","0_1_3_2","0_2_1_3","0_2_3_1","0_3_1_2","0_3_2_1"]
	color_seq = [ 'magenta','slateblue','orange','mediumspringgreen','cyan','dodgerblue']
	labels_seq = ["I,II,III,IV","I,II,IV,III","I,III,II,IV","I,III,IV,II","I,IV,II,III","I,IV,III,II"]
	# color_seq = ["0.6","0.6","0.6","0.6","0.6","0.6","0.6"]
	color_CRRVS = "0.3"

	# seedSeq_array = [100, 110, 120, 130, 140]
	seedSeq_array = [100]

	for kfCR in range( len(fCR_array) ): 
		for kd in range( len(d_array) ):

			# plot results for non-shuffled CR stimulation
			for kseq in range(len(seq_array)):

				labels_seq[kseq]
				seq = seq_array[kseq]

				fCR = fCR_array[ kfCR ]
				d = d_array[ kd ]

				# print(fCR,d,seq)
				backupFilename = backupDirectory + "/traj_CR_Astim_"+str(Astim)+"_npb_"+str(npb)+"_seq_"+seq+"_seed_"+str(seed)+"_d_"+str(d)+"_fCR_"+str(fCR)+".npz"

				if os.path.isfile( backupFilename ):
					# print("backup found")
					data_traj = np.load( backupFilename )

					weightData         = data_traj['mw']
					KuramotoTrajectory = data_traj['rho']

					tRho = KuramotoTrajectory[:,0]
					Rho  = KuramotoTrajectory[:,1]

					# print(tRho)

					tmw = weightData[:,0]
					mw  = weightData[:,1]

					if kseq == 0:
						RhoInit  = Rho[ tRho <= tSimOnset ]
						tRhoInit = tRho[ tRho <= tSimOnset ] 

						mwInit  = mw[ tmw <= tSimOnset ]
						tmwInit = tmw[ tmw <= tSimOnset ] 

						axes_rho[kd][kfCR].plot( tRhoInit , RhoInit, color = "black", lw=0.9 )
						axes_mw[kd][kfCR].plot(  tmwInit , mwInit, color = "black", lw=0.9  )

					Rho  = Rho[ tRho >= tSimOnset-1 ]
					tRho = tRho[ tRho >= tSimOnset-1 ] 

					mw  = mw[ tmw >= tSimOnset-1 ]
					tmw = tmw[ tmw >= tSimOnset-1 ]                     



					axes_rho[kd][kfCR].plot( tRho , Rho, color = color_seq[kseq], lw=0.9, label=labels_seq[kseq] )
					axes_mw[kd][kfCR].plot( tmw , mw, color = color_seq[kseq], lw=0.9, label=labels_seq[kseq] )

			# plot results for shuffled CR stimulation
			for kSeedSeq in range(len(seedSeq_array)):

				seedSeq = seedSeq_array[kSeedSeq]

				fCR = fCR_array[ kfCR ]
				d = d_array[ kd ]

				# shuffle period
				Tshuffle = np.round( 1.0/fCR , 4) # seconds

				# print(fCR,d,Tshuffle)

				backupFilename = backupDirectory + "/traj_shuffledCR_Astim_"+str(Astim)+"_npb_"+str(npb)+"_Tshuffle_"+str(Tshuffle)+"_seedSeq_"+str(seedSeq)+"_seed_"+str(seed)+"_d_"+str(d)+"_fCR_"+str(fCR)+".npz"

				if os.path.isfile( backupFilename ):
					# print("backup found")
					data_traj = np.load( backupFilename )

					weightData         = data_traj['mw']
					KuramotoTrajectory = data_traj['rho']
					# print(weightData)
					tRho = KuramotoTrajectory[:,0]
					Rho  = KuramotoTrajectory[:,1]

					tmw = weightData[:,0]
					mw  = weightData[:,1]

					Rho  = Rho[ tRho >= tSimOnset-1 ]
					tRho = tRho[ tRho >= tSimOnset-1 ] 

					mw  = mw[ tmw >= tSimOnset-1 ]
					tmw = tmw[ tmw >= tSimOnset-1 ]                     

					axes_rho[kd][kfCR].plot( tRho , Rho, color = color_CRRVS, lw=0.9, label="shuffled" )
					axes_mw[kd][kfCR].plot( tmw , mw, color = color_CRRVS, lw=0.9, label="shuffled" )


		axes_mw[0][kfCR].text( tmin-3500, 0.27,  "$f_{\mathrm{CR}}="+str(int(fCR))+"$ Hz", fontsize = tickFontsize, rotation = 90 )
		# axes_mw[2][0].legend(loc=0,frameon=False)

	xticks = [5000,6000,7000,8000,9000,10000,11000,12000]

	ytickRhos = [0,0.5,1]
	ytickMws  = [0,0.25,0.5,0.75,1]

	yminMws  = 0
	ymaxMws  = 0.55

	for axRow in axes_rho:
		for ax in axRow:
			ax.fill_between( x = [5000,6000], y1 = [0,0], y2=[1.2,1.2], color = "lightcoral", alpha = 0.2, zorder = -1 )

			ax.set_xticks( xticks )
			ax.set_xticklabels( ["" for x in xticks ] )

			ax.set_yticks( ytickRhos )
			ax.set_yticklabels( ["" for x in ytickRhos ] )

			ax.set_xlim(tmin,tmax)
			ax.set_ylim(0,1.2)


	for ax_rho in [ax_rho_d1_f1,ax_rho_d1_f2,ax_rho_d1_f3]:
		ax_rho.set_yticklabels( ["$0$", "", "$1$"], fontsize = tickFontsize )
		ax_rho.set_ylabel( r"$\rho$", fontsize = tickFontsize )



	for axRow in axes_mw:
		for ax in axRow:
			ax.fill_between( x = [5000,6000], y1 = [0,0], y2=[1,1], color = "lightcoral", alpha = 0.2, zorder = -1 )

			ax.set_xticks( xticks )
			ax.set_xticklabels( ["" for x in xticks ] )

			ax.set_yticks( ytickMws )
			ax.set_yticklabels( ["" for x in ytickMws ] )

			ax.set_xlim(tmin,tmax)
			ax.set_ylim(yminMws,ymaxMws)

	for ax_mw in [ax_mw_d1_f1,ax_mw_d1_f2,ax_mw_d1_f3]:
		ax_mw.set_yticklabels( ["$0$","","$0.5$","","$1$"], fontsize = tickFontsize )
		ax_mw.set_ylabel( r"$\langle w \rangle$", fontsize = tickFontsize, labelpad = -8 )

	for ax_mw in [ax_mw_d1_f3,ax_mw_d2_f3,ax_mw_d3_f3]:
		# [5000,6000,7000,8000,9000,10000,11000,12000]
		ax_mw.set_xticklabels( ["$0$", "", "", "$3000$","", "","$6000$", ""], fontsize = tickFontsize )
		ax_mw.set_xlabel( "$t$ (sec)", fontsize = tickFontsize )



	ax_rho_d1_f1.set_title( "$s="+str(d_array[0]/L)+"$ L $(0.32 d)$", fontsize = tickFontsize )
	ax_rho_d2_f1.set_title( "$s="+str(d_array[1]/L)+"$ L $(1.6 d)$", fontsize = tickFontsize )
	ax_rho_d3_f1.set_title( "$s="+str(d_array[2]/L)+"$ L $(8 d)$", fontsize = tickFontsize )

	offsetX = -500
	offsetX = -1500
	offset2 = -800
	ax_rho_d1_f1.text( 3700+offsetX, 1.15, "A", fontsize = 1.2*labelFontsize )
	ax_rho_d2_f1.text( 4100+offsetX-offset2, 1.15, "B", fontsize = 1.2*labelFontsize )
	ax_rho_d3_f1.text( 4100+offsetX-offset2, 1.15, "C", fontsize = 1.2*labelFontsize )

	ax_rho_d1_f2.text( 3700+offsetX, 1.15, "D", fontsize = 1.2*labelFontsize )
	ax_rho_d2_f2.text( 4100+offsetX-offset2, 1.15, "E", fontsize = 1.2*labelFontsize )
	ax_rho_d3_f2.text( 4100+offsetX-offset2, 1.15, "F", fontsize = 1.2*labelFontsize )

	ax_rho_d1_f3.text( 3700+offsetX, 1.15, "G", fontsize = 1.2*labelFontsize )
	ax_rho_d2_f3.text( 4100+offsetX-offset2, 1.15, "H", fontsize = 1.2*labelFontsize )
	ax_rho_d3_f3.text( 4100+offsetX-offset2, 1.15, "I", fontsize = 1.2*labelFontsize )

	return fig

# the following function generates Figure 5
def gen_Figure_connectivity_diagrams( pathSimData ):

	import matplotlib.pyplot as plt 
	import matplotlib.gridspec as gridspec

	fig = plt.figure( figsize = (8,6) )

	tickFontsize = 12
	labelFontsize = 15

	x1 = 0.05
	x2 = 0.35
	x3 = 0.42
	x4 = 0.59
	x5 = 0.6
	x6 = 0.77
	x7 = 0.78
	x8 = 0.95

	y1 = 0.95
	y2 = 0.75
	y3 = 0.7
	y4 = 0.5
	y5 = 0.475
	y6 = 0.275
	y7 = 0.25
	y8 = 0.05

	gsRaster_prior = gridspec.GridSpec(1, 1)
	gsRaster_prior.update( left=x1, right=x2, top=y1-0.02, bottom=y2+0.02 )
	axRaster_prior = fig.add_subplot(gsRaster_prior[0,0])

	gsRaster_seq1= gridspec.GridSpec(1, 1)
	gsRaster_seq1.update( left=x1, right=x2, top=y3-0.02, bottom=y4+0.02 )
	axRaster_seq1 = fig.add_subplot(gsRaster_seq1[0,0])

	gsRaster_seq2 = gridspec.GridSpec(1, 1)
	gsRaster_seq2.update( left=x1, right=x2, top=y5-0.02, bottom=y6+0.02 )
	axRaster_seq2 = fig.add_subplot(gsRaster_seq2[0,0])

	gsRaster_CRRVS = gridspec.GridSpec(1, 1)
	gsRaster_CRRVS.update( left=x1, right=x2, top=y7-0.02, bottom=y8+0.02 )
	axRaster_CRRVS = fig.add_subplot(gsRaster_CRRVS[0,0])

	# panels for cMatrix prior to stimulation
	gsCMatrixD1_prior = gridspec.GridSpec(1, 1)
	gsCMatrixD1_prior.update( left=x3, right=x4, top=y1, bottom=y2 )
	axCMatrixD1_prior = fig.add_subplot(gsCMatrixD1_prior[0,0])

	gsCMatrixD2_prior = gridspec.GridSpec(1, 1)
	gsCMatrixD2_prior.update( left=x5, right=x6, top=y1, bottom=y2 )
	axCMatrixD2_prior = fig.add_subplot(gsCMatrixD2_prior[0,0])

	gsCMatrixD3_prior = gridspec.GridSpec(1, 1)
	gsCMatrixD3_prior.update( left=x7, right=x8, top=y1, bottom=y2 )
	axCMatrixD3_prior = fig.add_subplot(gsCMatrixD3_prior[0,0])

	# panels for cMatrix after non-shuffled CR stimulation with Seq 1
	gsCMatrixD1_seq1 = gridspec.GridSpec(1, 1)
	gsCMatrixD1_seq1.update( left=x3, right=x4, top=y3, bottom=y4 )
	axCMatrixD1_seq1 = fig.add_subplot(gsCMatrixD1_seq1[0,0])

	gsCMatrixD2_seq1 = gridspec.GridSpec(1, 1)
	gsCMatrixD2_seq1.update( left=x5, right=x6, top=y3, bottom=y4 )
	axCMatrixD2_seq1 = fig.add_subplot(gsCMatrixD2_seq1[0,0])

	gsCMatrixD3_seq1 = gridspec.GridSpec(1, 1)
	gsCMatrixD3_seq1.update( left=x7, right=x8, top=y3, bottom=y4 )
	axCMatrixD3_seq1 = fig.add_subplot(gsCMatrixD3_seq1[0,0])

	# panels for cMatrix after non-shuffled CR stimulation with Seq 2
	gsCMatrixD1_seq2 = gridspec.GridSpec(1, 1)
	gsCMatrixD1_seq2.update( left=x3, right=x4, top=y5, bottom=y6 )
	axCMatrixD1_seq2 = fig.add_subplot(gsCMatrixD1_seq2[0,0])

	gsCMatrixD2_seq2 = gridspec.GridSpec(1, 1)
	gsCMatrixD2_seq2.update( left=x5, right=x6, top=y5, bottom=y6 )
	axCMatrixD2_seq2 = fig.add_subplot(gsCMatrixD2_seq2[0,0])

	gsCMatrixD3_seq2 = gridspec.GridSpec(1, 1)
	gsCMatrixD3_seq2.update( left=x7, right=x8, top=y5, bottom=y6 )
	axCMatrixD3_seq2 = fig.add_subplot(gsCMatrixD3_seq2[0,0])

	# panels for cMatrix after to shuffled CR stimulation
	gsCMatrixD1_SCR = gridspec.GridSpec(1, 1)
	gsCMatrixD1_SCR.update( left=x3, right=x4, top=y7, bottom=y8 )
	axCMatrixD1_SCR = fig.add_subplot(gsCMatrixD1_SCR[0,0])

	gsCMatrixD2_SCR = gridspec.GridSpec(1, 1)
	gsCMatrixD2_SCR.update( left=x5, right=x6, top=y7, bottom=y8 )
	axCMatrixD2_SCR = fig.add_subplot(gsCMatrixD2_SCR[0,0])

	gsCMatrixD3_SCR = gridspec.GridSpec(1, 1)
	gsCMatrixD3_SCR.update( left=x7, right=x8, top=y7, bottom=y8 )
	axCMatrixD3_SCR = fig.add_subplot(gsCMatrixD3_SCR[0,0])

	# parameter values
	d_values = [0.4, 2.0, 10.0]

	axes_prior = [axCMatrixD1_prior, axCMatrixD2_prior, axCMatrixD3_prior]
	axes_seq1 = [axCMatrixD1_seq1, axCMatrixD2_seq1, axCMatrixD3_seq1]
	axes_seq2 = [axCMatrixD1_seq2, axCMatrixD2_seq2, axCMatrixD3_seq2]
	axes_SCR = [axCMatrixD1_SCR, axCMatrixD2_SCR, axCMatrixD3_SCR]

	# generate raster plots
	par = {}
	par['Astim'] = 1.0
	par['fCR'] = 10.0 # Hz
	par['npb'] = 3
	par['s'] = 0.01
	par['pathSimData'] = pathSimData

	par['type'] = "initial"
	par['d'] = 0.4
	genRasterPlot( axRaster_prior , par )

	for kd in range( len(d_values) ):
		par['d']  = d_values[kd]

		par['type'] = "initial"
		# genRasterPlot( axRaster_prior , par )
		mw = genPlotCMatrix( axes_prior[kd] , par, 5000 )
		axes_prior[kd].text(-2.4,1.8,r"$"+str( np.round(mw,2) )+"$", fontsize = tickFontsize )
		axes_prior[kd].set_aspect(1)
		
		par['seq'] = "0_1_2_3"
		par['type'] = "non-shuffled"
		if par['d'] == 0.4:
			genRasterPlot( axRaster_seq1 , par )
		mw = genPlotCMatrix( axes_seq1[kd] , par, 6000 )
		axes_seq1[kd].text(-2.4,1.8,r"$"+str( np.round(mw,2) )+"$", fontsize = tickFontsize )
		axes_seq1[kd].set_aspect(1)

		par['seq'] = "0_3_1_2"
		par['type'] = "non-shuffled"
		if par['d'] == 0.4:
			genRasterPlot( axRaster_seq2 , par )
		mw = genPlotCMatrix( axes_seq2[kd] , par, 6000 )
		axes_seq2[kd].text(-1.4,1.8,r"$"+str( np.round(mw,2) )+"$", fontsize = tickFontsize )
		axes_seq2[kd].set_aspect(1)
		
		par['type'] = "shuffled"
		par['seedSeq'] = 100
		if par['d'] == 0.4:
			genRasterPlot( axRaster_CRRVS , par )
		mw = genPlotCMatrix( axes_SCR[kd] , par, 6000 )
		axes_SCR[kd].text(-2.4,1.8,r"$"+str( np.round(mw,2) )+"$", fontsize = tickFontsize )
		axes_SCR[kd].set_aspect(1)

		# set axes raster
		for ax in [axRaster_prior, axRaster_seq1, axRaster_seq2, axRaster_CRRVS]:
			ax.set_yticks([-2.5,2.5])
			ax.set_yticklabels(["$0$","$L$"], fontsize = tickFontsize)
	 
			ax.set_xticks([])
			ax.set_xticklabels([], fontsize = tickFontsize)
	 
			ax.set_frame_on(False)
			ax.get_yaxis().tick_left()
	  
			# ax.set_ylim(-105,1100)
			ax.set_ylim(-2.8,2.6)
	 
		for ax in [axRaster_prior]:
			
			# ax.plot([4999.8,4999.9],[-90,-90], lw=3, color="black")
			ax.plot([4999.8,4999.9],[-2.65,-2.65], lw=3, color="black")
			ax.set_xlim(4999,5000)
		   
	#     for ax in [axRaster_seq1, axRaster_seq2, axRaster_CRRVS]:
	#         ax.plot([5989.8,5989.9],[-90,-90], lw=3, color="black")
	#         ax.set_xlim(5989,5990)
		for ax in [axRaster_seq1, axRaster_seq2, axRaster_CRRVS]:
			# ax.plot([5999.8,5999.9],[-90,-90], lw=3, color="black")
			ax.plot([5999.8,5999.9],[-2.65,-2.65], lw=3, color="black")
			ax.set_xlim(5999,6000)
			
		
		#### specify layout for panels shown cMatrix
		# set axes cMatrix
		for ax in [axCMatrixD1_prior, axCMatrixD2_prior, axCMatrixD3_prior,axCMatrixD1_seq1, axCMatrixD2_seq1, axCMatrixD3_seq1,axCMatrixD1_seq2, axCMatrixD2_seq2, axCMatrixD3_seq2,axCMatrixD1_SCR, axCMatrixD2_SCR, axCMatrixD3_SCR]:
	 
			# ax.set_xticks([0,1000])
			ax.set_xticks([-2.5,2.5])
			ax.set_xticklabels(["",""], fontsize = tickFontsize)

			# ax.set_yticks([0,1000])
			ax.set_yticks([-2.5,2.5])
			ax.set_yticklabels(["",""], fontsize = tickFontsize)

		for ax in [axCMatrixD1_prior, axCMatrixD1_seq1, axCMatrixD1_seq2, axCMatrixD1_SCR]:
			ax.set_yticklabels(["$0$","$L$"], fontsize = tickFontsize)
			ax.set_ylabel("$x_{\mathrm{post}}$", fontsize = labelFontsize, labelpad=-5)
			# ax.set_yticklabels(["$0$","$N$"], fontsize = tickFontsize)
		
		for ax in [axCMatrixD1_SCR, axCMatrixD2_SCR, axCMatrixD3_SCR]:
			ax.set_xticklabels(["$0$","$L$"], fontsize = tickFontsize)
			ax.set_xlabel("$x_{\mathrm{pre}}$", fontsize = labelFontsize, labelpad=-5)
			# ax.set_xticklabels(["$0$","$N$"], fontsize = tickFontsize)

			   
	axCMatrixD1_prior.set_title( "$0.08 L (0.32 d)$", fontsize = tickFontsize )
	axCMatrixD2_prior.set_title( "$0.4 L (1.6 d)$", fontsize = tickFontsize )
	axCMatrixD3_prior.set_title( "$2.0 L (8 d)$", fontsize = tickFontsize )
		
	# panel labels
	# axRaster_prior.text(4998.8,1100,"A", fontsize = 1.2*labelFontsize)
	# axRaster_seq1.text(5988.8,1100,"E", fontsize = 1.2*labelFontsize)
	# axRaster_seq2.text(5988.8,1100,"I", fontsize = 1.2*labelFontsize)
	# axRaster_CRRVS.text(5988.8,1100,"M", fontsize = 1.2*labelFontsize)

	# axRaster_prior.text(4998.8,1100,"A", fontsize = 1.2*labelFontsize)
	# axRaster_seq1.text(5998.8,1100,"E", fontsize = 1.2*labelFontsize)
	# axRaster_seq2.text(5998.8,1100,"I", fontsize = 1.2*labelFontsize)
	# axRaster_CRRVS.text(5998.8,1100,"M", fontsize = 1.2*labelFontsize)

	axRaster_prior.text(4998.8,2.6,"A", fontsize = 1.2*labelFontsize)
	axRaster_seq1.text(5998.8,2.6,"E", fontsize = 1.2*labelFontsize)
	axRaster_seq2.text(5998.8,2.6,"I", fontsize = 1.2*labelFontsize)
	axRaster_CRRVS.text(5998.8,2.6,"M", fontsize = 1.2*labelFontsize)


	axCMatrixD1_prior.text(-4.1,2.8,"B", fontsize = 1.2*labelFontsize)
	axCMatrixD2_prior.text(-3.3,2.8,"C", fontsize = 1.2*labelFontsize)
	axCMatrixD3_prior.text(-3.3,2.8,"D", fontsize = 1.2*labelFontsize)
	  
	axCMatrixD1_seq1.text(-4.1,2.7,"F", fontsize = 1.2*labelFontsize)
	axCMatrixD2_seq1.text(-3.3,2.7,"G", fontsize = 1.2*labelFontsize)
	axCMatrixD3_seq1.text(-3.3,2.7,"H", fontsize = 1.2*labelFontsize)
	 
	axCMatrixD1_seq2.text(-4.1,2.5,"J", fontsize = 1.2*labelFontsize)
	axCMatrixD2_seq2.text(-3.3,2.5,"K", fontsize = 1.2*labelFontsize)
	axCMatrixD3_seq2.text(-3.3,2.5,"L", fontsize = 1.2*labelFontsize)

	axCMatrixD1_SCR.text(-4.2,2.5,"N", fontsize = 1.2*labelFontsize)
	axCMatrixD2_SCR.text(-3.4,2.5,"O", fontsize = 1.2*labelFontsize)
	axCMatrixD3_SCR.text(-3.3,2.5,"P", fontsize = 1.2*labelFontsize)

	return fig

# the following function generates Figure 6
def gen_Figure_connectivity_diagrams_single_pulse( pathSimData ):

	import matplotlib.pyplot as plt 
	import matplotlib.gridspec as gridspec

	fig = plt.figure( figsize = (8,6) )

	tickFontsize = 12
	labelFontsize = 15

	x1 = 0.05
	x2 = 0.35
	x3 = 0.42
	x4 = 0.59
	x5 = 0.6
	x6 = 0.77
	x7 = 0.78
	x8 = 0.95

	y1 = 0.95
	y2 = 0.75
	y3 = 0.7
	y4 = 0.5
	y5 = 0.475
	y6 = 0.275
	y7 = 0.25
	y8 = 0.05

	gsRaster_prior = gridspec.GridSpec(1, 1)
	gsRaster_prior.update( left=x1, right=x2, top=y1-0.02, bottom=y2+0.02 )
	axRaster_prior = fig.add_subplot(gsRaster_prior[0,0])

	gsRaster_seq1= gridspec.GridSpec(1, 1)
	gsRaster_seq1.update( left=x1, right=x2, top=y3-0.02, bottom=y4+0.02 )
	axRaster_seq1 = fig.add_subplot(gsRaster_seq1[0,0])

	gsRaster_seq2 = gridspec.GridSpec(1, 1)
	gsRaster_seq2.update( left=x1, right=x2, top=y5-0.02, bottom=y6+0.02 )
	axRaster_seq2 = fig.add_subplot(gsRaster_seq2[0,0])

	gsRaster_CRRVS = gridspec.GridSpec(1, 1)
	gsRaster_CRRVS.update( left=x1, right=x2, top=y7-0.02, bottom=y8+0.02 )
	axRaster_CRRVS = fig.add_subplot(gsRaster_CRRVS[0,0])

	# panels for cMatrix prior to stimulation
	gsCMatrixD1_prior = gridspec.GridSpec(1, 1)
	gsCMatrixD1_prior.update( left=x3, right=x4, top=y1, bottom=y2 )
	axCMatrixD1_prior = fig.add_subplot(gsCMatrixD1_prior[0,0])

	gsCMatrixD2_prior = gridspec.GridSpec(1, 1)
	gsCMatrixD2_prior.update( left=x5, right=x6, top=y1, bottom=y2 )
	axCMatrixD2_prior = fig.add_subplot(gsCMatrixD2_prior[0,0])

	gsCMatrixD3_prior = gridspec.GridSpec(1, 1)
	gsCMatrixD3_prior.update( left=x7, right=x8, top=y1, bottom=y2 )
	axCMatrixD3_prior = fig.add_subplot(gsCMatrixD3_prior[0,0])

	# panels for cMatrix after non-shuffled CR stimulation with Seq 1
	gsCMatrixD1_seq1 = gridspec.GridSpec(1, 1)
	gsCMatrixD1_seq1.update( left=x3, right=x4, top=y3, bottom=y4 )
	axCMatrixD1_seq1 = fig.add_subplot(gsCMatrixD1_seq1[0,0])

	gsCMatrixD2_seq1 = gridspec.GridSpec(1, 1)
	gsCMatrixD2_seq1.update( left=x5, right=x6, top=y3, bottom=y4 )
	axCMatrixD2_seq1 = fig.add_subplot(gsCMatrixD2_seq1[0,0])

	gsCMatrixD3_seq1 = gridspec.GridSpec(1, 1)
	gsCMatrixD3_seq1.update( left=x7, right=x8, top=y3, bottom=y4 )
	axCMatrixD3_seq1 = fig.add_subplot(gsCMatrixD3_seq1[0,0])

	# panels for cMatrix after non-shuffled CR stimulation with Seq 2
	gsCMatrixD1_seq2 = gridspec.GridSpec(1, 1)
	gsCMatrixD1_seq2.update( left=x3, right=x4, top=y5, bottom=y6 )
	axCMatrixD1_seq2 = fig.add_subplot(gsCMatrixD1_seq2[0,0])

	gsCMatrixD2_seq2 = gridspec.GridSpec(1, 1)
	gsCMatrixD2_seq2.update( left=x5, right=x6, top=y5, bottom=y6 )
	axCMatrixD2_seq2 = fig.add_subplot(gsCMatrixD2_seq2[0,0])

	gsCMatrixD3_seq2 = gridspec.GridSpec(1, 1)
	gsCMatrixD3_seq2.update( left=x7, right=x8, top=y5, bottom=y6 )
	axCMatrixD3_seq2 = fig.add_subplot(gsCMatrixD3_seq2[0,0])

	# panels for cMatrix after to shuffled CR stimulation
	gsCMatrixD1_SCR = gridspec.GridSpec(1, 1)
	gsCMatrixD1_SCR.update( left=x3, right=x4, top=y7, bottom=y8 )
	axCMatrixD1_SCR = fig.add_subplot(gsCMatrixD1_SCR[0,0])

	gsCMatrixD2_SCR = gridspec.GridSpec(1, 1)
	gsCMatrixD2_SCR.update( left=x5, right=x6, top=y7, bottom=y8 )
	axCMatrixD2_SCR = fig.add_subplot(gsCMatrixD2_SCR[0,0])

	gsCMatrixD3_SCR = gridspec.GridSpec(1, 1)
	gsCMatrixD3_SCR.update( left=x7, right=x8, top=y7, bottom=y8 )
	axCMatrixD3_SCR = fig.add_subplot(gsCMatrixD3_SCR[0,0])

	# parameter values
	d_values = [0.4, 2.0, 10.0]

	axes_prior = [axCMatrixD1_prior, axCMatrixD2_prior, axCMatrixD3_prior]
	axes_seq1 = [axCMatrixD1_seq1, axCMatrixD2_seq1, axCMatrixD3_seq1]
	axes_seq2 = [axCMatrixD1_seq2, axCMatrixD2_seq2, axCMatrixD3_seq2]
	axes_SCR = [axCMatrixD1_SCR, axCMatrixD2_SCR, axCMatrixD3_SCR]

	# generate raster plots
	par = {}
	par['Astim'] = 1.0
	par['fCR'] = 10.0 # Hz
	par['npb'] = 1
	par['s'] = 0.01
	par['pathSimData'] = pathSimData

	par['type'] = "initial"
	par['d'] = 0.4
	genRasterPlot( axRaster_prior , par )

	for kd in range( len(d_values) ):
		par['d']  = d_values[kd]

		par['type'] = "initial"
		# genRasterPlot( axRaster_prior , par )
		mw = genPlotCMatrix( axes_prior[kd] , par, 5000 )
		axes_prior[kd].text(-2.4,1.8,r"$"+str( np.round(mw,2) )+"$", fontsize = tickFontsize )
		axes_prior[kd].set_aspect(1)
		
		par['seq'] = "0_1_2_3"
		par['type'] = "non-shuffled"
		if par['d'] == 0.4:
			genRasterPlot( axRaster_seq1 , par )
		mw = genPlotCMatrix( axes_seq1[kd] , par, 6000 )
		axes_seq1[kd].text(-2.4,1.8,r"$"+str( np.round(mw,2) )+"$", fontsize = tickFontsize )
		axes_seq1[kd].set_aspect(1)

		par['seq'] = "0_3_1_2"
		par['type'] = "non-shuffled"
		if par['d'] == 0.4:
			genRasterPlot( axRaster_seq2 , par )
		mw = genPlotCMatrix( axes_seq2[kd] , par, 6000 )
		axes_seq2[kd].text(-1.4,1.8,r"$"+str( np.round(mw,2) )+"$", fontsize = tickFontsize )
		axes_seq2[kd].set_aspect(1)
		
		par['type'] = "shuffled"
		par['seedSeq'] = 100
		if par['d'] == 0.4:
			genRasterPlot( axRaster_CRRVS , par )
		mw = genPlotCMatrix( axes_SCR[kd] , par, 6000 )
		axes_SCR[kd].text(-2.4,1.8,r"$"+str( np.round(mw,2) )+"$", fontsize = tickFontsize )
		axes_SCR[kd].set_aspect(1)

		# set axes raster
		for ax in [axRaster_prior, axRaster_seq1, axRaster_seq2, axRaster_CRRVS]:
			# ax.set_yticks([0,1000])
			ax.set_yticks([-2.5,2.5])
			ax.set_yticklabels(["$0$","$L$"], fontsize = tickFontsize)
	 
			ax.set_xticks([])
			ax.set_xticklabels([], fontsize = tickFontsize)
	 
			ax.set_frame_on(False)
			ax.get_yaxis().tick_left()
	  
			# ax.set_ylim(-105,1100)
			ax.set_ylim(-2.8,2.6)
	 
	 
		for ax in [axRaster_prior]:
			
			#ax.plot([4999.8,4999.9],[-90,-90], lw=3, color="black")
			ax.plot([4999.8,4999.9],[-2.65,-2.65], lw=3, color="black")
			ax.set_xlim(4999,5000)
		   
	#     for ax in [axRaster_seq1, axRaster_seq2, axRaster_CRRVS]:
	#         ax.plot([5989.8,5989.9],[-90,-90], lw=3, color="black")
	#         ax.set_xlim(5989,5990)
		for ax in [axRaster_seq1, axRaster_seq2, axRaster_CRRVS]:
			ax.plot([5999.8,5999.9],[-2.55,-2.55], lw=3, color="black")
			ax.plot([5999.8,5999.9],[-2.55,-2.55], lw=3, color="black")
			ax.set_xlim(5999,6000)
			
		
		#### specify layout for panels shown cMatrix
		# set axes cMatrix
		for ax in [axCMatrixD1_prior, axCMatrixD2_prior, axCMatrixD3_prior,axCMatrixD1_seq1, axCMatrixD2_seq1, axCMatrixD3_seq1,axCMatrixD1_seq2, axCMatrixD2_seq2, axCMatrixD3_seq2,axCMatrixD1_SCR, axCMatrixD2_SCR, axCMatrixD3_SCR]:
	 
			# ax.set_xticks([0,1000])
			ax.set_xticks([-2.5,2.5])
			ax.set_xticklabels(["",""], fontsize = tickFontsize)

			# ax.set_yticks([0,1000])
			ax.set_yticks([-2.5,2.5])
			ax.set_yticklabels(["",""], fontsize = tickFontsize)

		for ax in [axCMatrixD1_prior, axCMatrixD1_seq1, axCMatrixD1_seq2, axCMatrixD1_SCR]:
			ax.set_yticklabels(["$0$","$L$"], fontsize = tickFontsize)
			ax.set_ylabel("$x_{\mathrm{post}}$", fontsize = labelFontsize, labelpad=-5)
			# ax.set_yticklabels(["$0$","$N$"], fontsize = tickFontsize)
		
		for ax in [axCMatrixD1_SCR, axCMatrixD2_SCR, axCMatrixD3_SCR]:
			ax.set_xticklabels(["$0$","$L$"], fontsize = tickFontsize)
			ax.set_xlabel("$x_{\mathrm{pre}}$", fontsize = labelFontsize, labelpad=-5)
			# ax.set_xticklabels(["$0$","$N$"], fontsize = tickFontsize)

			   
	axCMatrixD1_prior.set_title( "$0.08 L (0.32 d)$", fontsize = tickFontsize )
	axCMatrixD2_prior.set_title( "$0.4 L (1.6 d)$", fontsize = tickFontsize )
	axCMatrixD3_prior.set_title( "$2.0 L (8 d)$", fontsize = tickFontsize )
		
	# panel labels
	# axRaster_prior.text(4998.8,1100,"A", fontsize = 1.2*labelFontsize)
	# axRaster_seq1.text(5988.8,1100,"E", fontsize = 1.2*labelFontsize)
	# axRaster_seq2.text(5988.8,1100,"I", fontsize = 1.2*labelFontsize)
	# axRaster_CRRVS.text(5988.8,1100,"M", fontsize = 1.2*labelFontsize)
	
	# axRaster_prior.text(4998.8,1100,"A", fontsize = 1.2*labelFontsize)
	# axRaster_seq1.text(5998.8,1100,"E", fontsize = 1.2*labelFontsize)
	# axRaster_seq2.text(5998.8,1100,"I", fontsize = 1.2*labelFontsize)
	# axRaster_CRRVS.text(5998.8,1100,"M", fontsize = 1.2*labelFontsize)

	axRaster_prior.text(4998.8,2.6,"A", fontsize = 1.2*labelFontsize)
	axRaster_seq1.text(5998.8,2.6,"E", fontsize = 1.2*labelFontsize)
	axRaster_seq2.text(5998.8,2.6,"I", fontsize = 1.2*labelFontsize)
	axRaster_CRRVS.text(5998.8,2.6,"M", fontsize = 1.2*labelFontsize)

	axCMatrixD1_prior.text(-4.1,2.8,"B", fontsize = 1.2*labelFontsize)
	axCMatrixD2_prior.text(-3.3,2.8,"C", fontsize = 1.2*labelFontsize)
	axCMatrixD3_prior.text(-3.3,2.8,"D", fontsize = 1.2*labelFontsize)
	  
	axCMatrixD1_seq1.text(-4.1,2.7,"F", fontsize = 1.2*labelFontsize)
	axCMatrixD2_seq1.text(-3.3,2.7,"G", fontsize = 1.2*labelFontsize)
	axCMatrixD3_seq1.text(-3.3,2.7,"H", fontsize = 1.2*labelFontsize)
	 
	axCMatrixD1_seq2.text(-4.1,2.5,"J", fontsize = 1.2*labelFontsize)
	axCMatrixD2_seq2.text(-3.3,2.5,"K", fontsize = 1.2*labelFontsize)
	axCMatrixD3_seq2.text(-3.3,2.5,"L", fontsize = 1.2*labelFontsize)

	axCMatrixD1_SCR.text(-4.2,2.5,"N", fontsize = 1.2*labelFontsize)
	axCMatrixD2_SCR.text(-3.4,2.5,"O", fontsize = 1.2*labelFontsize)
	axCMatrixD3_SCR.text(-3.3,2.5,"P", fontsize = 1.2*labelFontsize)

	return fig



def getEstSynapticMeanWeightBetweenPopulations( pathToData, Teval, M, sites, dsites ):
	
	mwFilename = pathToData + "meanWeightTimeSeries_"+str(Teval)+"_sec.npy"
	if os.path.isfile( mwFilename ):
		mw = np.load( mwFilename )
	# else:
	# 	print("ERROR: ",mwFilename, "not found")

	adjFilename = pathToData + "FinalBackup/synConnections.npz"
	if os.path.isfile( adjFilename ):
		adj  = scipy.sparse.load_npz( adjFilename )
	# else:
	# 	print("ERROR: ",adjFilename, "not found")

	STNCenterFilename = pathToData + "FinalBackup/STNCenter.npy" 
	if os.path.isfile( STNCenterFilename ):
		STNCenter = np.load( pathToData + "FinalBackup/STNCenter.npy" )
	# else:
	# 	print("ERROR: ",STNCenterFilename, "not found")

	# get indices of neurons that are the closest to each stimulation site
	# list that contains np.arrays of neuron indices that are the closest to respective sites
	indNeuronsSites = []
	# contains five lists, each containing only the index of the first and last neuron in 
	# the population with corresponding closest stimulation site
	limitsPopSites = []
	# run through sites and fill "indNeuronsSites"
	for ksite in range( M ):
		indNeuronsSites.append([])
		siteLocation  = sites[ ksite ]
		# get indices of all neurons that are the closest to this site
		indNeuronsSites[ksite] = np.arange(1000)[ np.abs( STNCenter-siteLocation )<dsites/2.0 ]
		# print("site", ksite)
		# print(indNeuronsSites[ksite])	
		limitsPopSites.append([indNeuronsSites[ksite][0],indNeuronsSites[ksite][-1]])

	# get mean number of synapses for different pre and postsynaptic subpopulations
	meanNSyn_all  = np.mean( adj[:1000,:1000] )
	meanNSyn_1to2 = np.mean( adj[limitsPopSites[2][0]:limitsPopSites[2][1],limitsPopSites[1][0]:limitsPopSites[1][1]] )
	meanNSyn_2to1 = np.mean( adj[limitsPopSites[1][0]:limitsPopSites[1][1],limitsPopSites[2][0]:limitsPopSites[2][1]] )
	meanNSyn_1to1 = np.mean( adj[limitsPopSites[1][0]:limitsPopSites[1][1],limitsPopSites[1][0]:limitsPopSites[1][1]] )

	mwTeval_all = mw[-1,1]/meanNSyn_all
	mwTeval_1to2 = mw[-1,2]/meanNSyn_1to2
	mwTeval_2to1 = mw[-1,3]/meanNSyn_2to1
	mwTeval_1to1 = mw[-1,4]/meanNSyn_1to1
	
	return mwTeval_all, mwTeval_1to2, mwTeval_2to1, mwTeval_1to1

def getEstSynapticMeanWeightBetweenPopulations_referenceCase( pathToData, Teval, M, sites, dsites ):
	
	cMatrix = scipy.sparse.load_npz( pathToData  + str(Teval)+"_sec/cMatrix.npz" )
	adj  = scipy.sparse.load_npz( pathToData + "FinalBackup/synConnections.npz" )
	STNCenter = np.load( pathToData + "FinalBackup/STNCenter.npy" )

	# get indices of neurons that are the closest to each stimulation site
	# list that contains np.arrays of neuron indices that are the closest to respective sites
	indNeuronsSites = []
	# contains five lists, each containing only the index of the first and last neuron in 
	# the population with corresponding closest stimulation site
	limitsPopSites = []
	# run through sites and fill "indNeuronsSites"
	for ksite in range( M ):
		indNeuronsSites.append([])
		siteLocation  = sites[ ksite ]
		# get indices of all neurons that are the closest to this site
		indNeuronsSites[ksite] = np.arange(1000)[ np.abs( STNCenter-siteLocation )<dsites/2.0 ]
		# print("site", ksite)
		# print(indNeuronsSites[ksite])	
		limitsPopSites.append([indNeuronsSites[ksite][0],indNeuronsSites[ksite][-1]])

	# get mean number of synapses for different pre and postsynaptic subpopulations
	mwTeval_all  = np.mean(cMatrix[:1000,:1000])/np.mean( adj[:1000,:1000] )
	mwTeval_1to2 = np.mean(cMatrix[limitsPopSites[2][0]:limitsPopSites[2][1],limitsPopSites[1][0]:limitsPopSites[1][1]])/np.mean( adj[limitsPopSites[2][0]:limitsPopSites[2][1],limitsPopSites[1][0]:limitsPopSites[1][1]] )
	mwTeval_2to1 = np.mean(cMatrix[limitsPopSites[1][0]:limitsPopSites[1][1],limitsPopSites[2][0]:limitsPopSites[2][1]])/np.mean( adj[limitsPopSites[1][0]:limitsPopSites[1][1],limitsPopSites[2][0]:limitsPopSites[2][1]] )
	mwTeval_1to1 = np.mean(cMatrix[limitsPopSites[1][0]:limitsPopSites[1][1],limitsPopSites[1][0]:limitsPopSites[1][1]])/np.mean( adj[limitsPopSites[1][0]:limitsPopSites[1][1],limitsPopSites[1][0]:limitsPopSites[1][1]] )
	
	return mwTeval_all, mwTeval_1to2, mwTeval_2to1, mwTeval_1to1

def getChangeOfMeanWeight( pathToSimResults_getJ, pathToInitialNetwork, networkSeed, phi, fCR, d, npb):

	M = 4
	dsiteInUnitsOfL = 0.25
	L = 5.0 # mm

	dsites = float(dsiteInUnitsOfL)*L # mm
	# locations of stimulation sites (centers of stimulus profile)
	sites = [-1.5*dsites, -0.5*dsites, 0.5*dsites, 1.5*dsites ] 

	# evaluation time
	T1 = 5020 # sec
	pathToData = pathToSimResults_getJ + "/seed_"+str(networkSeed)+"_d_"+str(d)+"_mw_init_0.45/get_J_phi_f_phi_"+str(phi)+"_fCR_"+str(fCR)+"_M_"+str(M)+"_fintra_130.0_npb_"+str(npb)+"_Astim_1.0_Tstim_100.0_dsites_"+str(dsiteInUnitsOfL)+"_erw_0.25/"
	mwT1_all, mwT1_1to2, mwT1_2to1, mwT1_1to1 = getEstSynapticMeanWeightBetweenPopulations( pathToData, T1, M, sites, dsites )
	# print( getEstSynapticMeanWeightBetweenPopulations( pathToData, T1 ) )

	# reference time
	T0 = 5000 # sec
	pathToData = pathToInitialNetwork + "/seed_"+str(networkSeed)+"_d_"+str(d)+"_mw_init_0.45/"
	mwT0_all, mwT0_1to2, mwT0_2to1, mwT0_1to1 = getEstSynapticMeanWeightBetweenPopulations_referenceCase( pathToData, T0, M, sites, dsites )
	# print( getEstSynapticMeanWeightBetweenPopulations_referenceCase( pathToData, T0 ) )

	# print(mwT1_1to2,mwT0_1to2)
	changeOfMeanWeight_1to2_T1mT0 = mwT1_1to2-mwT0_1to2

	return changeOfMeanWeight_1to2_T1mT0, mwT0_1to2

def load_estimated_weight_changes( pathToSimResults_getJ, outputToInitialNetworks ):
	# do for all parameter combinations
	L = 5.0 # mm
	seed_array = [10,12,14,16,18]
	d_values = np.array([0.4,2.0,10.0])
	fCR_values = np.round( np.arange(4.0,22.0, 1.0) , 1)
	phase_shifts = np.round( np.arange(0.0,1.0, 0.05) , 2)
	npb_values = [1,3]
	result_array = []

	for d in d_values:
		for fCR in fCR_values:
			# print(d,fCR)
			for phi in phase_shifts:
				for npb in npb_values:
					print(d,fCR,phi,npb)
					# average over network seeds
					current_DeltaWList = []
					current_WT0List = []
					for networkseed in seed_array:
						try:
							Deltaw, mwT0 = getChangeOfMeanWeight( pathToSimResults_getJ, outputToInitialNetworks, networkseed, phi, fCR, d, npb)
							if len(current_DeltaWList) == 0:
								current_DeltaWList = [Deltaw]
								current_WT0List = [mwT0]
							else:
								current_DeltaWList.append(Deltaw)
								current_WT0List.append(mwT0)
						except:
							continue
						#     print("Warning: some data were not found.")

					if len(current_DeltaWList) != 0:
						current_results = [d,fCR,phi,npb,np.mean(current_DeltaWList), np.std(current_DeltaWList), len(current_DeltaWList), np.mean(current_WT0List), np.std(current_WT0List), len(current_WT0List)]
						result_array.append(current_results)
						print(current_results)

	return np.array(result_array)

# the following function generates Figure 8
def gen_figure_estimate_J( backupFolder, pathToSimResults_getJ ):
	
	# load result array
	result_array = np.load( backupFolder + "/temp_result_array_Nov20.npy" )
	
	T1= 5020 # sec
	T0= 5000 # sec

	ticksFontsize = 12
	labelFontsize = 15

	import matplotlib.pyplot as plt 
	import matplotlib.gridspec as gridspec

	gsRaster_npb1 = gridspec.GridSpec(1, 1)
	gsRaster_npb1.update( left=0.05, right=0.2, top=0.95, bottom=0.65 )

	gsRaster_npb3 = gridspec.GridSpec(1, 1)
	gsRaster_npb3.update( left=0.25, right=0.4, top=0.95, bottom=0.65 )

	gsJ_npb1 = gridspec.GridSpec(1, 1)
	gsJ_npb1.update( left=0.05, right=0.23, top=0.6, bottom=0.05 )

	gsJ_npb3 = gridspec.GridSpec(1, 1)
	gsJ_npb3.update( left=0.25, right=0.43, top=0.6, bottom=0.05 )

	fig = plt.figure( figsize = (15,4) )

	################## 1 pulse per stimulus 
	ax_Raster_npg1 = fig.add_subplot(gsRaster_npb1[0,0])

	# plot single-pulse stimuli example as a raster plot
	seed = 10
	phi = 0.3
	fCR = 10.0
	npb = 1

	backupFilename = backupFolder + "/data_raster_seed_"+str(seed)+"_phi_"+str(phi)+"_fCR_"+str(fCR)+"_npb_"+str(npb)+".npz"
	if os.path.isfile( backupFilename ):
		# load from backup
		dataRaster =  np.load( backupFilename )

		plotSpikeTrain = dataRaster["plotSpikeTrain"]
		neuronCenters = dataRaster["neuronCenters"]

	else:
		# load from simulation data
		example_1_directory = pathToSimResults_getJ + "/seed_"+str(seed)+"_d_2.0_mw_init_0.45/get_J_phi_f_phi_"+str(phi)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_1.0_Tstim_100.0_dsites_0.25_erw_0.25/"
		spk = np.load( example_1_directory+"spikeTimes_5020_sec.npy")
		neuronCenters = np.load( example_1_directory+"FinalBackup/STNCenter.npy")
		Tmin = 5019.5
		Tmax = 5020
		plotSpikeTrain = spk[ np.logical_and( 0.0001*spk[:,1]>Tmin , 0.0001*spk[:,1]<Tmax ) ]
		plotSpikeTrain = plotSpikeTrain[ plotSpikeTrain[:,0]<1000 ]

		np.savez( backupFilename, plotSpikeTrain=plotSpikeTrain, neuronCenters=neuronCenters  )

	ax_Raster_npg1.scatter( 0.0001*plotSpikeTrain[:,1] , neuronCenters[plotSpikeTrain[:,0]], s=0.02, color="black" )

	# plot single-pulse stimuli change of mean weight
	ax_J_npg1 = fig.add_subplot(gsJ_npb1[0,0])

	npb_value = 1
	plot_data = result_array[ result_array[:,3] == npb_value ]
	cax = ax_J_npg1.scatter( plot_data[:,1], plot_data[:,2], c=plot_data[:,4], vmin=-0.4, vmax=0.6, cmap="jet" )
	cbar = plt.colorbar(cax, ticks=[-0.4,-0.2,0,0.2,0.4,0.6])
	cbar.ax.set_yticklabels(["$-0.4$","","$0$","","$0.4$",""], fontsize = ticksFontsize)

	# print( "1", np.min(plot_data[:,4]), np.max(plot_data[:,4]) )


	################## 3 pulses per stimulus 
	ax_Raster_npg3 = fig.add_subplot(gsRaster_npb3[0,0])

	# plot burst stimuli example as a raster plot
	seed = 10
	phi = 0.3
	fCR = 10.0
	npb = 3

	backupFilename = backupFolder + "/data_raster_seed_"+str(seed)+"_phi_"+str(phi)+"_fCR_"+str(fCR)+"_npb_"+str(npb)+".npz"
	if os.path.isfile( backupFilename ):
		# load from backup
		dataRaster =  np.load( backupFilename )

		plotSpikeTrain = dataRaster["plotSpikeTrain"]
		neuronCenters = dataRaster["neuronCenters"]

	else:
		# load from simulation data
		example_2_directory = pathToSimResults_getJ + "/seed_"+str(seed)+"_d_2.0_mw_init_0.45/get_J_phi_f_phi_"+str(phi)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_1.0_Tstim_100.0_dsites_0.25_erw_0.25/"
		spk = np.load( example_2_directory+"spikeTimes_5020_sec.npy")
		neuronCenters = np.load( example_2_directory+"FinalBackup/STNCenter.npy")
		Tmin = 5019.5
		Tmax = 5020
		plotSpikeTrain = spk[ np.logical_and( 0.0001*spk[:,1]>Tmin , 0.0001*spk[:,1]<Tmax ) ]
		plotSpikeTrain = plotSpikeTrain[ plotSpikeTrain[:,0]<1000 ]

		np.savez( backupFilename, plotSpikeTrain=plotSpikeTrain, neuronCenters=neuronCenters  )

	ax_Raster_npg3.scatter( 0.0001*plotSpikeTrain[:,1] , neuronCenters[ plotSpikeTrain[:,0] ], s=0.02, color="black" )

	# plot burst stimuli change of mean weight
	ax_J_npg3 = fig.add_subplot(gsJ_npb3[0,0])

	npb_value = 3
	plot_data = result_array[ result_array[:,3] == npb_value ]
	cax = ax_J_npg3.scatter( plot_data[:,1], plot_data[:,2], c=plot_data[:,4], vmin=-0.4, vmax=0.6, cmap="jet" )
	cbar = plt.colorbar(cax, ticks=[-0.4,-0.2,0,0.2,0.4,0.6])
	cbar.ax.set_yticklabels(["$-0.4$","","$0$","","$0.4$",""], fontsize = ticksFontsize)
	cbar.set_label(r"$\Delta w$", fontsize = labelFontsize, labelpad = 0)

	# print(plot_data)
	# print( "3", np.min(plot_data[:,4]), np.max(plot_data[:,4]) )

	##################### specify layout
	# layout of raster plots
	yticks_Raster = [-2.5,-1.25,0,1.25,2.5]
	for ax in [ax_Raster_npg1, ax_Raster_npg3]:
		ax.axhline(-1.25, color = "red", ls="--", lw=0.6)
		ax.axhline(0.0, color = "red", ls="--", lw=0.6)
		ax.axhline(1.25, color = "red", ls="--", lw=0.6)

		ax.plot([5019.85,5019.95],[-2.7,-2.7], lw=3, color="black")

		ax.set_xticks([])
		ax.set_xticklabels([], fontsize = ticksFontsize)

		ax.set_yticks(yticks_Raster)
		ax.set_yticklabels(["" for x in yticks_Raster])

		ax.set_frame_on(False)
		ax.get_yaxis().tick_left()

		ax.set_ylim(-2.8,2.5)

	ax_Raster_npg1.set_yticklabels([r"$0$","","","",r"$L$"], fontsize = ticksFontsize)

	# layout of plots showing change of mean weight
	xticksChangeOfMeanWeight = [4,8,12,16,20]
	yticksChangeOfMeanWeight = [0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1]
	for ax in [ax_J_npg1,ax_J_npg3]:

		ax.set_xticks(xticksChangeOfMeanWeight)
		ax.set_xticklabels(["$"+str(x)+"$" for x in xticksChangeOfMeanWeight], fontsize = ticksFontsize)

		ax.set_yticks(yticksChangeOfMeanWeight)
		ax.set_yticklabels(["" for x in yticksChangeOfMeanWeight])

		ax.set_ylim(0,1)
		ax.set_xlim(4,21)

		ax.set_xlabel("$f$ (Hz)", fontsize = labelFontsize)

	ax_J_npg1.set_yticklabels(["$0$","",r"$1/4$","",r"$1/2$","",r"$3/4$","","$1$"], fontsize = ticksFontsize)

	ax_J_npg1.set_ylabel(r"$\phi_{x \rightarrow y}$", fontsize = labelFontsize)

	return fig


def get_mw_from_simulations(npb, fCR, d, seq, Teval, directory):
	# get simulation data    
	if seq == "I_II_III_IV":
		seqStr = "0_1_2_3"
	elif seq == "I_II_IV_III":
		seqStr = "0_1_3_2"
	elif seq == "I_III_II_IV":
		seqStr = "0_3_1_3"
	elif seq == "I_III_IV_II":
		seqStr = "0_2_3_1"
	elif seq == "I_IV_II_III":
		seqStr = "0_3_1_2"
	elif seq == "I_IV_III_II":
		seqStr = "0_3_2_1"
		
	# directory = "/Users/jkromer/Desktop/Projects/Stanford/scratch/lif-networks/Frontiers_SSTPMDBN/CR_stimulation_long"
	path_cMatrix = directory + "/seed_10_d_"+str(d)+"_mw_init_0.45/CR_stim_seq_"+str(seqStr)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_1.0_Tstim_7220.0_dsites_0.25_erw_0.25/"+str(Teval)+"_sec/cMatrix.npz"
	path_adj = directory    +  "/seed_10_d_"+str(d)+"_mw_init_0.45/CR_stim_seq_"+str(seqStr)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_1.0_Tstim_7220.0_dsites_0.25_erw_0.25/"+str(Teval)+"_sec/synConnections.npz"

	path_centerCoordinates = directory    +  "/seed_10_d_"+str(d)+"_mw_init_0.45/CR_stim_seq_"+str(seqStr)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_1.0_Tstim_7220.0_dsites_0.25_erw_0.25/"+str(Teval)+"_sec/STNCenter.npy"
	
	if os.path.isfile( path_cMatrix ):
		cMatrix = scipy.sparse.load_npz( path_cMatrix )[:1000,:1000]
		if os.path.isfile( path_adj ):
			adj     = scipy.sparse.load_npz( path_adj )[:1000,:1000]
			if os.path.isfile( path_centerCoordinates ):
				centerCoordinates = np.load( path_centerCoordinates )[:1000]
	else:
		print("ERROR: file not found")
		return np.nan
	
	# distinguish between inter- and intra-population connections
	dsiteInUnitsOfL = 0.25
	L = 5.0 # mm
	M=4
	dsites = float(dsiteInUnitsOfL)*L # mm
	# locations of stimulation sites (centers of stimulus profile)
	sites = [-1.5*dsites, -0.5*dsites, 0.5*dsites, 1.5*dsites ] 

	# get indices of neurons that are the closest to each stimulation site
	# list that contains np.arrays of neuron indices that are the closest to respective sites
	indNeuronsSites = []
	# contains five lists, each containing only the index of the first and last neuron in 
	# the population with corresponding closest stimulation site
	limitsPopSites = []
	# run through sites and fill "indNeuronsSites"
	for ksite in range( M ):
		indNeuronsSites.append([])
		siteLocation  = sites[ ksite ]
		# get indices of all neurons that are the closest to this site
		indNeuronsSites[ksite] = np.arange(1000)[ np.abs( centerCoordinates-siteLocation )<dsites/2.0 ]
		# print("site", ksite)
		# print(indNeuronsSites[ksite])	
		limitsPopSites.append([indNeuronsSites[ksite][0],indNeuronsSites[ksite][-1]])

	# get mean weights of intra-population and inter-population synapses
	total_weights_intra = 0
	total_number_of_synapses_intra = 0
	total_weights_inter = 0
	total_number_of_synapses_inter = 0
	
	for kpopPre in range(4):
		for kpopPost in range(4):
			# print( kpopPre, kpopPost )
			if kpopPre == kpopPost:
				# print( limitsPopSites[kpopPost][0], limitsPopSites[kpopPost][1], limitsPopSites[kpopPre][0], limitsPopSites[kpopPre][1] )
				total_weights_intra            += np.sum( cMatrix[ limitsPopSites[kpopPost][0]:limitsPopSites[kpopPost][1], limitsPopSites[kpopPre][0]:limitsPopSites[kpopPre][1] ] )
				total_number_of_synapses_intra +=     np.sum( adj[ limitsPopSites[kpopPost][0]:limitsPopSites[kpopPost][1], limitsPopSites[kpopPre][0]:limitsPopSites[kpopPre][1] ] )
			if kpopPre != kpopPost:
				# print( limitsPopSites[kpopPost][0], limitsPopSites[kpopPost][1], limitsPopSites[kpopPre][0], limitsPopSites[kpopPre][1] )
				total_weights_inter            += np.sum( cMatrix[ limitsPopSites[kpopPost][0]:limitsPopSites[kpopPost][1], limitsPopSites[kpopPre][0]:limitsPopSites[kpopPre][1] ] )
				total_number_of_synapses_inter +=    np.sum( adj[ limitsPopSites[kpopPost][0]:limitsPopSites[kpopPost][1], limitsPopSites[kpopPre][0]:limitsPopSites[kpopPre][1] ] )

		
	return  np.sum(cMatrix)/np.sum(adj), total_weights_intra/float(total_number_of_synapses_intra), total_weights_inter/float(total_number_of_synapses_inter)

def loadSimulation_results_intra_inter(directory):
	resultsSim = []
	Teval = 8000 # sec
	dic_resultsSim = {}

	for d in [0.4,2.0,10.0]:
		dic_resultsSim[d] = {}
		for npb in [1,3]:
			dic_resultsSim[d][npb] = {}
			for seq in ["I_II_III_IV","I_II_IV_III","I_III_II_IV","I_III_IV_II","I_IV_II_III","I_IV_III_II"] :
				fMwData = []
				for fCR in np.arange(4.0,21.1,1.0): # Hz
					try:
						mwAll, mwIntra, mwInter = get_mw_from_simulations(npb, fCR, d,seq,Teval, directory)
						# print( mwAll, mwIntra, mwInter )
						fMwData.append([fCR, mwAll, mwIntra, mwInter])
					except:
						continue
				dic_resultsSim[d][npb][seq] = np.array( fMwData )

	return dic_resultsSim

def get_mw_from_simulations_shuffledCR(npb, fCR, d, seedSeq, Teval, directory):
	# get simulation data    
	Tshuffle = np.round( 1.0/fCR , 4) # seconds

	# directory = "/Users/jkromer/Desktop/Projects/Stanford/scratch/lif-networks/Frontiers_SSTPMDBN/CR_stimulation_long"
	path_cMatrix = directory    +  "/seed_10_d_"+str(d)+"_mw_init_0.45/shuffled_CR_stim_Tshuffle_"+str(Tshuffle)+"_seedSeq_"+str(seedSeq)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_1.0_Tstim_7220.0_seedSeq_"+str(seedSeq)+"_dsites_0.25_erw_0.25/"+str(Teval)+"_sec/cMatrix.npz"
	path_adj = directory    +  "/seed_10_d_"+str(d)+"_mw_init_0.45/shuffled_CR_stim_Tshuffle_"+str(Tshuffle)+"_seedSeq_"+str(seedSeq)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_1.0_Tstim_7220.0_seedSeq_"+str(seedSeq)+"_dsites_0.25_erw_0.25/"+str(Teval)+"_sec/synConnections.npz"
	path_centerCoordinates = directory    +  "/seed_10_d_"+str(d)+"_mw_init_0.45/shuffled_CR_stim_Tshuffle_"+str(Tshuffle)+"_seedSeq_"+str(seedSeq)+"_fCR_"+str(fCR)+"_M_4_fintra_130.0_npb_"+str(npb)+"_Astim_1.0_Tstim_7220.0_seedSeq_"+str(seedSeq)+"_dsites_0.25_erw_0.25/"+str(Teval)+"_sec/STNCenter.npy"
	
	if os.path.isfile( path_cMatrix ):
		cMatrix = scipy.sparse.load_npz( path_cMatrix )[:1000,:1000]
		if os.path.isfile( path_adj ):
			adj     = scipy.sparse.load_npz( path_adj )[:1000,:1000]
			if os.path.isfile( path_centerCoordinates ):
				centerCoordinates = np.load( path_centerCoordinates )[:1000]
	else:
		print("ERROR: file not found")
		return np.nan
	
	# distinguish between inter- and intra-population connections
	dsiteInUnitsOfL = 0.25
	L = 5.0 # mm
	M=4
	dsites = float(dsiteInUnitsOfL)*L # mm
	# locations of stimulation sites (centers of stimulus profile)
	sites = [-1.5*dsites, -0.5*dsites, 0.5*dsites, 1.5*dsites ] 

	# get indices of neurons that are the closest to each stimulation site
	# list that contains np.arrays of neuron indices that are the closest to respective sites
	indNeuronsSites = []
	# contains five lists, each containing only the index of the first and last neuron in 
	# the population with corresponding closest stimulation site
	limitsPopSites = []
	# run through sites and fill "indNeuronsSites"
	for ksite in range( M ):
		indNeuronsSites.append([])
		siteLocation  = sites[ ksite ]
		# get indices of all neurons that are the closest to this site
		indNeuronsSites[ksite] = np.arange(1000)[ np.abs( centerCoordinates-siteLocation )<dsites/2.0 ]
		# print("site", ksite)
		# print(indNeuronsSites[ksite])	
		limitsPopSites.append([indNeuronsSites[ksite][0],indNeuronsSites[ksite][-1]])

	# get mean weights of intra-population and inter-population synapses
	total_weights_intra = 0
	total_number_of_synapses_intra = 0
	total_weights_inter = 0
	total_number_of_synapses_inter = 0
	
	for kpopPre in range(4):
		for kpopPost in range(4):
			# print( kpopPre, kpopPost )
			if kpopPre == kpopPost:
				# print( limitsPopSites[kpopPost][0], limitsPopSites[kpopPost][1], limitsPopSites[kpopPre][0], limitsPopSites[kpopPre][1] )
				total_weights_intra            += np.sum( cMatrix[ limitsPopSites[kpopPost][0]:limitsPopSites[kpopPost][1], limitsPopSites[kpopPre][0]:limitsPopSites[kpopPre][1] ] )
				total_number_of_synapses_intra +=     np.sum( adj[ limitsPopSites[kpopPost][0]:limitsPopSites[kpopPost][1], limitsPopSites[kpopPre][0]:limitsPopSites[kpopPre][1] ] )
			if kpopPre != kpopPost:
				# print( limitsPopSites[kpopPost][0], limitsPopSites[kpopPost][1], limitsPopSites[kpopPre][0], limitsPopSites[kpopPre][1] )
				total_weights_inter            += np.sum( cMatrix[ limitsPopSites[kpopPost][0]:limitsPopSites[kpopPost][1], limitsPopSites[kpopPre][0]:limitsPopSites[kpopPre][1] ] )
				total_number_of_synapses_inter +=    np.sum( adj[ limitsPopSites[kpopPost][0]:limitsPopSites[kpopPost][1], limitsPopSites[kpopPre][0]:limitsPopSites[kpopPre][1] ] )

		
	return  np.sum(cMatrix)/np.sum(adj), total_weights_intra/float(total_number_of_synapses_intra), total_weights_inter/float(total_number_of_synapses_inter)

def loadSimulation_results_intra_inter_shuffledCR(directory):
	resultsSim = []
	Teval = 8000 # sec
	dic_resultsSim = {}

	for d in [0.4,2.0,10.0]:
		dic_resultsSim[d] = {}
		for npb in [1,3]:
			dic_resultsSim[d][npb] = {}
			for seedSeq in [100] :
				fMwData = []
				for fCR in np.arange(4.0,21.1,1.0): # Hz
				#try:
					mwAll, mwIntra, mwInter = get_mw_from_simulations_shuffledCR(npb, fCR, d,seedSeq,Teval, directory)
					# print( mwAll, mwIntra, mwInter )
					fMwData.append([fCR, mwAll, mwIntra, mwInter])
				#except:
					#continue
				dic_resultsSim[d][npb][seedSeq] = np.array( fMwData )

	return dic_resultsSim

# the following function generates Figue 9
def genFigure_meanweight_theory_vs_sim( dic_plt_results, dic_resultsSim, dic_resultsSimShuffled, pathSimData ):


	import matplotlib.pyplot as plt 
	import matplotlib.gridspec as gridspec

	ticksFontsize = 12
	labelFontsize = 15

	fig = plt.figure( figsize = (10,6))

	seq_array = ["I_II_III_IV","I_II_IV_III","I_III_II_IV","I_III_IV_II","I_IV_II_III","I_IV_III_II"] 
	color_seq = [ 'magenta','slateblue','orange','mediumspringgreen','cyan','dodgerblue']
	labels_seq = ["I,II,III,IV","I,II,IV,III","I,III,II,IV","I,III,IV,II","I,IV,II,III","I,IV,III,II"]
	# color_seq = ["0.6","0.6","0.6","0.6","0.6","0.6","0.6"]
	linestyles = ["-","-","-","-.","-.","-."]
	lw = 1
	color_CRRVS = "0.3"
	# marker size shuffled CR
	msShuffed = 25

	# seed for shuffled CR sequence 
	seedSeq = 100

	##### single pulses, d = 0.08 L (=0.4 mm)
	npb=1
	d=0.4 # mm
	ax_single_pulses_d1_intra = fig.add_subplot(431)
	ax_single_pulses_d1_inter = fig.add_subplot(434)

	plt_results = dic_plt_results[npb][d]
	#print(dic_plt_results)
	for kseq in range( len(seq_array) ):
	    try:
	        plot_data_seq =  plt_results[ plt_results[:,1]==kseq ]
	        # intra-population
	        ax_single_pulses_d1_intra.plot( plot_data_seq[:,0], plot_data_seq[:,3], color=color_seq[kseq], ls=linestyles[kseq], lw=lw,label = labels_seq[kseq])
	        # inter-population
	        ax_single_pulses_d1_inter.plot( plot_data_seq[:,0], plot_data_seq[:,4], color=color_seq[kseq], ls=linestyles[kseq], lw=lw,label = labels_seq[kseq])

	        # plot simulation results
	        ax_single_pulses_d1_intra.scatter( dic_resultsSim[d][npb][seq_array[kseq]][:,0], dic_resultsSim[d][npb][seq_array[kseq]][:,2], color=color_seq[kseq], marker="x", clip_on = False)
	        ax_single_pulses_d1_inter.scatter( dic_resultsSim[d][npb][seq_array[kseq]][:,0], dic_resultsSim[d][npb][seq_array[kseq]][:,3], color=color_seq[kseq], marker="x", clip_on = False)
	    
	        dic_resultsSimShuffled
	    except:
	        continue
	        
	# results for shuffled CR
	ax_single_pulses_d1_intra.scatter( dic_resultsSimShuffled[d][npb][seedSeq][:,0], dic_resultsSimShuffled[d][npb][seedSeq][:,2], color=color_CRRVS, marker="d",s=msShuffed, label="shuffled", clip_on = False)
	ax_single_pulses_d1_inter.scatter( dic_resultsSimShuffled[d][npb][seedSeq][:,0], dic_resultsSimShuffled[d][npb][seedSeq][:,3], color=color_CRRVS, marker="d",s=msShuffed, label="shuffled", clip_on = False)

	##### bursts, d = 0.4 L (=2.0 mm)
	npb=3
	d=0.4 # mm
	ax_bursts_d1_intra = fig.add_subplot(4,3,7)
	ax_bursts_d1_inter = fig.add_subplot(4,3,10)
	plt_results = dic_plt_results[npb][d]

	for kseq in range( len(seq_array) ):
	    try:
	        plot_data_seq =  plt_results[ plt_results[:,1]==kseq ]
	        # intra-population
	        ax_bursts_d1_intra.plot( plot_data_seq[:,0], plot_data_seq[:,3], color=color_seq[kseq], ls=linestyles[kseq], lw=lw,label = labels_seq[kseq])
	        # inter-population
	        ax_bursts_d1_inter.plot( plot_data_seq[:,0], plot_data_seq[:,4], color=color_seq[kseq], ls=linestyles[kseq], lw=lw,label = labels_seq[kseq])

	        # plot simulation results
	        ax_bursts_d1_intra.scatter( dic_resultsSim[d][npb][seq_array[kseq]][:,0], dic_resultsSim[d][npb][seq_array[kseq]][:,2], color=color_seq[kseq], marker="x", clip_on = False)
	        ax_bursts_d1_inter.scatter( dic_resultsSim[d][npb][seq_array[kseq]][:,0], dic_resultsSim[d][npb][seq_array[kseq]][:,3], color=color_seq[kseq], marker="x", clip_on = False)
	    except:
	        continue
	        
	# results for shuffled CR
	ax_bursts_d1_intra.scatter( dic_resultsSimShuffled[d][npb][seedSeq][:,0], dic_resultsSimShuffled[d][npb][seedSeq][:,2], color=color_CRRVS, marker="d",s=msShuffed, clip_on = False)
	ax_bursts_d1_inter.scatter( dic_resultsSimShuffled[d][npb][seedSeq][:,0], dic_resultsSimShuffled[d][npb][seedSeq][:,3], color=color_CRRVS, marker="d",s=msShuffed, clip_on = False)

	##### single pulses, d = 0.4 L (=2.0 mm)
	npb=1
	d=2.0 # mm
	ax_single_pulses_d2_intra = fig.add_subplot(432)
	ax_single_pulses_d2_inter = fig.add_subplot(435)

	plt_results = dic_plt_results[npb][d]

	for kseq in range( len(seq_array) ):
	    try:
	        plot_data_seq =  plt_results[ plt_results[:,1]==kseq ]
	        # intra-population
	        ax_single_pulses_d2_intra.plot( plot_data_seq[:,0], plot_data_seq[:,3], color=color_seq[kseq], ls=linestyles[kseq], lw=lw,label = labels_seq[kseq])
	        # inter-population
	        ax_single_pulses_d2_inter.plot( plot_data_seq[:,0], plot_data_seq[:,4], color=color_seq[kseq], ls=linestyles[kseq], lw=lw,label = labels_seq[kseq])

	        # plot simulation results
	        ax_single_pulses_d2_intra.scatter( dic_resultsSim[d][npb][seq_array[kseq]][:,0], dic_resultsSim[d][npb][seq_array[kseq]][:,2], color=color_seq[kseq], marker="x", clip_on = False)
	        ax_single_pulses_d2_inter.scatter( dic_resultsSim[d][npb][seq_array[kseq]][:,0], dic_resultsSim[d][npb][seq_array[kseq]][:,3], color=color_seq[kseq], marker="x", clip_on = False)
	    except:
	        continue  
	        
	# results for shuffled CR
	ax_single_pulses_d2_intra.scatter( dic_resultsSimShuffled[d][npb][seedSeq][:,0], dic_resultsSimShuffled[d][npb][seedSeq][:,2], color=color_CRRVS, marker="d",s=msShuffed, clip_on = False)
	ax_single_pulses_d2_inter.scatter( dic_resultsSimShuffled[d][npb][seedSeq][:,0], dic_resultsSimShuffled[d][npb][seedSeq][:,3], color=color_CRRVS, marker="d",s=msShuffed, clip_on = False)

	##### bursts, d = 0.4 L (=2.0 mm)
	npb=3
	d=2.0 # mm
	ax_bursts_d2_intra = fig.add_subplot(4,3,8)
	ax_bursts_d2_inter = fig.add_subplot(4,3,11)
	plt_results = dic_plt_results[npb][d] 

	for kseq in range( len(seq_array) ):
	    try:
	        plot_data_seq =  plt_results[ plt_results[:,1]==kseq ]
	        # intra-population
	        ax_bursts_d2_intra.plot( plot_data_seq[:,0], plot_data_seq[:,3], color=color_seq[kseq], ls=linestyles[kseq], lw=lw,label = labels_seq[kseq])
	        # inter-population
	        ax_bursts_d2_inter.plot( plot_data_seq[:,0], plot_data_seq[:,4], color=color_seq[kseq], ls=linestyles[kseq], lw=lw,label = labels_seq[kseq])

	        # plot simulation results
	        ax_bursts_d2_intra.scatter( dic_resultsSim[d][npb][seq_array[kseq]][:,0], dic_resultsSim[d][npb][seq_array[kseq]][:,2], color=color_seq[kseq], marker="x", clip_on = False)
	        ax_bursts_d2_inter.scatter( dic_resultsSim[d][npb][seq_array[kseq]][:,0], dic_resultsSim[d][npb][seq_array[kseq]][:,3], color=color_seq[kseq], marker="x", clip_on = False)
	    except:
	        continue  
	        
	# results for shuffled CR
	ax_bursts_d2_intra.scatter( dic_resultsSimShuffled[d][npb][seedSeq][:,0], dic_resultsSimShuffled[d][npb][seedSeq][:,2], color=color_CRRVS, marker="d",s=msShuffed, clip_on = False)
	ax_bursts_d2_inter.scatter( dic_resultsSimShuffled[d][npb][seedSeq][:,0], dic_resultsSimShuffled[d][npb][seedSeq][:,3], color=color_CRRVS, marker="d",s=msShuffed, clip_on = False)

	##### single pulses, d = 2 L (=10.0 mm)
	npb=1
	d=10.0 # mm
	ax_single_pulses_d3_intra = fig.add_subplot(4,3,3)
	ax_single_pulses_d3_inter = fig.add_subplot(4,3,6)

	plt_results = dic_plt_results[npb][d]

	for kseq in range( len(seq_array) ):
	    try:
	        plot_data_seq =  plt_results[ plt_results[:,1]==kseq ]
	        # intra-population
	        ax_single_pulses_d3_intra.plot( plot_data_seq[:,0], plot_data_seq[:,3], color=color_seq[kseq], ls=linestyles[kseq], lw=lw,label = labels_seq[kseq])
	        # inter-population
	        ax_single_pulses_d3_inter.plot( plot_data_seq[:,0], plot_data_seq[:,4], color=color_seq[kseq], ls=linestyles[kseq], lw=lw,label = labels_seq[kseq])

	        # plot simulation results
	        ax_single_pulses_d3_intra.scatter( dic_resultsSim[d][npb][seq_array[kseq]][:,0], dic_resultsSim[d][npb][seq_array[kseq]][:,2], color=color_seq[kseq], marker="x", clip_on = False)
	        ax_single_pulses_d3_inter.scatter( dic_resultsSim[d][npb][seq_array[kseq]][:,0], dic_resultsSim[d][npb][seq_array[kseq]][:,3], color=color_seq[kseq], marker="x", clip_on = False)
	    except:
	        continue   
	        
	# results for shuffled CR
	ax_single_pulses_d3_intra.scatter( dic_resultsSimShuffled[d][npb][seedSeq][:,0], dic_resultsSimShuffled[d][npb][seedSeq][:,2], color=color_CRRVS, marker="d",s=msShuffed, clip_on = False)
	ax_single_pulses_d3_inter.scatter( dic_resultsSimShuffled[d][npb][seedSeq][:,0], dic_resultsSimShuffled[d][npb][seedSeq][:,3], color=color_CRRVS, marker="d",s=msShuffed, clip_on = False)

	##### bursts, d = 2 L (=10.0 mm)
	npb=3
	d=10.0 # mm
	ax_bursts_d3_intra = fig.add_subplot(4,3,9)
	ax_bursts_d3_inter = fig.add_subplot(4,3,12)

	plt_results = dic_plt_results[npb][d] 

	for kseq in range( len(seq_array) ):
	    try:
	        plot_data_seq =  plt_results[ plt_results[:,1]==kseq ]
	        # intra-population
	        ax_bursts_d3_intra.plot( plot_data_seq[:,0], plot_data_seq[:,3], color=color_seq[kseq], ls=linestyles[kseq], lw=lw,label = labels_seq[kseq])
	        # inter-population
	        ax_bursts_d3_inter.plot( plot_data_seq[:,0], plot_data_seq[:,4], color=color_seq[kseq], ls=linestyles[kseq], lw=lw,label = labels_seq[kseq])

	        # plot simulation results
	        ax_bursts_d3_intra.scatter( dic_resultsSim[d][npb][seq_array[kseq]][:,0], dic_resultsSim[d][npb][seq_array[kseq]][:,2], color=color_seq[kseq], marker="x", clip_on = False)
	        ax_bursts_d3_inter.scatter( dic_resultsSim[d][npb][seq_array[kseq]][:,0], dic_resultsSim[d][npb][seq_array[kseq]][:,3], color=color_seq[kseq], marker="x", clip_on = False)
	    except:
	        continue    
	        
	# results for shuffled CR
	ax_bursts_d3_intra.scatter( dic_resultsSimShuffled[d][npb][seedSeq][:,0], dic_resultsSimShuffled[d][npb][seedSeq][:,2], color=color_CRRVS, marker="d", s=msShuffed,clip_on = False)
	ax_bursts_d3_inter.scatter( dic_resultsSimShuffled[d][npb][seedSeq][:,0], dic_resultsSimShuffled[d][npb][seedSeq][:,3], color=color_CRRVS, marker="d", s=msShuffed, clip_on = False)


	# plot horizontal line for initial mean synaptic weight
	for ax in [ax_single_pulses_d1_intra,ax_single_pulses_d1_inter,ax_bursts_d1_intra,ax_bursts_d1_inter]:
	    d=0.4
	    directory = pathSimData + "/initial_networks/seed_10_d_"+str(d)+"_mw_init_0.45"
	    path_cMatrix = directory + "/5000_sec/cMatrix.npz"
	    path_adj = directory    +  "/5000_sec/synConnections.npz"
	    cMatrix = scipy.sparse.load_npz( path_cMatrix )[:1000,:1000]
	    adj     = scipy.sparse.load_npz( path_adj )[:1000,:1000]
	    ax.axhline( np.sum(cMatrix)/np.sum(adj) , color = "black", ls=":")
	    # print(d,np.sum(cMatrix)/np.sum(adj))
	    
	for ax in [ax_single_pulses_d2_intra,ax_single_pulses_d2_inter,ax_bursts_d2_intra,ax_bursts_d2_inter]:
	    d=2.0
	    directory = pathSimData + "/initial_networks/seed_10_d_"+str(d)+"_mw_init_0.45"
	    path_cMatrix = directory + "/5000_sec/cMatrix.npz"
	    path_adj = directory    +  "/5000_sec/synConnections.npz"
	    cMatrix = scipy.sparse.load_npz( path_cMatrix )[:1000,:1000]
	    adj     = scipy.sparse.load_npz( path_adj )[:1000,:1000]
	    ax.axhline( np.sum(cMatrix)/np.sum(adj) , color = "black", ls=":")
	    # print(d,np.sum(cMatrix)/np.sum(adj))
	    
	for ax in [ax_single_pulses_d3_intra,ax_single_pulses_d3_inter,ax_bursts_d3_intra,ax_bursts_d3_inter]:
	    d=10.0
	    directory = pathSimData + "/initial_networks/seed_10_d_"+str(d)+"_mw_init_0.45"
	    path_cMatrix = directory + "/5000_sec/cMatrix.npz"
	    path_adj = directory    +  "/5000_sec/synConnections.npz"
	    cMatrix = scipy.sparse.load_npz( path_cMatrix )[:1000,:1000]
	    adj     = scipy.sparse.load_npz( path_adj )[:1000,:1000]
	    ax.axhline( np.sum(cMatrix)/np.sum(adj) , color = "black", ls=":")
	    # print(d,np.sum(cMatrix)/np.sum(adj))
	    
	xticksChangeOfMeanWeight = [4,8,12,16,20]
	yticks=[0,0.2,0.4,0.6,0.8,1.0]
	for ax in [ax_single_pulses_d1_intra,ax_single_pulses_d1_inter,ax_single_pulses_d2_intra,ax_single_pulses_d2_inter,ax_single_pulses_d3_intra,ax_single_pulses_d3_inter,ax_bursts_d1_intra,ax_bursts_d1_inter,ax_bursts_d2_intra,ax_bursts_d2_inter,ax_bursts_d3_intra,ax_bursts_d3_inter]:
	    
	    ax.set_yticks(yticks)
	    ax.set_yticklabels(["" for x in yticks])
	    ax.set_ylim(-0.05,1.05)
	    
	    ax.set_xticks(xticksChangeOfMeanWeight)
	    ax.set_xticklabels(["" for x in xticksChangeOfMeanWeight], fontsize = ticksFontsize)
	    ax.set_xlim(4,21)
	   
	for ax in [ax_bursts_d1_inter,ax_bursts_d2_inter,ax_bursts_d3_inter]:
	    ax.set_xticklabels(["$"+str(x)+"$" for x in xticksChangeOfMeanWeight], fontsize = ticksFontsize)
	    ax.set_xlabel("$f_{\mathrm{CR}}$ (Hz)", fontsize = labelFontsize)
	    
	# for ax in [ax_single_pulses_d1,ax_bursts_d1]:
	#     ax.text(5,0.85,"$d=0.08$L", fontsize = ticksFontsize)
	# for ax in [ax_single_pulses_d2,ax_bursts_d2]:
	#     ax.text(5,0.85,"$d=0.4$L", fontsize = ticksFontsize)
	# for ax in [ax_single_pulses_d3,ax_bursts_d3]:
	#     ax.text(5,0.85,"$d=2$L", fontsize = ticksFontsize)
	 
	for ax in [ax_single_pulses_d1_intra,ax_bursts_d1_intra]:
	    ax.set_yticklabels(["$0$","","","","","$1$"], fontsize = ticksFontsize)
	    ax.set_ylabel(r"$\langle w \rangle_{\mathrm{intra}}$", fontsize = labelFontsize)
	for ax in [ax_single_pulses_d1_inter,ax_bursts_d1_inter]:
	    ax.set_yticklabels(["$0$","","","","","$1$"], fontsize = ticksFontsize)
	    ax.set_ylabel(r"$\langle w \rangle_{\mathrm{inter}}$", fontsize = labelFontsize)
	         
	ax_single_pulses_d1_intra.legend(frameon=False, ncol=2, fontsize = 0.7*ticksFontsize)

	ax_single_pulses_d1_intra.set_title("$s=0.08L$ $(0.32d)$", fontsize = labelFontsize)
	ax_single_pulses_d2_intra.set_title("$s=0.4L$ $(1.6d)$", fontsize = labelFontsize)
	ax_single_pulses_d3_intra.set_title("$s=2L$ $(8d)$", fontsize = labelFontsize)

	ax_single_pulses_d1_intra.text(-5,-0.65,"single pulse", fontsize = labelFontsize, rotation=90)
	ax_bursts_d1_intra.text(-5,-0.45,"burst", fontsize = labelFontsize, rotation=90)


	ax_single_pulses_d1_intra.text(0.5,1,"A", fontsize = 1.1*labelFontsize)
	ax_single_pulses_d1_inter.text(0.5,1,"A'", fontsize = 1.1*labelFontsize)
	ax_single_pulses_d2_intra.text(2,1,"B", fontsize = 1.1*labelFontsize)
	ax_single_pulses_d2_inter.text(2,1,"B'", fontsize = 1.1*labelFontsize)
	ax_single_pulses_d3_intra.text(2,1,"C", fontsize = 1.1*labelFontsize)
	ax_single_pulses_d3_inter.text(2,1,"C'", fontsize = 1.1*labelFontsize)

	ax_bursts_d1_intra.text(0.5,1,"E", fontsize = 1.1*labelFontsize)
	ax_bursts_d1_inter.text(0.5,1,"E'", fontsize = 1.1*labelFontsize)
	ax_bursts_d2_intra.text(2,1,"F", fontsize = 1.1*labelFontsize)
	ax_bursts_d2_inter.text(2,1,"F'", fontsize = 1.1*labelFontsize)
	ax_bursts_d3_intra.text(2,1,"G", fontsize = 1.1*labelFontsize)
	ax_bursts_d3_inter.text(2,1,"G'", fontsize = 1.1*labelFontsize)

	return fig 
	# fig.savefig( "Fig9.pdf" , bbox_inches="tight" )
	# fig.savefig( "Fig9.png" , bbox_inches="tight" )
	# fig.savefig( "Fig9.svg" , bbox_inches="tight" )



if __name__ == "__main__":

	### The following code was generated to do the evaluation on the computation cluster.
	
	# The code below loads the trajetories of the mean synaptic weight and calculates the Kuramoto order parameter 
	# for initial network simulations.
	if sys.argv[1] == "load_trajectories_Kuramoto_and_weights_initial":
		directory = '/scratch/users/jkromer/lif-networks/Frontiers_SSTPMDBN/initial_networks'
		outputDirectory = "/scratch/users/jkromer/lif-networks/Frontiers_SSTPMDBN/initial_networks/data_Kuramoto_weights"
		
		for seed in [10]: #, 12, 14,16,18]:
			for d in [0.4, 2.0, 10.0]:
				load_weight_data( directory, seed , d, outputDirectory )

	
	# The code below loads the trajetories of the mean synaptic weight and calculates the Kuramoto order parameter 
	# for simulations of non-shuffled CR.
	if sys.argv[1] == "load_trajectories_Kuramoto_and_weights_nonShuffledCR":
		
		Astim = 1.0
		npb = 3
		
		## outputToInitialNetworks ... This is the directory in which simulation results will be stored. This directory is generated if it doesn't already exist. 
		outputToInitialNetworks        = '/scratch/users/jkromer/lif-networks/Frontiers_SSTPMDBN/initial_networks'

		## outputPath_nonShuffledCR ... This is the directory in which simulation results will be stored. This directory is generated if it doesn't already exist. 
		outputPath_nonShuffledCR        = '/scratch/users/jkromer/lif-networks/Frontiers_SSTPMDBN/CR_stimulation'

		## outputPath_relAfterNonShuffledCR ... directory in which simulation results will be stored (is generated if it doesn't already exist) 
		outputPath_relAfterNonShuffledCR        = '/scratch/users/jkromer/lif-networks/Frontiers_SSTPMDBN/relaxation_after_CR_stimulation'
		
		pathToBackupOfKuramotoMeanW_nonShuffledCR = "/home/users/jkromer/Project1/Frontiers_SSTPMDBN/gitRepositories/Frontiers_SSTPMDBN/data/data_trajectories_mean_weight_kuramoto_Nov13"
		
		load_trajectories_Kuramoto_and_weights_nonShuffledCR( outputToInitialNetworks , outputPath_nonShuffledCR , outputPath_relAfterNonShuffledCR, pathToBackupOfKuramotoMeanW_nonShuffledCR, Astim, npb )

	# The code below loads the trajetories of the mean synaptic weight and calculates the Kuramoto order parameter 
	# for simulations of shuffled CR.
	if sys.argv[1] == "load_trajectories_Kuramoto_and_weights_shuffledCR":
		
		Astim = 1.0
		npb = 3
		
		## outputToInitialNetworks ... This is the directory in which simulation results will be stored. This directory is generated if it doesn't already exist. 
		outputToInitialNetworks        = '/scratch/users/jkromer/lif-networks/Frontiers_SSTPMDBN/initial_networks'

		## outputPath_shuffledCR ... This is the directory in which simulation results will be stored. This directory is generated if it doesn't already exist. 
		outputPath_shuffledCR        = '/scratch/users/jkromer/lif-networks/Frontiers_SSTPMDBN/shuffled_CR_stimulation'

		## outputPath_relAfterShuffledCR ... directory in which simulation results will be stored (is generated if it doesn't already exist) 
		outputPath_relAfterShuffledCR        = '/scratch/users/jkromer/lif-networks/Frontiers_SSTPMDBN/relaxation_after_shuffled_CR_stimulation'
		
		pathToBackupOfKuramotoMeanW_shuffledCR = "/home/users/jkromer/Project1/Frontiers_SSTPMDBN/gitRepositories/Frontiers_SSTPMDBN/data/data_trajectories_mean_weight_kuramoto_Nov13"
		
		load_trajectories_Kuramoto_and_weights_shuffledCR( outputToInitialNetworks , outputPath_shuffledCR , outputPath_relAfterShuffledCR, pathToBackupOfKuramotoMeanW_shuffledCR, Astim, npb )

	# The following code was used to load data from simulations on two-site stimulation to 
	# estimated the mean synaptic weight change.
	if sys.argv[1] == "load_estimated_weight_changes":

		pathSimData = "/scratch/users/jkromer/lif-networks/Frontiers_SSTPMDBN"

		## outputPath ... directory in which simulation results will be stored (is generated if it doesn't already exist) 
		pathToSimResults_getJ   = pathSimData + '/get_J_phi_f'

		inputToInitialNetworks = pathSimData+'/initial_networks'

		result_array = load_estimated_weight_changes( pathToSimResults_getJ, inputToInitialNetworks )

		print("result_array",result_array)

		# specify backup folder
		backupFolder = "../data/data_backup_fig_get_J"

		# Next, save results. NOTE: if file name is changed, it also needs to be changed in functions_analyze_data.gen_figure_estimate_J
		# np.save( backupFolder + "/temp_result_array_Nov18.npy", result_array )
		np.save( backupFolder + "/temp_result_array_Dec4.npy", result_array )



