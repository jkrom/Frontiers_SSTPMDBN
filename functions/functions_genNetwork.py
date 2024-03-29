##################################
# written by: Justus Kromer
##################################
# written and tested for Python 2.7.13
##################################
#
#	content:
#
#		contains all scripts for generating connectivity matrices
#
#		class ellipsoid() 	l 34
#							def __init__(self, float , float, float )
#							def isInVolumen(self, Points)
#
#		def placeNeurons( int_1d , int_1d ) 	
#
# 		def conProbability( float_1d, float )    	
#
#		def cartesianProduct( element_1d , element_1d )	 
#
#		def synReversalsCmatrix( float_3d, float_3d, int_1d, int_1d, float, float, float, float)	
#
#		def generate_connectivity_and_weight_matrix_Ebert( system_parameters , rnd_state_for_network_generation )
#
#		def placeNeurons_3D_cuboid( NSTN , NGPe, rx_STN, ry_STN, rz_STN, rx_GPe, ry_GPe, rz_GPe )
#
# 		def synReversalsCmatrix_3D_cuboid(positionsSTNNeurons, positionsGPeNeurons, NSTN, NGPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN, r_STN, r_GPe)
#
#		def generate_connectivity_and_weight_matrix_3D_cuboid( system_parameters , rnd_state_for_network_generation )
#
#	if implemented run 'python functions_genNetwork.py CLASS_OR_FUNCTIONNAME' to test function or class with 'CLASS_OR_FUNCTIONNAME'

# imports
from scipy.interpolate import interp1d
import numpy as np 
import itertools
import scipy




##################################
# function: conProbability
def conProbability(d, Cd):
	#
	#       Returns a the probability for neurons to connect at a certain distance d.
	#		Considers exponential decay as in 
	#		Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe	
	#
	#       input: d, Cd
	#           d ... 1d array of distances (same units as Cd)
	#			Cd ... scale for exponential decay (same units as d)
	#       return: STNNeuronPositons, GPeNeuronPositons	
	# 			p ... 1d array of connection probability for distances d
	# connection probability decays exponentially
	p=1/Cd*np.exp(-d/Cd )
	
	# return probabilities
	return p


##################################
# function: conProbability
#
#	Returns the cartesian product of a and b (list of all possible combinations of elements of 'a' an 'b')
#
#       input: a, b
#			a ... list of elements
#			b ... list of elements
#		return:  
#			list of all possible combinations of elements of a and b
def cartesianProduct(a,b):
	return np.array([x for x in itertools.product(a, b)])

#################################
# function: placeNeurons_3D_cuboid
def placeNeurons_1D( NSTN , NGPe, x_STN_min, x_STN_max, x_GPe_min, x_GPe_max ):
	#
	#       Places NSTN neurons in cuboid associated with subthalamic nucleus (STN) and 
	#		NGPe in cuboid associated with globus pallidus externus (GPe).
	#	    axes are aligned with coordinate system.	
	#
	#       input: NSTN, NGPe
	#           NSTN ... number of STN neurons that need to be placed
	#			NGPe ... number of GPe neurons that need to be placed
	#			rx ... max distance from center in x-direction
	#			ry ... max distance from center in y-direction
	#			rz ... max distance from center in z-direction
	#       return: STNNeuronPositons, GPeNeuronPositons	
	# 			STNNeuronPositons ... numpy array of STN neuron centers in 3d (mm) 
	# 			GPeNeuronPositons ... numpy array of GPe neuron centers in 3d (mm) 

	# 1) place STN neurons
	# (i) start with uniformly distributed positions in 1d .. 
	STNNeuronPositons=np.random.uniform( x_STN_min, x_STN_max, NSTN )
	
	# 2) place GPe neurons
	# (i) start with uniformly distributed positions in 1d .. 
	GPeNeuronPositons=np.random.uniform( x_GPe_min, x_GPe_max, NGPe )

	# return lists of neuron positions in 3d that were placed in STN and GPe volume, respectively 
	return STNNeuronPositons, GPeNeuronPositons

##################################
# function: synReversalsCmatrix
#   Returns a 2d matrix of integers indicating which neurons are connected by an excitatory synapse (entry 1),
#   which neurons are connected by inhibitory synapses (entry -1) or which neurons are not connected (entry = 0).
#   
#   The shape of that matrix is a block matrix with dimension (NSTN+NGPe, NSTN+NGPe)
#
#   Recurrent STN and GPe connections are implemented according to a 
#   distance-dependent connection probability with exponential shape taken from Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe  
#
#   Probability for internetwork connections does not depend on the distance, as in Ebert et al. 2014
#   
#   Distance-dependent connection are randomly implemented by calculating the probability for each possible connections and then
#   drawing the desired number of connections without replacement from the pool of all possible connections.
#
#   periodic boundary conditions are not applied
#
#       input:   positionsSTNNeurons, positionsGPeNeurons, NSTN, NGPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN
#           positionsSTNNeurons ... 1d np.array of STN neuron positions in 3d in units of mm
#           positionsGPeNeurons ... 1d np.array of GPe neuron positions in 3d in units of mm
#           NSTN ... total number of STN neurons
#           NGPe ... total number of GPe neurons
#           P_STN_STN ... Probability for STN -> STN connections (total number of connections is P_STN_STN * ( NSTN * NSTN ) )
#           P_STN_GPe ... Probability for STN -> GPe connections (total number of connections is P_STN_GPe * ( NSTN * NGPe ) )
#           P_GPe_GPe ... Probability for GPe -> GPe connections (total number of connections is P_GPe_GPe * ( NGPe * NGPe ) )
#           P_GPe_STN ... Probability for GPe -> STN connections (total number of connections is P_GPe_STN * ( NGPe * BSTN ) )
#           r_STN ... list containing max x,y,z distances from center for STN volume
#           r_GPe ... list containing max x,y,z distances from center for GPe volume
#           Cd_STN... characteristic distance for synaptic connections
#
#       return:
#           synReversals ... block matrix of integers and dimension (NSTN+NGPe, NSTN+NGPe). synReversals[i,j] contains information about the 
#                            connection between presynatpic neuron j and postsynatpic neuron i
#                            synReversals[i,j] = 1 -> exc. connections from j to i
#                            synReversals[i,j] = -1 -> inh. connections from j to i
#                            synReversals[i,j] = 0 -> no connections from j to i
def variable_distance_synReversalsCmatrix_1D(positionsSTNNeurons, positionsGPeNeurons, NSTN, NGPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN, Cd_STN ):

	# initiallize return matrix containing synReversals[i,j]
	# 1  ... presynaptic neuron j is connected to postsynapti neuron i by excitatory synapse
	# -1 ... presynaptic neuron j is connected to postsynapti neuron i  by inhibitory synapse
	# 0 ... no connections from j to i
	synReversals=np.zeros( (NSTN+NGPe, NSTN+NGPe) ) 

	# sort neurons according to x-coordinate
	positionsSTNNeurons = np.sort( positionsSTNNeurons )
	positionsGPeNeurons = np.sort( positionsGPeNeurons )

	##################################
	# CONNECTIONS FOR STN -> STN
	##################################
	# distance-dependent connection probability according to Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe 

	# total number of STN -> STN connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int( np.round( P_STN_STN * NSTN * NSTN ) )

	if totNumberOfConnection != 0:
		# implement array of all possible STN -> STN connections
		# first index for pre, second for post synaptic neuron
		allPossibleSTNtoSTNconnections=cartesianProduct( np.arange(NSTN),np.arange(NSTN) )

		# calculate distances related to these connections  in units of mm
		# apply periodic boundary conditions
		vectors_connecting_neurons = positionsSTNNeurons[allPossibleSTNtoSTNconnections[:,1]]-positionsSTNNeurons[allPossibleSTNtoSTNconnections[:,0]]

		# calculate distances related to these connections  in units of mm
		distances=np.abs( vectors_connecting_neurons )

		# probability densities to implement a connection with those lengths according to 
		# Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe
		# we consider the same value here. It is important that Cd_STN is significantly shorter than the dimension of the volume though
		#Cd_STN = 0.5 # characteristic scale in mm
		probs=conProbability( distances, Cd_STN ) # connection probability density in 1/mm

		# exclude self connections
		indizesOfNonSelfconnections=allPossibleSTNtoSTNconnections[:,0]!=allPossibleSTNtoSTNconnections[:,1]
		probs=probs[indizesOfNonSelfconnections]    # 1/mm
		allPossibleSTNtoSTNconnections=allPossibleSTNtoSTNconnections[indizesOfNonSelfconnections]

		# normalize probabiltiy to one
		probs=1/np.sum(probs)*probs   # probs contains the probability for each connection to be selected if only a single connection was implemented
		
		# implement synaptic connections according to probabilities
		STNSTNconnectionsIndizes=np.random.choice( len(allPossibleSTNtoSTNconnections) , totNumberOfConnection, p=probs, replace=False)
		
		# STNSTNconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
		STNSTNconnections=allPossibleSTNtoSTNconnections[ STNSTNconnectionsIndizes ]

		# add these connections to the return matrix 'synReversals'
		for connection in STNSTNconnections:
			# add excitatory connection
			# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
			synReversals[ connection[1],  connection[0] ]=1

	##################################  
	# CONNECTIONS FOR STN-> GPe
	##################################
	#  the connections probability for STN -> GPe connections does not depend on the distance

	# total number of STN-> GPe connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int(np.round(P_STN_GPe*NSTN*NGPe))

	if totNumberOfConnection != 0:
		# implement array of all possible STN -> GPe connections
		# first index for pre, second for post synaptic neuron      neuron indices for STN neurons are 0 - NSTN-1 and for GPe neurons NSTN - NSTN+NGPe-1
		allPossibleSTNtoGPeconnections=cartesianProduct( np.arange(NSTN),np.arange(NSTN, NSTN+NGPe ) )


		# all STN-> GPe are implemented with the same probability
		probs=np.full( len(allPossibleSTNtoGPeconnections), 1/float(totNumberOfConnection) )

		# normalize to probs one
		probs=1/np.sum(probs)*probs # probs contains the probability for each connection to be selected if only a single connection was implemented
		
		# implement synaptic connections according to probabilities
		STNGPeConnectionsIndizes=np.random.choice(len(allPossibleSTNtoGPeconnections), totNumberOfConnection, p=probs, replace=False)
		
		# STNGPeconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
		STNGPeconnections=allPossibleSTNtoGPeconnections[ STNGPeConnectionsIndizes ]

		# add these connections to the return matrix 'synReversals'
		for connection in STNGPeconnections:
			# add excitatory connection
			# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
			synReversals[ connection[1],  connection[0] ]=1



	##################################  
	# CONNECTIONS FOR GPe-> GPe
	##################################
	# distance-dependent connection probability according to Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe 

	# total number of GPe-> GPe connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int(np.round(P_GPe_GPe*NGPe*NGPe))

	if totNumberOfConnection != 0:
		# implement array of all possible GPe -> GPe connections
		# first index for pre, second for post synaptic neuron      neuron indices for STN neurons are 0 - NSTN-1 and for GPe neurons NSTN - NSTN+NGPe-1
		allPossibleGPetoGPeconnections=cartesianProduct( np.arange( NSTN, NSTN+NGPe ) ,np. arange( NSTN, NSTN+NGPe )  )


		# calculate distances related to these connection  in units of mm
		# apply periodic boundary conditions
		vectors_connecting_neurons = positionsGPeNeurons[allPossibleGPetoGPeconnections[:,1]-NSTN]-positionsGPeNeurons[allPossibleGPetoGPeconnections[:,0]-NSTN]

		# calculate distances related to these connection  in units of mm
		distances=np.abs( vectors_connecting_neurons )

		# probability densities to implement a connection with those lengths according to 
		# Ebert et al. 2014. p5 sec. 2.2. THREE-DIMENSIONAL MODEL OF THE STN AND THE GPe
		# we consider the same value here. It is important that Cd_GPe is significantly shorter than the dimension of the volume though 
		Cd_GPe = 0.63 # characteristic scale in mm
		probs=conProbability(distances, Cd_GPe ) # connection probability density in 1/mm

		# exclude self connections
		indizesOfNonSelfconnections=allPossibleGPetoGPeconnections[:,0]!=allPossibleGPetoGPeconnections[:,1]
		probs=probs[indizesOfNonSelfconnections]
		allPossibleGPetoGPeconnections=allPossibleGPetoGPeconnections[indizesOfNonSelfconnections]

		# normalize to probs one
		probs=1/np.sum(probs)*probs # probs contains the probability for each connection to be selected if only a single connection was implemented
		
		# implement synaptic connections according to probabilities
		GPeGPeconnectionsIndizes=np.random.choice(len(allPossibleGPetoGPeconnections), totNumberOfConnection, p=probs, replace=False)
		
		# GPeGPeconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
		GPeGPeconnections=allPossibleGPetoGPeconnections[ GPeGPeconnectionsIndizes ]

		# add these connections to the return matrix 'synReversals'
		for connection in GPeGPeconnections:
			# add inhibitory connection
			# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
			synReversals[ connection[1],  connection[0] ]=-1


	##################################  
	# CONNECTIONS FOR GPe-> STN
	##################################
	#  the connections probability for STN -> GPe connections does not depend on the distance

	# total number of GPe-> STN connections (round is actually not necessary, just to avoid non integer input )
	totNumberOfConnection=int(np.round(P_GPe_STN*NGPe*NSTN))

	if totNumberOfConnection != 0:
		# implement array of all possible GPe -> STN connections
		# first index for pre, second for post synaptic neuron      neuron indices for STN neurons are 0 - NSTN-1 and for GPe neurons NSTN - NSTN+NGPe-1
		allPossibleGPetoSTNconnections=cartesianProduct( np.arange(NSTN, NSTN+NGPe ) ,np.arange(NSTN )  )

		# all GPe -> STN are implemented with the same probability  
		probs=np.full( len(allPossibleGPetoSTNconnections), 1/float(totNumberOfConnection) )

		# normalize to probs one
		probs=1/np.sum(probs)*probs # probs contains the probability for each connection to be selected if only a single connection was implemented

		# GPeSTNconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
		GPeSTNconnectionsIndizes=np.random.choice(len(allPossibleGPetoSTNconnections), totNumberOfConnection, p=probs, replace=False)
		
		# GPeSTNconnections is the array of all connections that were selected. First entry is index of presynaptic neuron, second entry index of the postsynaptic neuron
		GPeSTNconnections=allPossibleGPetoSTNconnections[ GPeSTNconnectionsIndizes ]

		# add these connections to the return matrix 'synReversals'
		for connection in GPeSTNconnections:
			# add inhibitory connection
			# note that synReversals[i,j] is refers to connections from presynaptic neuron j to presynaptic neuron i
			synReversals[ connection[1],  connection[0] ]=-1
		
	# return result
	return synReversals



def sequence_paper_generate_connectivity_and_weight_matrix_1D( system_parameters , rnd_state_for_network_generation, d_synaptic_length_scale ):

	# load needed parameters from system_parameters
	# number of STN neurons
	N_STN = system_parameters['N_STN']
	# number of GPe neurons
	N_GPe = system_parameters['N_GPe']
	# total number of neurons
	N = N_STN+ N_GPe
	# probability for STN -> STN connection
	P_STN_STN = system_parameters['P_STN_STN']
	# probability for STN -> GPe connection
	P_STN_GPe = system_parameters['P_STN_GPe']
	# probability for GPe -> GPe connection
	P_GPe_GPe = system_parameters['P_GPe_GPe']
	# probability for GPe -> STN connection
	P_GPe_STN = system_parameters['P_GPe_STN']

	# synaptic transmission delay in time steps
	StepsTauSynDelaySTNSTN=int(system_parameters['tauSynDelaySTNSTN']/system_parameters['dt']) # time steps
	StepsTauSynDelayGPeGPe=int(system_parameters['tauSynDelayGPeGPe']/system_parameters['dt']) # time steps
	StepsTauSynDelayGPeSTN=int(system_parameters['tauSynDelayGPeSTN']/system_parameters['dt']) # time steps
	StepsTauSynDelaySTNGPe=int(system_parameters['tauSynDelaySTNGPe']/system_parameters['dt']) # time steps

	# max strengths exc coupling
	cMaxExc=system_parameters['cMaxExc']	
	# mean initial strengths exc coupling
	cExcInit=system_parameters['cExcInit']

	# max inh coupling
	cMaxInh=system_parameters['cMaxInh']
	# mean initial strengths inh coupling
	cInhInit=system_parameters['cInhInit']

	# set state of random number generator
	np.random.set_state( rnd_state_for_network_generation  )


	x_STN_min =system_parameters['x_STN_min']
	x_STN_max =system_parameters['x_STN_max']
	
	x_GPe_min =system_parameters['x_GPe_min']
	x_GPe_max =system_parameters['x_GPe_max']

	# get connectivity matrix
	STNCenter, GPeCenter= placeNeurons_1D( N_STN , N_GPe, x_STN_min, x_STN_max, x_GPe_min, x_GPe_max )

	# sort neurons according to x-coordinate
	STNCenter = np.sort( STNCenter )
	GPeCenter = np.sort( GPeCenter )

	synConnections= variable_distance_synReversalsCmatrix_1D(STNCenter, GPeCenter, N_STN, N_GPe, P_STN_STN, P_STN_GPe, P_GPe_GPe, P_GPe_STN, d_synaptic_length_scale )


	# set diagonal to zero 
	diaZero=np.ones( (N,N) )-np.diag( np.ones( N ) )
	synConnections=synConnections*diaZero

	# decouple GPe  ( this is done since we only used STN neurons in our simulations, uncomment if not needed)
	for kNeuron in range(N_STN,N):
		synConnections[:,kNeuron]=np.zeros(N)
		synConnections[kNeuron,:]=np.zeros(N)

	#########################################################################################
	#    in the following additional arrays are introduced to speed up simulations
	# get indicec post and presynaptic neurons to speed up STDP
	PostSynNeurons = {}
	PreSynNeurons = {}

	# max numbers of corresponding synapses
	maxNumberOfPostSynapticNeurons=0
	maxNumberOfPreSynapticNeurons=0

	for kNeuron in range(N_STN):

		PostSynNeurons[kNeuron]=np.nonzero( ( synConnections[:,kNeuron].astype(int) ).tolist() )[0].tolist()
		PreSynNeurons[kNeuron]=np.nonzero( ( synConnections[kNeuron,:].astype(int) ).tolist() )[0].tolist()

		# add random intra-network connections in case of no post/pre neurons
		# this is to help getting fully connected networks
		if len(PostSynNeurons[kNeuron])==0:

			# add random connection
			if kNeuron < N_STN:
				kPost=np.random.choice( range(kNeuron)+range(kNeuron+1,N_STN) )
				synConnections[kPost,kNeuron]=1


		if len(PreSynNeurons[kNeuron])==0:
			# add random connection
			# this is to help getting fully connected networks
			if kNeuron < N_STN:
				kPre=np.random.choice( range(kNeuron)+range(kNeuron+1,N_STN) )
				synConnections[kNeuron,kPre]=1

		# update max numbers of connections
		if maxNumberOfPostSynapticNeurons<len(PostSynNeurons[kNeuron]):
			maxNumberOfPostSynapticNeurons=len(PostSynNeurons[kNeuron])
		if maxNumberOfPreSynapticNeurons<len(PreSynNeurons[kNeuron]):
			maxNumberOfPreSynapticNeurons=len(PreSynNeurons[kNeuron])

	# generate numpy array with post synaptic neurons to speed up simulations ...
	numpyPostSynapticNeurons=np.full((N,maxNumberOfPostSynapticNeurons),N+1)
	numpyPreSynapticNeurons=np.full((N,maxNumberOfPreSynapticNeurons),N+1)

	# ... and corresponding matrix containing transimission delays in time steps
	transmissionDelaysPostSynNeurons=np.full((N,maxNumberOfPostSynapticNeurons),-1.0)
	transmissionDelaysPreSynNeurons=np.full((N,maxNumberOfPreSynapticNeurons),-1.0)

	# gen numpy array with post synaptic neurons
	for kPreSyn in range(N_STN):
		postSynNeuronskPre=PostSynNeurons[kPreSyn]
		for kPostSyn in range(len(postSynNeuronskPre)):
			numpyPostSynapticNeurons[kPreSyn, kPostSyn]=postSynNeuronskPre[kPostSyn]

			kPostNeuron=postSynNeuronskPre[kPostSyn]
			if (kPreSyn < N_STN):
				if (kPostNeuron < N_STN):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelaySTNSTN

				if (kPostNeuron >= N_STN) and (kPostNeuron < N):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelaySTNGPe


			if (kPreSyn >= N_STN) and (kPreSyn < N):
				if (kPostNeuron < N_STN):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelayGPeSTN

				if (kPostNeuron >= N_STN) and (kPostNeuron < N):
					transmissionDelaysPostSynNeurons[kPreSyn, kPostSyn]=StepsTauSynDelayGPeGPe

	# gen numpy array with post synaptic neurons
	for kPostSyn in range(N_STN):
		preSynNeuronskPre=PreSynNeurons[kPostSyn]
		for kPreSyn in range(len(preSynNeuronskPre)):
			numpyPreSynapticNeurons[kPostSyn, kPreSyn]=preSynNeuronskPre[kPreSyn]

			kPreNeuron=preSynNeuronskPre[kPreSyn]
			if (kPostSyn < N_STN):
				if (kPreNeuron < N_STN):
					transmissionDelaysPreSynNeurons[kPostSyn, kPreSyn]=StepsTauSynDelaySTNSTN

				if (kPreNeuron >= N_STN) and (kPreNeuron < N):
					transmissionDelaysPreSynNeurons[kPostSyn, kPreSyn]=StepsTauSynDelayGPeSTN

			if (kPostSyn >= N_STN) and (kPostSyn < N):
				if (kPreNeuron < N_STN):
					transmissionDelaysPreSynNeurons[kPostSyn, kPreSyn]=StepsTauSynDelaySTNGPe

				if (kPreNeuron >= N_STN) and (kPreNeuron < N):
					transmissionDelaysPreSynNeurons[kPostSyn, kPreSyn]=StepsTauSynDelayGPeGPe

	# synaptic weight matrix
	cMatrix=np.zeros( (N , N) )

	# initialize synaptic weights by setting random weights to zero so that the mean initial
	# synaptic weights are cExcInit/cMaxExc and cInhInit/cMaxInh for excitatory and inhibitory connections, 
	# respectively
	# mean inital weights
	if cMaxExc != 0:
		meanInitalExcWeight=cExcInit/cMaxExc
	else:
		meanInitalExcWeight=0

	if cMaxInh != 0:
		meanInitalInhWeight=cInhInit/cMaxInh
	else:
		meanInitalInhWeight=0

	# initialize excitatory connections
	P1e=meanInitalExcWeight
	P0e=1-P1e
	cMatrix[:,:N_STN]=np.random.choice([0.0,1.0],(N,N_STN),p=[P0e,P1e])

	# initialize inhibitory connections
	P1i=meanInitalInhWeight
	P0i=1-P1i
	cMatrix[:,N_STN:]=np.random.choice([0.0,1.0],(N,N_GPe),p=[P0i,P1i])

	# filter weits with actual connections according to connectivity matrix
	cMatrix=cMatrix*synConnections
	cMatrix=scipy.sparse.csc_matrix(cMatrix)
	csc_Zero=scipy.sparse.csc_matrix(np.zeros( ( N,N ) ))
	csc_Ones=scipy.sparse.csc_matrix(np.ones( ( N,N ) ))

	# output struct containing neuron positions in mm
	neuronLoc = { 'STN_center_mm' : STNCenter , 'GPe_center_mm' : GPeCenter }

	# output struct containing objects that are related to network structure but only needed during simulation
	sim_objects = { 'max_N_pre' : maxNumberOfPreSynapticNeurons ,'max_N_post' : maxNumberOfPostSynapticNeurons , 'numpyPostSynapticNeurons' : numpyPostSynapticNeurons , 'numpyPreSynapticNeurons' : numpyPreSynapticNeurons , 'td_PostSynNeurons' : transmissionDelaysPostSynNeurons , 'td_PreSynNeurons' : transmissionDelaysPreSynNeurons , 'csc_Zero' : csc_Zero , 'csc_Ones' : csc_Ones	}
	

	# return output
	return synConnections , cMatrix , neuronLoc , sim_objects




