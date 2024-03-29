import numpy as np 

## returns "wSignalArray" which specifies how strong the influence of 
## the stimuli delivered to the M stimulation sites is on individual neurons
#  
#  stimulus profiles for individual stimuli have the shape:
#
#  wSignal_ij = wkl/(1 + (Xi - Xsitej)^2/sigmaj^2)              (1)
#
#  Here, i is the index of the neuron and j is the index of the stimulation site.
#  sigmaj is the width the profile in mm. To get an equivalent rectangular width of "A" choose sigmaj="A/pi" . 
#
#  ouput:
#  "wSignalArray" has dimensions (N, nElectrodes) where N is total number of STN and GPe neurons and
#                 nElectrodes it the number of stimulation sites
#
#  input:
#  "sites"        numpy array of lengths "nElectrodes" containing the center coordinates of all stimulation sites
#  "positions"    Dictionary in which each key refers to a numpy array of the X coordinates of neurons
#                 of a certain type. The total number of neurons "N" is obtained by summing the lengths of position arrays
#                 for all keys in this array.
#  "stim_par"     dictionary contains all stimulus parameters
#                 specifically
#                 stim_par[ "sigmaj" ] ... Numpy array of lengths "number of neuron types" (the number of keys in "positions").
#                                          Entry j is the width of stimuli delivered the jth stimulation site in mm.
#                 stim_par["receive_stim"] . array of floats of shape ( nElectrodes , number of neuron types). If entry [k,l] is wkl then the 
#                                            stimuli delivered to the kth stimulation site affect neuron type l with a relative strengths of wkl (see Equation 1). 
#                                            
def getWarray( sites, positions, stim_par ):
	
	# total number which is calculated based on "positions" (see below).
	N = 0
	# List of length "total number of neurons" containing integers specifying the neuron 
	# type according to its position in "list(positions.keys())"
	neuronTypes = []
	NumberOfNeuronTypes = len( positions.keys() )
	# List of length "total number of neurons" containing X postions of individual neurons.
	NeuronPositions = []

	for kneurontype in range( NumberOfNeuronTypes ):
		neurontype = list(positions.keys())[kneurontype]
		
		# calculation of the total number of neurons
		N+=len(positions[neurontype])
		# add neuron positions for this "neurontype" to "NeuronPositions"
		NeuronPositions.extend( positions[neurontype] )
		# add "neurontype" to "neuronTypes"
		neuronTypes.extend( kneurontype*np.ones( len(positions[neurontype]) ).astype(int) )
		
	# get number of stimulation sites
	nElectrodes = len(sites)

	# initialize "wSignalArray"
	wSignalArray = np.zeros( (N, nElectrodes) )
	
	# run through all stimulation sites
	for j in range( nElectrodes ):
		
		Xj = sites[j] # mm
		sigmaj = stim_par[ "sigmaj" ][j] # mm

		for i in range( N ):
			
			Xi = NeuronPositions[ i ] # mm

			wSignalArray[i,j] = stim_par["receive_stim"][ j, neuronTypes[i] ]/(1.0 + np.power( (Xi-Xj)/sigmaj , 2 ) )
			
	return wSignalArray


