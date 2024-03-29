##################################
# written by: Justus Kromer
##################################
# written and tested for Python 2.7.13
##################################

# imports
import sys
import numpy as np
import itertools 




##################################
# non-shuffled CR
##################################
class fixed_Sequence_bursts_overlapping:

	## input:           
	#   timeBetweenBursts      ... time between end of burst and beginning of next burst
	#   totalStimulationTime   ... total stimulation time in sec
	#   Nelectrodes            ... number of stimulation sites  (4 for standard CR stimultion)
	#   sequence               ... string of form "a_b_c_d" where a,b,c,d are numbers specifying 
	#                              activated stimulation sites
	#  	fintra 				   ... intraburst frequency
	#   npb                    ... number of pulses per burst
	def __init__(self, fCR, totalStimulationTime, M, dt, sequence, fintra, npb ):

		# time interval between end of burst and beginning of next burst in ms
		self.fCR=fCR    # Hz

		# total stimulation time in sec
		self.Tstim=totalStimulationTime*1000.0 # ms

		# number of stimulation sites
		self.M=M

		# integration time step used in simulations
		self.dt=dt # ms

		# inverse time between first spikes of subsequent bursts
		self.burstFrequency = 0        # Hz

		# array that contains pre-calculated stimulus
		# kth element is value of stimulation current delivered at k*dt after stimulus onset
		self.signalOnTrain=np.zeros(1)

		# number of time steps in 'signalOnTrain'
		self.lengthSignalOneElectrode=0

		# integration time step at which current stimulus is delivered (beginning of next stimulus)
		# is used to get end of current stimulus and beginning of next one
		self.CurrentStimulusOnset=np.arange(self.M) # time steps

		# directory name for output from simulations using this stimulation protocol
		self.signalName=''

		# start of next cycle
		self.startNextCycle = 0
 
		self.sequence = sequence
		self.Sequence = np.array( sequence.split('_') ).astype( int )

		# intraburst frequency in Hz
		self.fintra = fintra 

		# number of pulses per burst
		self.npb = npb
		# print 'sequence'
		# print self.Sequence

	##################################
	#   function: initialize_Fixed_Sequence_CR_overlapping_Chaos_Paper(self )
	def initialize_fixed_Sequence_bursts_overlapping(self ):
	#
	#       initializes signal calculated the full signal of one electrode for one/Nelectrodes signal periods

		# number of integration time steps in a CR cycle
		self.time_steps_per_cycle_period = int(1000./(self.fCR * self.dt))

		# time steps of stimulus onset
		timeStepsToNextStimulusOnset = int( float( self.time_steps_per_cycle_period )/float(self.M) )
		self.stimOnsets = timeStepsToNextStimulusOnset * np.arange( self.M )

		########## generate pulse shape ##########
		### single pulse characteristics
		# pos rectangular pulses of unit amplitude, duration 0.2 ms followed by a 
		# negative counterpart of length 3 ms and amplitude 1/15, interpulse interval 1/130 s
		# positive rectangular pulse
		tStartPosPuls=0.2 # ms
		tStepStartPosPuls= int(tStartPosPuls/self.dt)
		lengthsPosRect=0.4*1.0 # ms
		tSteplengthsPosRect= int(lengthsPosRect/self.dt)

		# normalized such that integral over time in ms yields one
		AmpPosPuls=1.0/(0.4*1.0)

		# negative pulse
		tStartNegPuls=lengthsPosRect+0.2 # ms
		tStepStartNegPuls= int(tStartNegPuls/self.dt)
		lengthsNegRect=0.8*1.0 # ms         # motivated by Tass et al. 2012 (monkey study)
		tSteplengthsNegRect= int(lengthsNegRect/self.dt)
		AmpNegPuls= -(AmpPosPuls*lengthsPosRect)/lengthsNegRect # ensures charge balance by scaling the amplitude

		### minimal interval between subsequent pulses is adjusted to 130 Hz DBS pulsFrequency
		# number of pulses per burst
		
		pulsFrequency=self.fintra #  Hz
		pulsPeriod=1.0/float(pulsFrequency*0.001) # ms  // for 130 Hz this is approx 7.69 ms

		self.pulsLength = int( (pulsPeriod)/self.dt ) # (number of timesteps until next pulse starts 

		#### in case of burst stimuli one stimulus consistes of 
		# create signal for one electrode
		self.lengthSignalOneElectrode=self.pulsLength
		self.signalOnTrain=np.zeros(self.lengthSignalOneElectrode)


		# directory in which output for this stimulation protocol is saved
		self.signalName='/CR_stim_seq_'+self.sequence+'_fCR_'+str(self.fCR)+'_M_'+str(self.M)+'_fintra_'+str(self.fintra)+'_npb_'+str(self.npb)


		# construct a single stimulus
		for ktimeSteps in range( self.pulsLength ):

			kStep = ktimeSteps

			# add pos pulse
			if ((ktimeSteps)<tSteplengthsPosRect+tStepStartPosPuls) and (tStepStartPosPuls) <= (ktimeSteps):

				self.signalOnTrain[kStep]=AmpPosPuls

			# add neg pulse
			if ((ktimeSteps)<tSteplengthsNegRect+tStepStartNegPuls) and (tStepStartNegPuls) <= (ktimeSteps):

				self.signalOnTrain[kStep]=AmpNegPuls


		# original currents to contact
		# rows contain currents for all time steps during one CR cycle 
		# m th row contains the currents for the m th stimulus activation in the CR cycle
		self.originalCurrentsToContacts = np.zeros( ( self.M , self.time_steps_per_cycle_period ) )
		
		################################
		# check which pattern is delivered
		# self.Sequence = [0,0,0,0] corresponds to periodic stimulation
		# run through stimulation contacts
		if self.sequence != "0_0_0_0":
			for k in range( len( self.Sequence  ) ):

				m = self.Sequence[k]

				### add currents according to individual stimuli
				# run through pulses per burst
				for kpulse in range( self.npb ):

					koffset = kpulse * self.pulsLength
					# print( koffset ,self.time_steps_per_cycle_period )
					# run through indivudual pulses
					for kSignal in range( len(self.signalOnTrain) ):

						kcurrent = ( kSignal + koffset + self.stimOnsets[k] ) % self.time_steps_per_cycle_period

						self.originalCurrentsToContacts[ m, kcurrent ] = self.signalOnTrain[kSignal]

						# if kcurrent < len( self.originalCurrentsToContacts[ m ] ):

						# 	self.originalCurrentsToContacts[ k, kSignal+self.stimOnsets[m] ] = self.signalOnTrain[kSignal]
		
						# else:
						# 	print( 'Warning: '+str(m)+'th stimulus would reaches into next cycle and is cut off' )
		else:
			## implement periodic stimulation
			## activate all stimulation sites at 0
			# run through pulses per burst
			for kpulse in range( self.npb ):
				koffset = kpulse * self.pulsLength
				# run through time step during one stimulus
				for kSignal in range( len(self.signalOnTrain) ):
					kcurrent = ( kSignal + koffset ) % self.time_steps_per_cycle_period
					self.originalCurrentsToContacts[ :, kcurrent ] = self.signalOnTrain[kSignal]*np.ones( len( self.Sequence  ) )


		# import matplotlib.pyplot as plt 
		# fig = plt.figure()
		# ax = fig.add_subplot(111)
		# ax.imshow (self.originalCurrentsToContacts )

		# ax.set_aspect( 100*len(self.originalCurrentsToContacts)/float(M) )
		# plt.show()

		return 0

	##################################
	#   function: enterNewCycle(self)
	def enterNewCycle(self, timeStep):

		# currents to contacts
		self.current_Currents = np.copy( self.originalCurrentsToContacts )

		#print self.current_Currents
		self.startNextCycle += self.time_steps_per_cycle_period

	##################################
	#   function: getCurrent( self, timeStep )
	def getCurrent( self, timeStep ):
		#print timeStep, self.startNextCycle
		if timeStep == self.startNextCycle:
			currentOutputTimeStep = 0
			self.enterNewCycle( timeStep )
		if timeStep > 0:
			currentOutputTimeStep = timeStep % self.time_steps_per_cycle_period
			#print currentOutputTimeStep
		else:
			currentOutputTimeStep = 0
		if timeStep < 0:
			#print 'exit 1'
			return 0*self.originalCurrentsToContacts[:,0]
		# return current
		#print 'exit 2'
		#print timeStep
		return self.current_Currents[ :,currentOutputTimeStep ]

##################################
# shuffled CR
##################################
class shuffled_CR:

	## input:           
	#   timeBetweenBursts      ... time between end of burst and beginning of next burst
	#   totalStimulationTime   ... total stimulation time in sec
	#   Nelectrodes            ... number of stimulation sites  (4 for standard CR stimultion)
	#  	fintra 				   ... intraburst frequency
	#   npb                    ... number of pulses per burst
	#   T_shuffle              ... shuffle time after which a new CR sequence is drawn from all possible sequences in sec
	#   seed_sequence          ... seed for creating the random sequence
	def __init__(self, fCR, totalStimulationTime, M, dt, fintra, npb, Tshuffle, seed_sequence ):
		# print("SVS_CR_bursts")
		# time interval between end of burst and beginning of next burst in ms
		self.fCR=fCR    # Hz

		# total stimulation time in sec
		self.Tstim=totalStimulationTime*1000.0 # ms

		# number of stimulation sites
		self.M=M

		# integration time step used in simulations
		self.dt=dt # ms

		# inverse time between first spikes of subsequent bursts
		self.burstFrequency = 0        # Hz

		# array that contains pre-calculated stimulus
		# kth element is value of stimulation current delivered at k*dt after stimulus onset
		self.signalOnTrain=np.zeros(1)

		# number of time steps in 'signalOnTrain'
		self.lengthSignalOneElectrode=0

		# integration time step at which current stimulus is delivered (beginning of next stimulus)
		# is used to get end of current stimulus and beginning of next one
		self.CurrentStimulusOnset=np.arange(self.M) # time steps

		# directory name for output from simulations using this stimulation protocol
		self.signalName=''

		# start of next cycle
		self.startNextCycle = 0
 
		# intraburst frequency in Hz
		self.fintra = fintra 

		# number of pulses per burst
		self.npb = npb

		# shuffling time after which a new sequence is selected
		self.Tshuffle = Tshuffle # sec
		# get shuffling time in time steps
		self.Tshuffle_steps = int( self.Tshuffle * 1000.0/self.dt)

		## start with a random sequence
		self.nextSeedSequence = seed_sequence # integer that is used as seed to create the next CR sequence when Tshuffle has passed
		
		# directory in which output for this stimulation protocol is saved
		self.signalName='/shuffled_CR_stim_Tshuffle_'+str(self.Tshuffle)+'_seedSeq_'+str( seed_sequence )+'_fCR_'+str(self.fCR)+'_M_'+str(self.M)+'_fintra_'+str(self.fintra)+'_npb_'+str(self.npb)


		# print( '#################' )
		# print( self.signalName )

	##################################
	#   function: initialize_Fixed_Sequence_CR_overlapping_Chaos_Paper(self )
	def initialize_shuffled_CR(self ):
	#
	#       initializes signal calculated the full signal of one electrode for one/Nelectrodes signal periods

		# number of integration time steps in a CR cycle
		self.time_steps_per_cycle_period = int(1000./(self.fCR * self.dt))

		# time steps of stimulus onset
		timeStepsToNextStimulusOnset = int( float( self.time_steps_per_cycle_period )/float(self.M) )
		self.stimOnsets = timeStepsToNextStimulusOnset * np.arange( self.M )

		##########################################
		########## generate pulse shape ##########
		##########################################
		### single pulse characteristics
		# pos rectangular pulses of unit amplitude, duration 0.2 ms followed by a 
		# negative counterpart of length 3 ms and amplitude 1/15, interpulse interval 1/130 s
		# positive rectangular pulse
		tStartPosPuls=0.2 # ms
		tStepStartPosPuls= int(tStartPosPuls/self.dt)
		lengthsPosRect=0.4*1.0 # ms
		tSteplengthsPosRect= int(lengthsPosRect/self.dt)

		# normalized such that integral over time in ms yields one
		AmpPosPuls=1.0/(0.4*1.0)

		# negative pulse
		tStartNegPuls=lengthsPosRect+0.2 # ms
		tStepStartNegPuls= int(tStartNegPuls/self.dt)
		lengthsNegRect=0.8*1.0 # ms         # motivated by Tass et al. 2012 (monkey study)
		tSteplengthsNegRect= int(lengthsNegRect/self.dt)
		AmpNegPuls= -(AmpPosPuls*lengthsPosRect)/lengthsNegRect # ensures charge balance by scaling the amplitude

		### minimal interval between subsequent pulses is adjusted to 130 Hz DBS pulsFrequency
		# number of pulses per burst
		
		pulsFrequency=self.fintra #  Hz
		pulsPeriod=1.0/float(pulsFrequency*0.001) # ms  // approx 7.69 ms

		self.pulsLength = int( (pulsPeriod)/self.dt ) # (number of timesteps until next pulse starts 

		#### in case of burst stimuli one stimulus consistes of 
		# create signal for one electrode
		self.lengthSignalOneElectrode=self.pulsLength
		self.signalOnTrain=np.zeros(self.lengthSignalOneElectrode)



		#############################################
		########## construct a single stimulus ######
		#############################################
		for ktimeSteps in range( self.pulsLength ):

			kStep = ktimeSteps

			# add pos pulse
			if ((ktimeSteps)<tSteplengthsPosRect+tStepStartPosPuls) and (tStepStartPosPuls) <= (ktimeSteps):

				self.signalOnTrain[kStep]=AmpPosPuls

			# add neg pulse
			if ((ktimeSteps)<tSteplengthsNegRect+tStepStartNegPuls) and (tStepStartNegPuls) <= (ktimeSteps):

				self.signalOnTrain[kStep]=AmpNegPuls

		# original currents to contact
		# rows contain currents for all time steps during one CR cycle 
		# m th row contains the currents for the m th stimulus activation in the CR cycle
		self.originalCurrentsToContacts = np.zeros( ( self.M , self.time_steps_per_cycle_period ) )
		
		##########################################
		########## get next random sequence ######
		##########################################
		# set seed of random number generator
		np.random.seed( self.nextSeedSequence )
		# create random sequence
		self.Sequence = np.random.choice( self.M, self.M, replace=False)
		
		# get next random seedSequence
		self.nextSeedSequence = np.random.randint(0,2147483647) 


		####################################################################################
		########## get stimulus intensity for individual time bins and subpopulations ######
		####################################################################################
		# check which pattern is delivered
		# self.Sequence = [0,0,0,0] corresponds to periodic stimulation
		# run through stimulation contacts
		for k in range( len( self.Sequence  ) ):

			m = self.Sequence[k]

			### add currents according to individual stimuli
			# run through pulses per burst
			for kpulse in range( self.npb ):

				koffset = kpulse * self.pulsLength
				# print( koffset ,self.time_steps_per_cycle_period )
				# run through indivudual pulses
				for kSignal in range( len(self.signalOnTrain) ):

					kcurrent = ( kSignal + koffset + self.stimOnsets[k] ) % self.time_steps_per_cycle_period

					self.originalCurrentsToContacts[ m, kcurrent ] = self.signalOnTrain[kSignal]

		return 0


	##################################
	#   function: enterNewCycle(self)
	def enterNewCycle(self, timeStep):

		# currents to contacts
		self.current_Currents = np.copy( self.originalCurrentsToContacts )

		#print self.current_Currents
		self.startNextCycle += self.time_steps_per_cycle_period

	##################################
	#   function: getCurrent( self, timeStep )
	#	timeStep ... in units of 0.1 ms
	def getCurrent( self, timeStep ):

		# shuffle sequence
		if ( timeStep % self.Tshuffle_steps == 0 ):
			if timeStep > 0:
				# print('shuffle', timeStep)
				# generate array of stimulus intensities for each time bin and contact for new sequence
				self.initialize_shuffled_CR()
				self.current_Currents = np.copy( self.originalCurrentsToContacts )

		#print timeStep, self.startNextCycle
		if timeStep == self.startNextCycle:
			# print( self.Sequence )
			currentOutputTimeStep = 0
			self.enterNewCycle( timeStep )
		if timeStep > 0:
			currentOutputTimeStep = timeStep % self.time_steps_per_cycle_period
			#print currentOutputTimeStep
		else:
			currentOutputTimeStep = 0
		if timeStep < 0:

			return 0*self.originalCurrentsToContacts[:,0]

		return self.current_Currents[ :,currentOutputTimeStep ]

##################################
# stimulation of second and third site with given phase lag
##################################
class two_center_sites_at_phase_lag:

	## input:           
	#   fCR      			   ... CR frequency (Hz)
	#   M            		   ... total number of stimulation sites  (4 for standard CR stimultion)
	#   phase_shift            ... phase shift between stimuli delivered to second and third site
	#  	fintra 				   ... intraburst frequency (Hz)
	#   npb                    ... number of pulses per burst
	def __init__(self, fCR, M, dt, phase_shift, fintra, npb ):

		# time interval between end of burst and beginning of next burst in ms
		self.fCR=fCR    # Hz

		# total number of stimulation sites
		self.M=M

		# integration time step used in simulations
		self.dt=dt # ms

		# inverse time between first pulses of subsequent stimuli
		self.burstFrequency = 0        # Hz

		# array that contains pre-calculated stimulus
		# kth element is value of stimulation current delivered at k*dt after stimulus onset
		self.signalOnTrain=np.zeros(1)

		# number of time steps in 'signalOnTrain'
		self.lengthSignalOneElectrode=0

		# integration time step at which current stimulus is delivered (beginning of next stimulus)
		# is used to get end of current stimulus and beginning of next one
		self.CurrentStimulusOnset=np.arange(self.M) # time steps

		# directory name for output from simulations using this stimulation protocol
		self.signalName=''

		# start of next cycle
		self.startNextCycle = 0

		# phase shift between stimuli delivered to sites two and three 
		self.phase_shift = phase_shift

		# intraburst frequency in Hz
		self.fintra = fintra 

		# number of pulses per burst
		self.npb = npb
		# print 'sequence'
		# print self.Sequence

	##################################
	#   function: initialize_Fixed_Sequence_CR_overlapping_Chaos_Paper(self )
	def initialize_two_center_sites_at_phase_lag(self ):
	#
	#       initializes signal calculated the full signal of one electrode for one/Nelectrodes signal periods

		# number of integration time steps in a CR cycle
		self.time_steps_per_cycle_period = int(1000./(self.fCR * self.dt))
		# phase shift between stimuli delivered to second an third site in time steps
		self.phase_shift_time_steps = int( self.phase_shift*self.time_steps_per_cycle_period )

		# time steps of stimulus onset
		timeStepsToNextStimulusOnset = int( float( self.time_steps_per_cycle_period )/float(self.M) )
		self.stimOnsets = timeStepsToNextStimulusOnset * np.arange( self.M )

		########## generate pulse shape ##########
		### single pulse characteristics
		# pos rectangular pulses of unit amplitude, duration 0.2 ms followed by a 
		# negative counterpart of length 3 ms and amplitude 1/15, interpulse interval 1/130 s
		# positive rectangular pulse
		tStartPosPuls=0.2 # ms
		tStepStartPosPuls= int(tStartPosPuls/self.dt)
		lengthsPosRect=0.4*1.0 # ms
		tSteplengthsPosRect= int(lengthsPosRect/self.dt)

		# normalized such that integral over time in ms yields one
		AmpPosPuls=1.0/(0.4*1.0)

		# negative pulse
		tStartNegPuls=lengthsPosRect+0.2 # ms
		tStepStartNegPuls= int(tStartNegPuls/self.dt)
		lengthsNegRect=0.8*1.0 # ms         # motivated by Tass et al. 2012 (monkey study)
		tSteplengthsNegRect= int(lengthsNegRect/self.dt)
		AmpNegPuls= -(AmpPosPuls*lengthsPosRect)/lengthsNegRect # ensures charge balance by scaling the amplitude

		### minimal interval between subsequent pulses is adjusted to 130 Hz DBS pulsFrequency
		# number of pulses per burst		
		pulsFrequency=self.fintra #  Hz
		pulsPeriod=1.0/float(pulsFrequency*0.001) # ms  // for 130 Hz this is approx 7.69 ms

		self.pulsLength = int( (pulsPeriod)/self.dt ) # (number of timesteps until next pulse starts 

		########## create signal for one stimulation site ########## 
		self.lengthSignalOneElectrode=self.pulsLength
		self.signalOnTrain=np.zeros(self.lengthSignalOneElectrode)

		# directory in which output for this stimulation protocol is saved
		self.signalName='/get_J_phi_f_phi_'+str(self.phase_shift)+'_fCR_'+str(self.fCR)+'_M_'+str(self.M)+'_fintra_'+str(self.fintra)+'_npb_'+str(self.npb)

		# construct a single pulse stimulus
		# self.signalOnTrain ... contains the stimulus amplitude for each time bin
		for ktimeSteps in range( self.pulsLength ):

			kStep = ktimeSteps

			# add pos pulse
			if ((ktimeSteps)<tSteplengthsPosRect+tStepStartPosPuls) and (tStepStartPosPuls) <= (ktimeSteps):

				self.signalOnTrain[kStep]=AmpPosPuls

			# add neg pulse
			if ((ktimeSteps)<tSteplengthsNegRect+tStepStartNegPuls) and (tStepStartNegPuls) <= (ktimeSteps):

				self.signalOnTrain[kStep]=AmpNegPuls

		# original currents to contact
		# rows contain currents for all time steps during one CR cycle 
		# m th row contains the currents for the m th stimulus activation in the CR cycle
		self.originalCurrentsToContacts = np.zeros( ( self.M , self.time_steps_per_cycle_period ) )
		
		# add stimuli delivered to site two and three at phase lag self.phase_shift
		### add currents according to individual stimuli
		ksite = 1
		# number of time step between onset of stimulus delivered to 
		# second site (ksite = 1) and third site second site (ksite = 2)
		kphase_shift = 0
		# run through pulses per burst
		for kpulse in range( self.npb ):
			# koffset is the distance between subsequent stimulus pulses in time steps
			koffset = kpulse * self.pulsLength
			
			# run through indivudual pulses
			for kSignal in range( len(self.signalOnTrain) ):

				kcurrent = ( kSignal + koffset + kphase_shift ) % self.time_steps_per_cycle_period

				self.originalCurrentsToContacts[ ksite, kcurrent ] = self.signalOnTrain[kSignal]


		ksite = 2
		# number of time step between onset of stimulus delivered to 
		# second site (ksite = 1) and third site second site (ksite = 2)
		kphase_shift = self.phase_shift_time_steps
		# run through pulses per burst
		for kpulse in range( self.npb ):
			# koffset is the distance between subsequent stimulus pulses in time steps
			koffset = kpulse * self.pulsLength
			
			# run through indivudual pulses
			for kSignal in range( len(self.signalOnTrain) ):

				kcurrent = ( kSignal + koffset + kphase_shift ) % self.time_steps_per_cycle_period

				self.originalCurrentsToContacts[ ksite, kcurrent ] = self.signalOnTrain[kSignal]


				
		# import matplotlib.pyplot as plt 
		# # plt.plot( self.signalOnTrain )
		# fig = plt.figure()
		# ax = fig.add_subplot(111)
		# ax.imshow(self.originalCurrentsToContacts)
		# ax.set_aspect(1000)
		# plt.show()
		# exit()

		return 0

	##################################
	#   function: enterNewCycle(self)
	def enterNewCycle(self, timeStep):

		# currents to contacts
		self.current_Currents = np.copy( self.originalCurrentsToContacts )

		#print self.current_Currents
		self.startNextCycle += self.time_steps_per_cycle_period

	##################################
	#   function: getCurrent( self, timeStep )
	def getCurrent( self, timeStep ):
		#print timeStep, self.startNextCycle
		if timeStep == self.startNextCycle:
			currentOutputTimeStep = 0
			self.enterNewCycle( timeStep )
		if timeStep > 0:
			currentOutputTimeStep = timeStep % self.time_steps_per_cycle_period
			#print currentOutputTimeStep
		else:
			currentOutputTimeStep = 0
		if timeStep < 0:
			#print 'exit 1'
			return 0*self.originalCurrentsToContacts[:,0]
		# return current
		#print 'exit 2'
		#print timeStep
		return self.current_Currents[ :,currentOutputTimeStep ]


# for test runs
if __name__ == "__main__":

	# test fixed sequence CR
	if sys.argv[1]=='fixed_Sequence_bursts_overlapping':

		fCR = 21.0 # Hz
		totalStimulationTime = 200.0 # ms
		M = 4
		dt = 0.1 # ms
		sequence = "0_2_3_1"
		# sequence = "0_0_0_0"
		fintra = 130.0 # Hz
		npb = 1

		sequence=fixed_Sequence_bursts_overlapping(fCR, totalStimulationTime, M, dt, sequence, fintra, npb )
		sequence.initialize_fixed_Sequence_bursts_overlapping( )

		print( sequence.signalName )

		totNumberOfNeurons = M

		# prepare plot of CR sequence
		# time
		t0 = -100. # ms
		times = np.arange(t0, 1.0*totalStimulationTime-t0, dt)
		currents=np.zeros( (len(times), totNumberOfNeurons) )

		for kstep in range( len( times ) ):
			kstep = int(times[kstep]/dt)
			#print kstep + int(t0/dt)
			currents[ kstep ] = sequence.getCurrent( kstep + int(t0/dt) )

		#print currents
		import matplotlib.pyplot as plt

		figPulseShape=plt.figure()

		# spatio temporal sequence
		ax1=figPulseShape.add_subplot(311)
		ax1.imshow( np.transpose( currents ), vmin = -1, vmax = 1, cmap = "bwr" )
		ax1.set_aspect(  float(len( times ))/float(totNumberOfNeurons)  ) 
		ax1.set_xticks( ( np.arange(t0, totalStimulationTime, 100.0  ) - t0 )/dt )
		ax1.set_xticklabels( np.arange(t0, totalStimulationTime, 100.0  ) )
		ax1.set_xlabel('t (ms)')
		ax1.set_aspect(  float(0.1*len( times ))/float(totNumberOfNeurons)  ) 
		ax1.set_yticks(  np.arange(totNumberOfNeurons)   ) 
		ax1.set_ylabel( 'stim site' )

		# applied current to individual neurons
		ax2=figPulseShape.add_subplot(313)

		for kNeuron in range(totNumberOfNeurons):
			ax2.plot( times, currents[:,kNeuron] )
		
		ax2.set_xticks( ( np.arange(0, totalStimulationTime, 100.0  ) ) )
		ax2.set_xticklabels( np.arange(0, totalStimulationTime, 100.0  ) )

		ax2.set_xlabel('t (ms)')
		plt.show()


	# test SVS_CR_bursts
	if sys.argv[1]=='shuffled_CR':

		# test sequence
		NumberOfPulsesPerBurst=3
		Nelectrodes=4

		# integration time step ( currently things are adjusted for dt=0.1 ms )
		dt=0.1 # ms

		totalStimulationTime=2000.0  # ms
		fCR = 21.0 # Hz
		e_pulse_scale = 0.5
		intraburst_frequency = 130.0 # Hz
	
		# seed for sequence selection
		seed_sequence = 110
		Tshuffle = 0.0952  # sec

		sequence=shuffled_CR( fCR, totalStimulationTime, Nelectrodes, dt, intraburst_frequency, NumberOfPulsesPerBurst, Tshuffle, seed_sequence )
		sequence.initialize_shuffled_CR( )

		# prepare plot of CR sequence
		# time
		t0 = -100. # ms

		times = np.arange(t0, 1.5*totalStimulationTime, dt)
		currents=np.zeros( (len(times), Nelectrodes) )

		for kstep in range( len( times ) ):
			kstep = int(times[kstep]/dt)
			# print(sequence.getCurrent( kstep + int(t0/dt) ))
			currents[ kstep ] = sequence.getCurrent( kstep + int(t0/dt) )

		import matplotlib.pyplot as plt

		figPulseShape=plt.figure()

		# spatio temporal sequence
		ax1=figPulseShape.add_subplot(311)
		ax1.imshow( np.abs(np.transpose( currents )), cmap='binary' )
		ax1.set_aspect(  float(len( times ))/float(Nelectrodes)  ) 
		ax1.set_xlabel('t in ms')
		ax1.set_aspect(  float(0.1*len( times ))/float(Nelectrodes)  ) 
		ax1.set_yticks(  np.arange(Nelectrodes)   ) 
		ax1.set_ylabel( 'stim site' )
		ax1.set_xticks([1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,21000])
		ax1.set_xticklabels(["0","","","","","","","","","","1000","","","","","","","","","","2000"])

		
		# applied current to individual neurons
		ax2=figPulseShape.add_subplot(313)

		for kNeuron in range(Nelectrodes):
			ax2.plot( times, currents[:,kNeuron] )
		
		ax2.set_xlabel('t in ms')
		plt.show()


	# test two_center_sites_at_phase_lag
	if sys.argv[1]=='two_center_sites_at_phase_lag':

		fCR = 21.0 # Hz
		totalStimulationTime = 1000.0 # ms
		M = 4
		dt = 0.1 # ms
		phase_shift = 0.1
		fintra = 130.0 # Hz
		npb = 3
		# sequence = two_center_sites_at_phase_lag( fCR, totalStimulationTime, M, dt, phase_shift, fintra, npb )
		sequence = two_center_sites_at_phase_lag( fCR, M, dt, phase_shift, fintra, npb )
		sequence.initialize_two_center_sites_at_phase_lag()

		# prepare plot of CR sequence
		# time
		t0 = -100. # ms

		times = np.arange(t0, 1.5*totalStimulationTime, dt)
		currents=np.zeros( (len(times), M) )

		for kstep in range( len( times ) ):
			kstep = int(times[kstep]/dt)
			print(sequence.getCurrent( kstep + int(t0/dt) ))
			currents[ kstep ] = sequence.getCurrent( kstep + int(t0/dt) )

		import matplotlib.pyplot as plt

		figPulseShape=plt.figure()

		# spatio temporal sequence
		ax1=figPulseShape.add_subplot(311)
		ax1.imshow( np.abs(np.transpose( currents )), cmap='binary' )
		ax1.set_aspect(  1.0/10000 ) # float(len( times ))/float(M)  ) 
		ax1.set_xlabel('t in ms')
		ax1.set_aspect(  float(0.1*len( times ))/float(M)  ) 
		ax1.set_yticks(  np.arange(M)   ) 
		ax1.set_ylabel( 'stim site' )
		ax1.set_xticks([1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,21000])
		ax1.set_xticklabels(["0","","","","","","","","","","1000","","","","","","","","","","2000"])

		
		# applied current to individual neurons
		ax2=figPulseShape.add_subplot(313)

		for kNeuron in range(M):
			ax2.plot( times, currents[:,kNeuron] )
		
		ax2.set_xlabel('t in ms')
		plt.show()










