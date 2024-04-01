############################################################
### imports
import sys
import numpy as np
import os


############################################################
####### simulations for initial network generation
#### Simulation results are used to generate Figure 1 and 2 and to get
#### network configurations with which simulations on non-shuffled and shuffled CR are started.
############################################################
## pathToSimScript ... path to the simulation script "get_initial_networks_d.py"
## outputPath ... directory in which simulation results will be stored (is generated if it doesn't already exist) 
## pathForSubmissionStringTextFile ... in this file all the submission strings will be stored (used to submit jobs to computing cluster)
def run_simulations_initial_networks( pathToSimScript, outputPath, pathForSubmissionStringTextFile ):

    # seed values for network and noise generation
    seed_array = [10, 12, 14, 16, 18]
    # initial mean synaptic weights for which simulations are performed
    # initialMeanWeights = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
    initialMeanWeights = [0.0, 0.15, 0.3, 0.45] # , 0.6, 0.75, 0.9]
    # systems length scale
    L = 5.0 # mm
    # length scale of synaptic connections
    d_values = L*np.array([0.08, 0.4, 2.0]) # in units of L

    # loop over seeds, mean weights and d_values
    for seed in seed_array:
        for mw in initialMeanWeights:
            for d in d_values:

                # set output directory
                outputDirectory = outputPath + '/seed_'+str(seed)+'_d_'+str(d)+'_mw_init_'+str(mw)

                ### input parameters for get_initial_networks_d.py
                # sys.argv[1] ... output directory
                # sys.argv[2] ... seed for initial network
                # sys.argv[3] ... synaptic length scale d 
                # sys.argv[4] ... initial mean synaptic weight
                submissionString = 'srun python '+str(pathToSimScript)+'get_initial_networks_d.py '+ outputDirectory + ' ' + str(seed) + ' ' + str(d) + ' ' + str(mw) + ' ' + '\n' 
            
                # python command for running simulations is ...
                print( 'python '+str(pathToSimScript)+'get_initial_networks_d.py '+ outputDirectory + ' ' + str(seed) + ' ' + str(d) + ' ' + str(mw)  )

                # adds job to end of list
                with open(pathForSubmissionStringTextFile, "a") as myfile:
                                                
                    myfile.write( submissionString )
                    myfile.write( '##\n' )

# The following function can be used to continue initial network simulations from backups
# if they didn't finish or if longer simulations are needed.
def run_or_cont_simulations_initial_networks( pathToSimScript, pathToContinueScript, outputPath, pathForSubmissionStringTextFile ):

    # seed values for network and noise generation
    seed_array = [10]
    # initial mean synaptic weights for which simulations are performed
    initialMeanWeights = [0.0, 0.15, 0.3, 0.45]
    # systems length scale
    L = 5.0 # mm
    # length scale of synaptic connections
    d_values = L*np.array([0.08, 0.4, 2.0]) # in units of L

    # loop over seeds, mean weights and d_values
    for seed in seed_array:
        for mw in initialMeanWeights:
            for d in d_values:

                # set output directory
                outputDirectory = outputPath + '/seed_'+str(seed)+'_d_'+str(d)+'_mw_init_'+str(mw)

                # check whether simulation has already been performed
                # check whether simulation needs to be continued or started again
                ## find last backup 
                lastBackup=''
                ## all possible backups are 
                TBackupSteps = np.arange(1000,20000,1000)
                pathToBackup = outputDirectory

                for backups in [ outputDirectory+"/"+str(X)+"_sec" for X in TBackupSteps]:
                    #print('#', backups + "/systemState.npy")
                    if os.path.isfile( backups + "/systemState.npy" ):
                        # print('FOUND',backups)
                        lastBackup = backups

                # no backup found, new simulation
                if lastBackup == '':
                    ### input parameters for get_initial_networks_d.py
                    # sys.argv[1] ... output directory
                    # sys.argv[2] ... seed for initial network
                    # sys.argv[3] ... synaptic length scale d 
                    # sys.argv[4] ... initial mean synaptic weight
                    submissionString = 'srun python '+str(pathToSimScript)+'get_initial_networks_d.py '+ outputDirectory + ' ' + str(seed) + ' ' + str(d) + ' ' + str(mw) + ' ' + '\n' 
                else:
                    # backup found -> continue simulations
                    Trelax = 20020 # sec
                    backupDirectory = lastBackup
                    # print(backupDirectory)
                    backupPar = outputDirectory

                    ### input parameters for get_initial_networks_d.py
                    # sys.argv[1] ... backup directory (from which simulation is started)
                    # sys.argv[2] ... simulated relaxation time after cessation of stimulation in seconds
                    # sys.argv[3] ... output directory
                    shellCommand = 'python '+str(pathToContinueScript)+'relaxation.py '+ backupDirectory + " " + str(Trelax) + " " + str(outputDirectory) + " " + backupPar +'\n' 
                    submissionString = 'srun ' + shellCommand

                    # python command for running simulations is ...
                    print( shellCommand ) 


                # adds job to end of list
                with open(pathForSubmissionStringTextFile, "a") as myfile:
                                                
                    myfile.write( submissionString )
                    myfile.write( '##\n' )

############################################################
####### simulations for CR stimulation of networks with different synaptic length scales
#### Figures 4, 5, 6
############################################################
## pathToSimScript ... path to the simulation scripts "CR_stimulation_spatial_stimulus_profile.py" (non-shuffled CR) and "shuffled_CR_stimulation_spatial_stimulus_profile.py" (shuffled CR)
## outputToInitialNetworks .. the path at which initial networks are saved
## outputPath ... directory in which simulation results will be stored (is generated if it doesn't already exist) 
## pathForSubmissionStringTextFile ... in this file all the submission strings will be stored (used to submit jobs to computation cluster)
# 1a) simulations for non-shuffled CR
def run_simulations_CR_stimulation_Results_1( pathToSimScript, outputToInitialNetworks, outputPath, pathForSubmissionStringTextFile ):

    # seed values for network and noise generation
    seed_array = [10]
    # initial mean synaptic weights for which simulations are performed
    initialMeanWeights = [0.45]
    # systems length scale
    L = 5.0 # mm
    # length scale of synaptic connections
    d_values = L*np.array([0.08, 0.4, 2.0]) # in units of L
    # stimulation amplitude
    Astim = 1.0
    # CR frequencies
    fCR_values = [4.0, 10.0, 21.0] # Hz
    # number of pulses
    nPulse_values = [1,3]
    # stimulation time
    Tstim = 1020 # sec
    # number of stimulation sites
    nSites = 4
    # simuled CR sequences
    CRsequence_values = ["0_1_2_3", "0_1_3_2", "0_2_1_3", "0_2_3_1", "0_3_1_2", "0_3_2_1"]
    # intra burst frequency (this corresponds to the inverse time between two subsequent pulses within a single stimulus burst)
    # this should not exceed about 130 Hz as individual pulses might get cut off
    # for single pulse stimuli (nPulse=1) this can just be set to 130 Hz
    fintra = 130.0 # Hz
    # distance between stimulation sites in units of system length scale (L=5 cm)
    dsites = 0.25
    # width of stimulus profile in units of dsites
    erw = 0.25

    # loop over seeds, mean weights and d_values
    for seed in seed_array:
        for mw in initialMeanWeights:
            for d in d_values:
                for fCR in fCR_values:
                    for nPulse in nPulse_values:
                        for CRsequence in CRsequence_values:

                            # set output directory
                            # the full output path will be:
                            # outputDirectory+ '/CR_stim_seq_'+CRsequence+'_fCR_'+fCR+'_M_'+nSites+'_fintra_'+fintra+'_npb_'+nPulse+'_Astim_'+Astim+'_Tstim_'+Tstim+'_dsites_'+dsites+'_erw_'+erw 
                            outputDirectory = outputPath + '/seed_'+str(seed)+'_d_'+str(d)+'_mw_init_'+str(mw)
                            
                            # set path to initial network from which simulation is started
                            backupDirectory = outputToInitialNetworks+ "/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_"+str(mw)+"/5000_sec"
                            
                            ### input parameters for get_initial_networks_d.py
                            # sys.argv[1] ... backup directory (from which simulation is started)
                            # sys.argv[2] ... stimulation amplitude in units of Astim
                            # sys.argv[3] ... CR frequency in Hz (time during which each site receives one stimulus (single pulse or burst))
                            # sys.argv[4] ... number of pulses per burst stimulus
                            # sys.argv[5] ... total stimulation time in seconds
                            # sys.argv[6] ... number of stimulation sites
                            # sys.argv[7] ... string specifying CR sequence, e.g., 0_1_2_3 to activity sites in order 0-1-2-3
                            # sys.argv[8] ... output directory
                            # sys.argv[9] ... intraburst spike frequency, fintra, in Hz ( pulse duration is 1/fintra ), should not exceed 150 ms as otherwise parts of pulses are cut off, see functions_stim.py
                            # sys.argv[10]... distance between stimulation sites in units of L=5.0 mm, use 0.25 for four equally spaced sites (at -3/8 L, -1/8 L, 1/8 L, 3/8 L )
                            # sys.argv[11]... width of stimulation profile in units of dsites, use 0.5
                            shellCommand = 'python '+str(pathToSimScript)+'CR_stimulation_spatial_stimulus_profile.py '+ backupDirectory + " " + str(Astim) + " " + str(fCR) + " " + str(nPulse) + " " + str(Tstim) + " " + str(nSites) + " " + CRsequence + " " + outputDirectory + " " + str(fintra) + " " + str(dsites) + " " + str(erw) + '\n' 
                            submissionString = 'srun ' + shellCommand
                        

                            # python command for running simulations is ...
                            print( shellCommand ) 


                            # adds job to end of list
                            with open(pathForSubmissionStringTextFile, "a") as myfile:
                                                            
                                myfile.write( submissionString )
                                myfile.write( '##\n' )

# 2a) simulations for shuffled CR
def run_simulations_shuffled_CR_stimulation_Results_1( pathToSimScript, outputToInitialNetworks, outputPath, pathForSubmissionStringTextFile ):

    # seed values for network and noise generation
    seed_array = [10]
    # initial mean synaptic weights for which simulations are performed
    initialMeanWeights = [0.45]
    # systems length scale
    L = 5.0 # mm
    # length scale of synaptic connections
    d_values = L*np.array([0.08, 0.4, 2.0]) # in units of L
    # stimulation amplitude
    # FIXME select amplitude
    # Astim = 1.0
    Astim = 0.1
    # CR frequencies
    fCR_values = [4.0, 10.0, 21.0] # Hz
    # number of pulses
    nPulse_values = [1,3]
    # stimulation time
    Tstim = 1020 # sec
    # number of stimulation sites
    nSites = 4
    # shuffled CR with a shuffle Period of 1/fCR is simlated for the following 
    # seeds for random sequence generation
    seedSequence_values = [100, 110, 120, 130, 140]

    # intra burst frequency (this corresponds to the inverse time between two subsequent pulses within a single stimulus burst)
    # this should not exceed about 130 Hz as individual pulses might get cut off
    # for single pulse stimuli (nPulse=1) this can just be set to 130 Hz
    fintra = 130.0 # Hz
    # distance between stimulation sites in units of system length scale (L=5 cm)
    dsites = 0.25
    # width of stimulus profile in units of dsites
    erw = 0.25

    # loop over seeds, mean weights and d_values
    for seed in seed_array:
        for mw in initialMeanWeights:
            for d in d_values:
                for fCR in fCR_values:
                    for nPulse in nPulse_values:
                        for seedSequence in seedSequence_values:

                            # set output directory
                            # the full output path will be:
                            # outputDirectory+ '/CR_stim_seq_'+CRsequence+'_fCR_'+fCR+'_M_'+nSites+'_fintra_'+fintra+'_npb_'+nPulse+'_Astim_'+Astim+'_Tstim_'+Tstim+'_dsites_'+dsites+'_erw_'+erw 
                            outputDirectory = outputPath + '/seed_'+str(seed)+'_d_'+str(d)+'_mw_init_'+str(mw)
                            
                            # set path to initial network from which simulation is started
                            backupDirectory = outputToInitialNetworks+ "/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_"+str(mw)+"/5000_sec"
                            
                            ### input parameters for get_initial_networks_d.py
                            # sys.argv[1] ... backup directory (from which simulation is started)
                            # sys.argv[2] ... stimulation amplitude in units of Astim
                            # sys.argv[3] ... CR frequency in Hz (time during which each site receives one stimulus (single pulse or burst))
                            # sys.argv[4] ... number of pulses per burst stimulus
                            # sys.argv[5] ... total stimulation time in seconds
                            # sys.argv[6] ... number of stimulation sites
                            # sys.argv[7] ... seed for random sequence generation
                            # sys.argv[8] ... output directory
                            # sys.argv[9] ... intraburst spike frequency, fintra, in Hz ( pulse duration is 1/fintra ), should not exceed 150 ms as otherwise parts of pulses are cut off, see functions_stim.py
                            # sys.argv[10]... shuffle period
                            Tshuffle = np.round( 1.0/fCR , 4) # seconds
                            # sys.argv[11]... distance between stimulation sites in units of L=5.0 mm, use 0.25 for four equally spaced sites (at -3/8 L, -1/8 L, 1/8 L, 3/8 L )
                            # sys.argv[12]... width of stimulation profile in units of dsites, use 0.5
                            shellCommand = 'python '+str(pathToSimScript)+'shuffled_CR_stimulation_spatial_stimulus_profile.py '+ backupDirectory + " " + str(Astim) + " " + str(fCR) + " " + str(nPulse) + " " + str(Tstim) + " " + str(nSites) + " " + str(seedSequence) + " " + outputDirectory + " " + str(fintra) + " " + str(Tshuffle) + " " + str(dsites) + " " + str(erw) + '\n' 
                            submissionString = 'srun ' + shellCommand
                        

                            # python command for running simulations is ...
                            print( shellCommand ) 


                            # adds job to end of list
                            with open(pathForSubmissionStringTextFile, "a") as myfile:
                                                            
                                myfile.write( submissionString )
                                myfile.write( '##\n' )

# 1b) relaxation after non-shuffled CR
def run_or_cont_simulations_CR_stimulation_Results_1_relaxation( pathToSimScript, outputToInitialNetworks, outputPath, pathForSubmissionStringTextFile ):

    # seed values for network and noise generation
    seed_array = [10]
    # initial mean synaptic weights for which simulations are performed
    initialMeanWeights = [0.45]
    # systems length scale
    L = 5.0 # mm
    # length scale of synaptic connections
    d_values = L*np.array([0.08, 0.4, 2.0]) # in units of L
    # stimulation amplitude
    Astim = 1.0
    # CR frequencies
    fCR_values = [4.0, 10.0, 21.0] # Hz
    # number of pulses
    nPulse_values = [1,3]
    # stimulation time
    Tstim = 1020.0 # sec
    # number of stimulation sites
    nSites = 4
    # simuled CR sequences
    CRsequence_values = ["0_1_2_3", "0_1_3_2", "0_2_1_3", "0_2_3_1", "0_3_1_2", "0_3_2_1"]
    # intra burst frequency (this corresponds to the inverse time between two subsequent pulses within a single stimulus burst)
    # this should not exceed about 130 Hz as individual pulses might get cut off
    # for single pulse stimuli (nPulse=1) this can just be set to 130 Hz
    fintra = 130.0 # Hz
    # distance between stimulation sites in units of system length scale (L=5 cm)
    dsites = 0.25
    # width of stimulus profile in units of dsites
    erw = 0.25

    # time of backup from which relaxation is started in seconds
    TstartRelax = 6000 # sec
    # time after cessation of stimulation that is simulated
    Trelax = 14020 # sec

    # loop over seeds, mean weights and d_values
    for seed in seed_array:
        for mw in initialMeanWeights:
            for d in d_values:
                for fCR in fCR_values:
                    for nPulse in nPulse_values:
                        for CRsequence in CRsequence_values:

                            # outputDirectory+ '/CR_stim_seq_'+CRsequence+'_fCR_'+fCR+'_M_'+nSites+'_fintra_'+fintra+'_npb_'+nPulse+'_Astim_'+Astim+'_Tstim_'+Tstim+'_dsites_'+dsites+'_erw_'+erw 
                            outputDirectory = outputPath + "/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_"+str(mw)+"/CR_stim_seq_"+CRsequence+"_fCR_"+str(fCR)+"_M_4_fintra_"+str(fintra)+"_npb_"+str(nPulse)+"_Astim_"+str(Astim)+"_Tstim_"+str(Tstim)+"_dsites_"+str(dsites)+"_erw_"+str(erw)+"_TstartRelax_"+str(TstartRelax)+"_sec"
                            
                            # check whether simulation has already been performed
                            # check whether simulation needs to be continued or started again
                            ## find last backup 
                            lastBackup=''
                            ## all possible backups are 
                            TBackupSteps = np.arange(6000,20000,200)
                            pathToBackup = outputDirectory

                            for backups in [ outputDirectory+"/"+str(X)+"_sec" for X in TBackupSteps]:
                                #print('#', backups + "/systemState.npy")
                                if os.path.isfile( backups + "/systemState.npy" ):
                                    # print('FOUND',backups)
                                    lastBackup = backups

                            # no backup found, new simulation
                            if lastBackup == '':
                                Trelax = 4020 # sec
                                backupDirectory = outputToInitialNetworks+ "/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_"+str(mw)+"/CR_stim_seq_"+CRsequence+"_fCR_"+str(fCR)+"_M_4_fintra_"+str(fintra)+"_npb_"+str(nPulse)+"_Astim_"+str(Astim)+"_Tstim_"+str(Tstim)+"_dsites_"+str(dsites)+"_erw_"+str(erw)+"/"+str(TstartRelax)+"_sec"
                                backupPar = backupDirectory
                            else:
                                # backup found -> continue simulations
                                Trelax = 14020 # sec
                                backupDirectory = lastBackup
                                backupPar =       outputToInitialNetworks+ "/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_"+str(mw)+"/CR_stim_seq_"+CRsequence+"_fCR_"+str(fCR)+"_M_4_fintra_"+str(fintra)+"_npb_"+str(nPulse)+"_Astim_"+str(Astim)+"_Tstim_"+str(Tstim)+"_dsites_"+str(dsites)+"_erw_"+str(erw)+"/"+str(TstartRelax)+"_sec/"

                            # set output directory
                            # the full output path will be:

                            ### input parameters for get_initial_networks_d.py
                            # sys.argv[1] ... backup directory (from which simulation is started)
                            # sys.argv[2] ... simulated relaxation time after cessation of stimulation in seconds
                            # sys.argv[3] ... output directory
                            # sys.argv[4] ... only needed when continuing previous relaxation simulation
                            shellCommand = 'python '+str(pathToSimScript)+'relaxation.py '+ backupDirectory + " " + str(Trelax) + " " + str(outputDirectory) + " " + backupPar + '\n' 
                            submissionString = 'srun ' + shellCommand
                        

                            # python command for running simulations is ...
                            print( shellCommand ) 


                            # adds job to end of list
                            with open(pathForSubmissionStringTextFile, "a") as myfile:
                                                            
                                myfile.write( submissionString )
                                myfile.write( '##\n' )

# 2b) relaxation after shuffled CR
def run_or_cont_simulations_shuffled_CR_stimulation_Results_1_relaxation( pathToSimScript, outputToInitialNetworks, outputPath, pathForSubmissionStringTextFile ):

    # seed values for network and noise generation
    seed_array = [10]
    # initial mean synaptic weights for which simulations are performed
    initialMeanWeights = [0.45]
    # systems length scale
    L = 5.0 # mm
    # length scale of synaptic connections
    d_values = L*np.array([0.08, 0.4, 2.0]) # in units of L
    # stimulation amplitude
    Astim = 1.0
    # CR frequencies
    fCR_values = [4.0, 10.0, 21.0] # Hz
    # number of pulses
    nPulse_values = [1,3]
    # stimulation time
    Tstim = 1020.0 # sec
    # number of stimulation sites
    nSites = 4
    # shuffled CR with a shuffle Period of 1/fCR is simlated for the following 
    # seeds for random sequence generation
    # seedSequence_values = [100, 110, 120, 130, 140]
    seedSequence_values = [100]

    # intra burst frequency (this corresponds to the inverse time between two subsequent pulses within a single stimulus burst)
    # this should not exceed about 130 Hz as individual pulses might get cut off
    # for single pulse stimuli (nPulse=1) this can just be set to 130 Hz
    fintra = 130.0 # Hz
    # distance between stimulation sites in units of system length scale (L=5 cm)
    dsites = 0.25
    # width of stimulus profile in units of dsites
    erw = 0.25

    # time for backup from which relaxation is started in seconds
    TstartRelax = 6000 # sec
    # time after cessation of stimulation that is simulated
    Trelax = 4020 # sec

    # loop over seeds, mean weights and d_values
    for seed in seed_array:
        for mw in initialMeanWeights:
            for d in d_values:
                for fCR in fCR_values:
                    for nPulse in nPulse_values:
                        for seedSequence in seedSequence_values:

                            # calculate shuffle period
                            Tshuffle = np.round( 1.0/fCR , 4) # seconds
                        
                            # set output directory
                            # the full output path will be:
                            outputDirectory = outputPath + "/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_"+str(mw)+"/shuffled_CR_stim_Tshuffle_"+str(Tshuffle)+"_seedSeq_"+str(seedSequence)+"_fCR_"+str(fCR)+"_M_4_fintra_"+str(fintra)+"_npb_"+str(nPulse)+"_Astim_"+str(Astim)+"_Tstim_"+str(Tstim)+"_seedSeq_"+str(seedSequence)+"_dsites_"+str(dsites)+"_erw_"+str(erw)+"_TstartRelax_"+str(TstartRelax)+"_sec"
                            
                            # check whether simulation has already been performed
                            # check whether simulation needs to be continued or started again
                            ## find last backup 
                            lastBackup=''
                            ## all possible backups are 
                            TBackupSteps = np.arange(6000,20000,200)
                            pathToBackup = outputDirectory

                            for backups in [ outputDirectory+"/"+str(X)+"_sec" for X in TBackupSteps]:
                                #print('#', backups + "/systemState.npy")
                                if os.path.isfile( backups + "/systemState.npy" ):
                                    # print('FOUND',backups)
                                    lastBackup = backups

                            # no backup found, new simulation
                            if lastBackup == '':
                                Trelax = 4020 # sec
                                backupDirectory = outputToInitialNetworks+ "/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_"+str(mw)+"/shuffled_CR_stim_Tshuffle_"+str(Tshuffle)+"_seedSeq_"+str(seedSequence)+"_fCR_"+str(fCR)+"_M_4_fintra_"+str(fintra)+"_npb_"+str(nPulse)+"_Astim_"+str(Astim)+"_Tstim_"+str(Tstim)+"_seedSeq_"+str(seedSequence)+"_dsites_"+str(dsites)+"_erw_"+str(erw)+"/"+str(TstartRelax)+"_sec"   
                                backupPar = backupDirectory
                            else:
                                # backup found -> continue simulations
                                Trelax = 14020 # sec
                                backupDirectory = lastBackup
                                backupPar       = outputToInitialNetworks+ "/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_"+str(mw)+"/shuffled_CR_stim_Tshuffle_"+str(Tshuffle)+"_seedSeq_"+str(seedSequence)+"_fCR_"+str(fCR)+"_M_4_fintra_"+str(fintra)+"_npb_"+str(nPulse)+"_Astim_"+str(Astim)+"_Tstim_"+str(Tstim)+"_seedSeq_"+str(seedSequence)+"_dsites_"+str(dsites)+"_erw_"+str(erw)+"/"+str(TstartRelax)+"_sec/"


                            ### input parameters for get_initial_networks_d.py
                            # sys.argv[1] ... backup directory (from which simulation is started)
                            # sys.argv[2] ... simulated relaxation time after cessation of stimulation in seconds
                            # sys.argv[3] ... output directory
                            # sys.argv[4] ... only needed when continuing previous relaxation simulation
                            shellCommand = 'python '+str(pathToSimScript)+'relaxation.py '+ backupDirectory + " " + str(Trelax) + " " + str(outputDirectory) + " " + backupPar +'\n' 
                            submissionString = 'srun ' + shellCommand
     
                            # python command for running simulations is ...
                            print( shellCommand ) 


                            # adds job to end of list
                            with open(pathForSubmissionStringTextFile, "a") as myfile:
                                                            
                                myfile.write( submissionString )
                                myfile.write( '##\n' )


############################################################
####### simulations of two-site stimulation as illustrated in Figure 8
#### Results were used in Figure 8 and are use to obtain approximations 
#### shown in Figure 9.
############################################################
# d = 0.4 L
def run_simulations_get_J_phi_f( pathToSimScript, outputToInitialNetworks, outputPath, pathForSubmissionStringTextFile ):

    # seed values for network and noise generation
    seed_array = [10,12,14,16,18]
    # initial mean synaptic weights for which simulations are performed
    initialMeanWeights = [0.45]
    # systems length scale
    L = 5.0 # mm
    # length scale of synaptic connections
    d_values = L*np.array([0.4]) # in units of L
    # stimulation amplitude
    Astim = 1.0
    # CR frequencies
    fCR_values = np.round( np.arange(4.0,22.0, 1.0) , 1) # Hz
    # number of pulses
    nPulse_values = [1,3]
    # stimulation time
    Tstim = 100 # sec
    # number of stimulation sites
    nSites = 4
    # list of phase shifts between stimuli delivered to the second and third site
    phase_shifts = np.round( np.arange(0.0,1.0, 0.05) , 2)
    # intra burst frequency (this corresponds to the inverse time between two subsequent pulses within a single stimulus burst)
    # this should not exceed about 130 Hz as individual pulses might get cut off
    # for single pulse stimuli (nPulse=1) this can just be set to 130 Hz
    fintra = 130.0 # Hz
    # distance between stimulation sites in units of system length scale (L=5 cm)
    dsites = 0.25
    # Width of stimulus profile in units of dsites note 
    # that pi is added in simulation script.
    erw = 0.25

    counter =  0

    # loop over seeds, mean weights and d_values
    for seed in seed_array:
        for mw in initialMeanWeights:
            for d in d_values:
                for fCR in fCR_values:
                    for nPulse in nPulse_values:
                        for phase_shift in phase_shifts:

                            # set output directory
                            # the full output path will be:
                            # outputDirectory+ '/CR_stim_seq_'+CRsequence+'_fCR_'+fCR+'_M_'+nSites+'_fintra_'+fintra+'_npb_'+nPulse+'_Astim_'+Astim+'_Tstim_'+Tstim+'_dsites_'+dsites+'_erw_'+erw 
                            outputDirectory = outputPath + '/seed_'+str(seed)+'_d_'+str(d)+'_mw_init_'+str(mw)
                            
                            # set path to initial network from which simulation is started
                            backupDirectory = outputToInitialNetworks+ "/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_"+str(mw)+"/5000_sec"
                            
                            ### input parameters for get_initial_networks_d.py
                            # sys.argv[1] ... backup directory (from which simulation is started)
                            # sys.argv[2] ... stimulation amplitude in units of Astim
                            # sys.argv[3] ... CR frequency in Hz (time during which each site receives one stimulus (single pulse or burst))
                            # sys.argv[4] ... number of pulses per burst stimulus
                            # sys.argv[5] ... stimulation time in seconds
                            # sys.argv[6] ... phase shift between stimuli delivered to second and third site
                            # sys.argv[7] ... output directory
                            # sys.argv[8] ... intraburst spike frequency, fintra, in Hz ( pulse duration is 1/fintra ), should not exceed 150 ms as otherwise parts of pulses are cut off, see functions_stim.py
                            # sys.argv[9]... distance between stimulation sites in units of L=5.0 mm, use 0.25 for four equally spaced sites (at -3/8 L, -1/8 L, 1/8 L, 3/8 L )
                            # sys.argv[10]... width of stimulation profile in units of dsites, use 0.5
                            shellCommand = 'python '+str(pathToSimScript)+'get_J_phi_f_spatial_stimulus_profile.py '+ backupDirectory + " " + str(Astim) + " " + str(fCR) + " " + str(nPulse) + " " + str(Tstim) + " " + str(phase_shift) + " " + outputDirectory + " " + str(fintra) + " " + str(dsites) + " " + str(erw) + '\n' 
                            submissionString = 'srun ' + shellCommand
                        

                            # python command for running simulations is ...
                            print( shellCommand ) 

                            # adds job to end of list
                            with open(pathForSubmissionStringTextFile, "a") as myfile:
                                                            
                                myfile.write( submissionString )
                                myfile.write( '##\n' )

                            counter+=1
                            print(counter)

    print(counter)

# d = 0.08 L and 2 L
def run_simulations_get_J_phi_f_ForComparissonToSimulations( pathToSimScript, outputToInitialNetworks, outputPath, pathForSubmissionStringTextFile ):

    # seed values for network and noise generation
    seed_array = [10,12,14,16,18]
    # initial mean synaptic weights for which simulations are performed
    initialMeanWeights = [0.45]
    # systems length scale
    L = 5.0 # mm
    # length scale of synaptic connections
    d_values = L*np.array([0.08, 2.0]) # in units of L
    # stimulation amplitude
    Astim = 1.0
    # CR frequencies
    # fCR_values = np.round( [4.0, 10.0, 21.0] , 1) # Hz
    fCR_values = np.round( np.arange(4.0,22.0, 1.0) , 1) # Hz
    # number of pulses
    nPulse_values = [1,3]
    # stimulation time
    Tstim = 100 # sec
    # number of stimulation sites
    nSites = 4
    # list of phase shifts between stimuli delivered to the second and third site
    phase_shifts = np.round( [0.0, 0.25, 0.5, 0.75] , 2)
    # intra burst frequency (this corresponds to the inverse time between two subsequent pulses within a single stimulus burst)
    # this should not exceed about 130 Hz as individual pulses might get cut off
    # for single pulse stimuli (nPulse=1) this can just be set to 130 Hz
    fintra = 130.0 # Hz
    # distance between stimulation sites in units of system length scale (L=5 cm)
    dsites = 0.25
    # width of stimulus profile in units of dsites
    erw = 0.25

    counter =  0

    # loop over seeds, mean weights and d_values
    for seed in seed_array:
        for mw in initialMeanWeights:
            for d in d_values:
                for fCR in fCR_values:
                    for nPulse in nPulse_values:
                        for phase_shift in phase_shifts:

                            # set output directory
                            # the full output path will be:
                            # outputDirectory+ '/CR_stim_seq_'+CRsequence+'_fCR_'+fCR+'_M_'+nSites+'_fintra_'+fintra+'_npb_'+nPulse+'_Astim_'+Astim+'_Tstim_'+Tstim+'_dsites_'+dsites+'_erw_'+erw 
                            outputDirectory = outputPath + '/seed_'+str(seed)+'_d_'+str(d)+'_mw_init_'+str(mw)
                            
                            # set path to initial network from which simulation is started
                            backupDirectory = outputToInitialNetworks+ "/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_"+str(mw)+"/5000_sec"
                            
                            ### input parameters for get_initial_networks_d.py
                            # sys.argv[1] ... backup directory (from which simulation is started)
                            # sys.argv[2] ... stimulation amplitude in units of Astim
                            # sys.argv[3] ... CR frequency in Hz (time during which each site receives one stimulus (single pulse or burst))
                            # sys.argv[4] ... number of pulses per burst stimulus
                            # sys.argv[5] ... stimulation time in seconds
                            # sys.argv[6] ... phase shift between stimuli delivered to second and third site
                            # sys.argv[7] ... output directory
                            # sys.argv[8] ... intraburst spike frequency, fintra, in Hz ( pulse duration is 1/fintra ), should not exceed 150 ms as otherwise parts of pulses are cut off, see functions_stim.py
                            # sys.argv[9]... distance between stimulation sites in units of L=5.0 mm, use 0.25 for four equally spaced sites (at -3/8 L, -1/8 L, 1/8 L, 3/8 L )
                            # sys.argv[10]... width of stimulation profile in units of dsites, use 0.5
                            shellCommand = 'python '+str(pathToSimScript)+'get_J_phi_f_spatial_stimulus_profile.py '+ backupDirectory + " " + str(Astim) + " " + str(fCR) + " " + str(nPulse) + " " + str(Tstim) + " " + str(phase_shift) + " " + outputDirectory + " " + str(fintra) + " " + str(dsites) + " " + str(erw) + '\n' 
                            submissionString = 'srun ' + shellCommand
                        

                            # python command for running simulations is ...
                            print( shellCommand ) 


                            # adds job to end of list
                            with open(pathForSubmissionStringTextFile, "a") as myfile:
                                                            
                                myfile.write( submissionString )
                                myfile.write( '##\n' )

                            counter+=1
                            print(counter)

    print(counter)


############################################################
####### simulations of non-shuffled and shuffled CR for data points shown 
####### in Figure 9. 
############################################################
# 1) non-shuffled CR
def run_simulations_CR_stimulation_CompareToTheory( pathToSimScript, outputToInitialNetworks, outputPath, pathForSubmissionStringTextFile ):

    # seed values for network and noise generation
    seed_array = [10]
    # initial mean synaptic weights for which simulations are performed
    initialMeanWeights = [0.45]
    # systems length scale
    L = 5.0 # mm
    # length scale of synaptic connections
    d_values = L*np.array([0.08, 0.4, 2.0]) # in units of L
    # stimulation amplitude
    Astim = 1.0
    # CR frequencies
    fCR_values = np.arange(4.0,21.1,1.0) # Hz
    # number of pulses
    nPulse_values = [1,3]
    # stimulation time
    Tstim = 7220 # sec
    # number of stimulation sites
    nSites = 4
    # simuled CR sequences
    CRsequence_values = ["0_1_2_3", "0_1_3_2", "0_2_1_3", "0_2_3_1", "0_3_1_2", "0_3_2_1"]
    # intra burst frequency (this corresponds to the inverse time between two subsequent pulses within a single stimulus burst)
    # this should not exceed about 130 Hz as individual pulses might get cut off
    # for single pulse stimuli (nPulse=1) this can just be set to 130 Hz
    fintra = 130.0 # Hz
    # distance between stimulation sites in units of system length scale (L=5 cm)
    dsites = 0.25
    # width of stimulus profile in units of dsites
    erw = 0.25

    # loop over seeds, mean weights and d_values
    for seed in seed_array:
        for mw in initialMeanWeights:
            for d in d_values:
                for fCR in fCR_values:
                    for nPulse in nPulse_values:
                        for CRsequence in CRsequence_values:

                            # set output directory
                            # the full output path will be:
                            outputDirectory = outputPath + '/seed_'+str(seed)+'_d_'+str(d)+'_mw_init_'+str(mw)
                            
                            # set path to initial network from which simulation is started
                            backupDirectory = outputToInitialNetworks+ "/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_"+str(mw)+"/5000_sec"
                            
                            ### input parameters for get_initial_networks_d.py
                            # sys.argv[1] ... backup directory (from which simulation is started)
                            # sys.argv[2] ... stimulation amplitude in units of Astim
                            # sys.argv[3] ... CR frequency in Hz (time during which each site receives one stimulus (single pulse or burst))
                            # sys.argv[4] ... number of pulses per burst stimulus
                            # sys.argv[5] ... total stimulation time in seconds
                            # sys.argv[6] ... number of stimulation sites
                            # sys.argv[7] ... string specifying CR sequence, e.g., 0_1_2_3 to activity sites in order 0-1-2-3
                            # sys.argv[8] ... output directory
                            # sys.argv[9] ... intraburst spike frequency, fintra, in Hz ( pulse duration is 1/fintra ), should not exceed 150 ms as otherwise parts of pulses are cut off, see functions_stim.py
                            # sys.argv[10]... distance between stimulation sites in units of L=5.0 mm, use 0.25 for four equally spaced sites (at -3/8 L, -1/8 L, 1/8 L, 3/8 L )
                            # sys.argv[11]... width of stimulation profile in units of dsites, use 0.5
                            shellCommand = 'python '+str(pathToSimScript)+'CR_stimulation_spatial_stimulus_profile.py '+ backupDirectory + " " + str(Astim) + " " + str(fCR) + " " + str(nPulse) + " " + str(Tstim) + " " + str(nSites) + " " + CRsequence + " " + outputDirectory + " " + str(fintra) + " " + str(dsites) + " " + str(erw) + '\n' 
                            submissionString = 'srun ' + shellCommand
                        

                            # python command for running simulations is ...
                            print( shellCommand ) 


                            # adds job to end of list
                            with open(pathForSubmissionStringTextFile, "a") as myfile:
                                                            
                                myfile.write( submissionString )
                                myfile.write( '##\n' )

# 2) shuffled CR
def run_simulations_shuffled_CR_stimulation_CompareToTheory( pathToSimScript, outputToInitialNetworks, outputPath, pathForSubmissionStringTextFile ):

    # seed values for network and noise generation
    seed_array = [10]
    # initial mean synaptic weights for which simulations are performed
    initialMeanWeights = [0.45]
    # systems length scale
    L = 5.0 # mm
    # length scale of synaptic connections
    d_values = L*np.array([0.08, 0.4, 2.0]) # in units of L
    # stimulation amplitude
    Astim = 1.0
    # CR frequencies
    fCR_values = np.arange(4.0,21.1,1.0) # Hz
    # number of pulses
    nPulse_values = [1,3]
    # stimulation time
    Tstim = 7220 # sec
    # number of stimulation sites
    nSites = 4
    # shuffled CR with a shuffle Period of 1/fCR is simlated for the following 
    # seeds for random sequence generation
    seedSequence_values = [100]

    # intra burst frequency (this corresponds to the inverse time between two subsequent pulses within a single stimulus burst)
    # this should not exceed about 130 Hz as individual pulses might get cut off
    # for single pulse stimuli (nPulse=1) this can just be set to 130 Hz
    fintra = 130.0 # Hz
    # distance between stimulation sites in units of system length scale (L=5 cm)
    dsites = 0.25
    # width of stimulus profile in units of dsites
    erw = 0.25

    # loop over seeds, mean weights and d_values
    for seed in seed_array:
        for mw in initialMeanWeights:
            for d in d_values:
                for fCR in fCR_values:
                    for nPulse in nPulse_values:
                        for seedSequence in seedSequence_values:

                            # set output directory
                            # the full output path will be:
                            outputDirectory = outputPath + '/seed_'+str(seed)+'_d_'+str(d)+'_mw_init_'+str(mw)
                            
                            # set path to initial network from which simulation is started
                            backupDirectory = outputToInitialNetworks+ "/seed_"+str(seed)+"_d_"+str(d)+"_mw_init_"+str(mw)+"/5000_sec"
                            
                            ### input parameters for get_initial_networks_d.py
                            # sys.argv[1] ... backup directory (from which simulation is started)
                            # sys.argv[2] ... stimulation amplitude in units of Astim
                            # sys.argv[3] ... CR frequency in Hz (time during which each site receives one stimulus (single pulse or burst))
                            # sys.argv[4] ... number of pulses per burst stimulus
                            # sys.argv[5] ... stimulation time in seconds
                            # sys.argv[6] ... number of stimulation sites
                            # sys.argv[7] ... seed for random sequence generation
                            # sys.argv[8] ... output directory
                            # sys.argv[9] ... intraburst spike frequency, fintra, in Hz ( pulse duration is 1/fintra ), should not exceed 150 ms as otherwise parts of pulses are cut off, see functions_stim.py
                            # sys.argv[10]... shuffle period
                            Tshuffle = np.round( 1.0/fCR , 4) # seconds
                            # sys.argv[11]... distance between stimulation sites in units of L=5.0 mm, use 0.25 for four equally spaced sites (at -3/8 L, -1/8 L, 1/8 L, 3/8 L )
                            # sys.argv[12]... width of stimulation profile in units of dsites, use 0.5
                            shellCommand = 'python '+str(pathToSimScript)+'shuffled_CR_stimulation_spatial_stimulus_profile.py '+ backupDirectory + " " + str(Astim) + " " + str(fCR) + " " + str(nPulse) + " " + str(Tstim) + " " + str(nSites) + " " + str(seedSequence) + " " + outputDirectory + " " + str(fintra) + " " + str(Tshuffle) + " " + str(dsites) + " " + str(erw) + '\n' 
                            submissionString = 'srun ' + shellCommand
                        

                            # python command for running simulations is ...
                            print( shellCommand ) 


                            # adds job to end of list
                            with open(pathForSubmissionStringTextFile, "a") as myfile:
                                                            
                                myfile.write( submissionString )
                                myfile.write( '##\n' )



# specifies directory in which the results are saved
pathToData = 'Frontiers_SSTPMDBN'



if __name__ == "__main__":

    ############################################################
    ############  get initial network configurations
    ############################################################
    # results from these simulations were used in ...
    # Figures 1 and 2
    if sys.argv[1] == 'get_initial_networks':

        # path to simulation script
        ## pathToSimScript ... path to the simulation script "get_initial_networks_d.py"
        pathToSimScript           = '/home/users/jkromer/Project1/Frontiers_SSTPMDBN/gitRepositories/initial_states/'
        ## outputPath ... directory in which simulation results will be stored (is generated if it doesn't already exist) 
        outputPath        = pathToData + '/initial_networks'
        ## pathForSubmissionStringTextFile ... in this file name all the submission strings will be stored (used to submit jobs to computation cluster)
        pathForSubmissionStringTextFile = '/home/users/jkromer/Project1/jobLists/Frontiers_SSTPMDBN_initial_networks.txt'

        run_simulations_initial_networks( pathToSimScript, outputPath, pathForSubmissionStringTextFile )

    # continue initial network simulations
    if sys.argv[1] == 'cont_get_initial_networks':

        # path to simulation script
        ## pathToSimScript ... path to the simulation script "get_initial_networks_d.py"
        pathToSimScript           = '/home/users/jkromer/Project1/Frontiers_SSTPMDBN/gitRepositories/initial_states/'

        pathToContinueScript = "/home/users/jkromer/Project1/Frontiers_SSTPMDBN/gitRepositories/Frontiers_SSTPMDBN/relaxation/"
        ## outputPath ... directory in which simulation results will be stored (is generated if it doesn't already exist) 
        outputPath        = pathToData + '/initial_networks'
        ## pathForSubmissionStringTextFile ... in this file name all the submission strings will be stored (used to submit jobs to computation cluster)
        pathForSubmissionStringTextFile = '/home/users/jkromer/Project1/jobLists/Frontiers_SSTPMDBN_initial_networks.txt'

        run_or_cont_simulations_initial_networks( pathToSimScript, pathToContinueScript, outputPath, pathForSubmissionStringTextFile )


    ############################################################
    ############  get results for Figures 4, 5, 6
    ############################################################
    # 1a) non-shuffled CR
    if sys.argv[1] == 'get_results_for_CR_stimulation':
 
        # path to simulation script
        ## pathToSimScript ... path to the simulation script "CR_stimulation_spatial_stimulus_profile.py"
        pathToSimScript           = "/home/users/jkromer/Project1/Frontiers_SSTPMDBN/gitRepositories/Frontiers_SSTPMDBN/CR_stimulation/"
        ## outputToInitialNetworks ... this is the path to the intial networks, from which simulations on CR stimulation are started
        outputToInitialNetworks = pathToData + '/initial_networks'
        ## outputPath ... directory in which simulation results will be stored (is generated if it doesn't already exist) 
        outputPath        = pathToData + '/CR_stimulation'
        ## pathForSubmissionStringTextFile ... in this file name all the submission strings will be stored (used to submit jobs to computation cluster)
        pathForSubmissionStringTextFile = '/home/users/jkromer/Project1/jobLists/Frontiers_SSTPMDBN_CR_stimulation.txt'

        run_simulations_CR_stimulation_Results_1( pathToSimScript, outputToInitialNetworks, outputPath, pathForSubmissionStringTextFile )

    # 2a) shuffled CR
    if sys.argv[1] == 'get_results_for_shuffled_CR_stimulation':

        
        # path to simulation script
        ## pathToSimScript ... path to the simulation script "CR_stimulation_spatial_stimulus_profile.py"
        pathToSimScript           = "/home/users/jkromer/Project1/Frontiers_SSTPMDBN/gitRepositories/Frontiers_SSTPMDBN/CR_stimulation/"
        ## outputToInitialNetworks ... this is the path to the intial networks, from which simulations on CR stimulation are started
        outputToInitialNetworks = pathToData + '/initial_networks'
        ## outputPath ... directory in which simulation results will be stored (is generated if it doesn't already exist) 
        outputPath        = pathToData + '/shuffled_CR_stimulation'
        ## pathForSubmissionStringTextFile ... in this file name all the submission strings will be stored (used to submit jobs to computation cluster)
        pathForSubmissionStringTextFile = '/home/users/jkromer/Project1/jobLists/Frontiers_SSTPMDBN_shuffled_CR_stimulation.txt'


        run_simulations_shuffled_CR_stimulation_Results_1( pathToSimScript, outputToInitialNetworks, outputPath, pathForSubmissionStringTextFile )
  
    # # 1b) relaxation after non-shuffled CR
    # if sys.argv[1] == 'get_results_for_relaxation_after_CR_stimulation':

        
    #     # path to simulation script
    #     ## pathToSimScript ... path to the simulation script "relaxation.py"
    #     pathToSimScript           = "/home/users/jkromer/Project1/Frontiers_SSTPMDBN/gitRepositories/Frontiers_SSTPMDBN/relaxation/"
    #     ## outputToInitialNetworks ... this is the path to the backups after CR stimulation from which simulations for relaxation are started
    #     outputToInitialNetworks = pathToData + '/CR_stimulation'
    #     ## outputPath ... directory in which simulation results will be stored (is generated if it doesn't already exist) 
    #     outputPath        = pathToData + '/relaxation_after_CR_stimulation'
    #     ## pathForSubmissionStringTextFile ... in this file name all the submission strings will be stored (used to submit jobs to computation cluster)
    #     pathForSubmissionStringTextFile = '/home/users/jkromer/Project1/jobLists/Frontiers_SSTPMDBN_relaxation_after_CR_stimulation.txt'

    #     run_simulations_CR_stimulation_Results_1_relaxation( pathToSimScript, outputToInitialNetworks, outputPath, pathForSubmissionStringTextFile )

    # # 2b) relaxation after shuffled CR
    # if sys.argv[1] == 'get_results_for_relaxation_after_shuffled_CR_stimulation':

        
    #     # path to simulation script
    #     ## pathToSimScript ... path to the simulation script "relaxation.py"
    #     pathToSimScript           = "/home/users/jkromer/Project1/Frontiers_SSTPMDBN/gitRepositories/Frontiers_SSTPMDBN/relaxation/"
    #     ## outputToInitialNetworks ... this is the path to the backups after CR stimulation from which simulations for relaxation are started
    #     outputToInitialNetworks = pathToData + '/shuffled_CR_stimulation'
    #     ## outputPath ... directory in which simulation results will be stored (is generated if it doesn't already exist) 
    #     outputPath              = pathToData + '/relaxation_after_shuffled_CR_stimulation'
    #     ## pathForSubmissionStringTextFile ... in this file name all the submission strings will be stored (used to submit jobs to computation cluster)
    #     pathForSubmissionStringTextFile = '/home/users/jkromer/Project1/jobLists/Frontiers_SSTPMDBN_relaxation_after_shuffled_CR_stimulation.txt'

    #     run_simulations_shuffled_CR_stimulation_Results_1_relaxation( pathToSimScript, outputToInitialNetworks, outputPath, pathForSubmissionStringTextFile )

    # 1b) relaxation after non-shuffled CR
    if sys.argv[1] == 'cont_get_results_for_relaxation_after_CR_stimulation':

        
        # path to simulation script
        ## pathToSimScript ... path to the simulation script "relaxation.py"
        pathToSimScript           = "/home/users/jkromer/Project1/Frontiers_SSTPMDBN/gitRepositories/Frontiers_SSTPMDBN/relaxation/"
        ## outputToInitialNetworks ... this is the path to the backups after CR stimulation from which simulations for relaxation are started
        outputToInitialNetworks = pathToData + '/CR_stimulation'
        ## outputPath ... directory in which simulation results will be stored (is generated if it doesn't already exist) 
        outputPath        = pathToData + '/relaxation_after_CR_stimulation'
        ## pathForSubmissionStringTextFile ... in this file name all the submission strings will be stored (used to submit jobs to computation cluster)
        pathForSubmissionStringTextFile = '/home/users/jkromer/Project1/jobLists/Frontiers_SSTPMDBN_relaxation_after_CR_stimulation.txt'

        # run_simulations_CR_stimulation_Results_1( pathToSimScript, outputToInitialNetworks, outputPath, pathForSubmissionStringTextFile )

        run_or_cont_simulations_CR_stimulation_Results_1_relaxation( pathToSimScript, outputToInitialNetworks, outputPath, pathForSubmissionStringTextFile )

    # 2b) relaxation after shuffled CR
    if sys.argv[1] == 'cont_get_results_for_relaxation_after_shuffled_CR_stimulation':

        
        # path to simulation script
        ## pathToSimScript ... path to the simulation script "relaxation.py"
        pathToSimScript           = "/home/users/jkromer/Project1/Frontiers_SSTPMDBN/gitRepositories/Frontiers_SSTPMDBN/relaxation/"
        ## outputToInitialNetworks ... this is the path to the backups after CR stimulation from which simulations for relaxation are started
        outputToInitialNetworks = pathToData + '/shuffled_CR_stimulation'
        ## outputPath ... directory in which simulation results will be stored (is generated if it doesn't already exist) 
        outputPath              = pathToData + '/relaxation_after_shuffled_CR_stimulation'
        ## pathForSubmissionStringTextFile ... in this file name all the submission strings will be stored (used to submit jobs to computation cluster)
        pathForSubmissionStringTextFile = '/home/users/jkromer/Project1/jobLists/Frontiers_SSTPMDBN_relaxation_after_shuffled_CR_stimulation.txt'

        run_or_cont_simulations_shuffled_CR_stimulation_Results_1_relaxation( pathToSimScript, outputToInitialNetworks, outputPath, pathForSubmissionStringTextFile )

    
    ############################################################
    ############  get results for Figures 8 and 9
    ############################################################
    # d = 0.4 L
    if sys.argv[1] == 'get_results_for_get_J_phi_f':

        
        # path to simulation script
        ## pathToSimScript ... path to the simulation script "CR_stimulation_spatial_stimulus_profile.py"
        pathToSimScript           = "/home/users/jkromer/Project1/Frontiers_SSTPMDBN/gitRepositories/Frontiers_SSTPMDBN/get_J_phi_f/"
        ## inputToInitialNetworks ... this is the path to the intial networks, from which simulations on CR stimulation are started
        inputToInitialNetworks = pathToData + '/initial_networks'
        ## outputPath ... directory in which simulation results will be stored (is generated if it doesn't already exist) 
        outputPath        = pathToData + '/get_J_phi_f'
        ## pathForSubmissionStringTextFile ... in this file name all the submission strings will be stored (used to submit jobs to computation cluster)
        pathForSubmissionStringTextFile = '/home/users/jkromer/Project1/jobLists/Frontiers_SSTPMDBN_get_J_phi_f.txt'

        run_simulations_get_J_phi_f( pathToSimScript, inputToInitialNetworks, outputPath, pathForSubmissionStringTextFile )

    # d = 0.08 L and 2 L
    if sys.argv[1] == 'get_results_for_get_J_phi_f_ForComparissonToSimulations':

        
        # path to simulation script
        ## pathToSimScript ... path to the simulation script "CR_stimulation_spatial_stimulus_profile.py"
        pathToSimScript           = "/home/users/jkromer/Project1/Frontiers_SSTPMDBN/gitRepositories/Frontiers_SSTPMDBN/get_J_phi_f/"
        ## inputToInitialNetworks ... this is the path to the intial networks, from which simulations on CR stimulation are started
        inputToInitialNetworks = pathToData + '/initial_networks'
        ## outputPath ... directory in which simulation results will be stored (is generated if it doesn't already exist) 
        outputPath        = pathToData + '/get_J_phi_f'
        ## pathForSubmissionStringTextFile ... in this file name all the submission strings will be stored (used to submit jobs to computation cluster)
        pathForSubmissionStringTextFile = '/home/users/jkromer/Project1/jobLists/Frontiers_SSTPMDBN_get_J_phi_f_comparisson.txt'

        run_simulations_get_J_phi_f_ForComparissonToSimulations( pathToSimScript, inputToInitialNetworks, outputPath, pathForSubmissionStringTextFile )


    ############################################################
    ############  get results for Figure 9
    ############################################################
    # 1) non-shuffled CR
    if sys.argv[1] == 'get_results_for_CR_stimulation_CompareToTheory':

        
        # path to simulation script
        ## pathToSimScript ... path to the simulation script "CR_stimulation_spatial_stimulus_profile.py"
        pathToSimScript           = "/home/users/jkromer/Project1/Frontiers_SSTPMDBN/gitRepositories/Frontiers_SSTPMDBN/CR_stimulation/"
        ## outputToInitialNetworks ... this is the path to the intial networks, from which simulations on CR stimulation are started
        outputToInitialNetworks = pathToData + '/initial_networks'
        ## outputPath ... directory in which simulation results will be stored (is generated if it doesn't already exist) 
        outputPath        = pathToData + '/CR_stimulation_long'
        ## pathForSubmissionStringTextFile ... in this file name all the submission strings will be stored (used to submit jobs to computation cluster)
        pathForSubmissionStringTextFile = '/home/users/jkromer/Project1/jobLists/Frontiers_SSTPMDBN_CR_stimulation_long.txt'

        run_simulations_CR_stimulation_CompareToTheory( pathToSimScript, outputToInitialNetworks, outputPath, pathForSubmissionStringTextFile )

    # 2) shuffled CR
    if sys.argv[1] == 'get_results_for_shuffled_CR_stimulation_CompareToTheory':

        
        # path to simulation script
        ## pathToSimScript ... path to the simulation script "CR_stimulation_spatial_stimulus_profile.py"
        pathToSimScript           = "/home/users/jkromer/Project1/Frontiers_SSTPMDBN/gitRepositories/Frontiers_SSTPMDBN/CR_stimulation/"
        ## outputToInitialNetworks ... this is the path to the intial networks, from which simulations on CR stimulation are started
        outputToInitialNetworks = pathToData + '/initial_networks'
        ## outputPath ... directory in which simulation results will be stored (is generated if it doesn't already exist) 
        outputPath        = pathToData + '/shuffled_CR_stimulation_long'
        ## pathForSubmissionStringTextFile ... in this file name all the submission strings will be stored (used to submit jobs to computation cluster)
        pathForSubmissionStringTextFile = '/home/users/jkromer/Project1/jobLists/Frontiers_SSTPMDBN_shuffled_CR_stimulation_long.txt'


        run_simulations_shuffled_CR_stimulation_CompareToTheory( pathToSimScript, outputToInitialNetworks, outputPath, pathForSubmissionStringTextFile )


