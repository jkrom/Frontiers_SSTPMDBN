import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt


# Approximate number of synapses between subpopulations x and y for 
# given ratio between distances between adjacent sites and synaptic length scale. This 
# corresponds to the soluation of Equation 10. 
def B(x, y, l):
    
    if x==y:
        return 2*np.exp(-l)/(l*l)*(1-np.exp(l)+l*np.exp(l))
    else:
        return np.exp(-l*(np.abs(y-x)+1))*(np.exp(l)-1)*(np.exp(l)-1)/(l*l)
    
# normalization of number of synaptic connections
def Normalization(l,M):
    
    NormB = 0
    for x in range(M):
        for y in range(M):
            NormB += B(x, y, l)
            
    return NormB


# The following functions evalutes the numbers of synaptic connections between
# individual subpopulations in simulation data and calculates the approximate number of connections for Figure 7
def calc_approx_b_and_get_simulations( ouputDirectory, pathToSimulationData_initial):

    # number of stimulation sites
    M = 4
    seed_array = [10,12,14,16,18]
    # distance between adjacent sites
    ds = 1.25 # mm

    # loop over synaptic length scales
    for d in [0.4, 2.0, 10.0]: # mm

        dic_adj = {}

        # l is the ratio between the distance between adjacent sites 
        # and the synaptic length scale.
        l = ds/d

        # load adjacency matrices from simulation output
        for seed in seed_array:

            # path to synConnections.npz for initial network realizations
            directory = pathToSimulationData_initial + '/seed_'+str(seed)+'_d_'+str(d)+'_mw_init_0.0/0_sec'

            filename = directory + '/synConnections.npz'

            adj = scipy.sparse.load_npz( filename )

            dic_adj[seed] = {'adj': adj, 'Ntotal':np.sum( adj ), 'centerCoordinates':np.load( directory + '/STNCenter.npy' ) }
            dic_adj[seed]['Dxy'] = []

        # calculate distances between presynaptic and postsynaptic neurons
        for i in range(1000):
            for j in range(1000):

                for seed in seed_array:
                    if dic_adj[seed]['adj'][i,j] == 1:
                        dic_adj[seed]['Dxy'].append([ i, j, dic_adj[seed]['centerCoordinates'][i], dic_adj[seed]['centerCoordinates'][j], np.abs( dic_adj[seed]['centerCoordinates'][i] - dic_adj[seed]['centerCoordinates'][j]) ] )

        for seed in seed_array:
            dic_adj[seed]['Dxy'] = np.array( dic_adj[seed]['Dxy'] )

        # calculate relative number of connections for all M x M pairs of subpopulations
        counter = 0
        theoryData = []
        simulationData = []
        X0 = -2.5 # mm

        for x in range(M):
            for y in range(M):

                theoryData.append([ counter , B(x, y, l)/Normalization(l,M) , '$'+str(x)+str(y)+'$' ])

                Nxy_current = []
                for seed in seed_array:
                    ind_in_current_xPopylation = np.logical_and( dic_adj[seed]['Dxy'][:,2] >= X0 + x*ds , dic_adj[seed]['Dxy'][:,2] < X0 + (x+1)*ds  )
                    ind_in_current_yPopylation = np.logical_and( dic_adj[seed]['Dxy'][:,3] >= X0 + y*ds , dic_adj[seed]['Dxy'][:,3] < X0 + (y+1)*ds  ) 
                    Nxy_current.append( len( dic_adj[seed]['Dxy'][ np.logical_and( ind_in_current_xPopylation , ind_in_current_yPopylation ) ] )/float( dic_adj[seed]['Ntotal'] ) )

                current_list = [  counter , np.mean(Nxy_current), np.std(Nxy_current)  ]
                for list_element in Nxy_current:
                    current_list.append( list_element )
                simulationData.append( current_list )

                counter+=1

        theoryData = np.array(theoryData)
        simulationData = np.array( simulationData )

        # save theoretical approximations and simulation results
        np.save( ouputDirectory+'/theoryData_d_'+str(d)+'.npy' , theoryData  )
        np.save( ouputDirectory+'/simulationData_d_'+str(d)+'_individualData.npy' , simulationData  )


# relative number of connections between subpopulations x and y 
def b(x,y,l,M):
    return B(x, y, l)/Normalization(l,M)

def Deltaw(d,fCR,phi,npb, Jvalues):
    
    ind_d_fCR = np.logical_and( Jvalues[:,0]==d, Jvalues[:,1]==fCR )
    ind_phi_npb = np.logical_and( Jvalues[:,2]==phi, Jvalues[:,3]==npb )
    # print( np.round( result_array,2))
    # [d,fCR,phi,npb,np.mean(current_DeltaWList), np.std(current_DeltaWList), len(current_DeltaWList)]
    return Jvalues[ np.logical_and( ind_d_fCR , ind_phi_npb) ][0,4], Jvalues[ np.logical_and( ind_d_fCR , ind_phi_npb) ][0,7]

# calculates the approximate mean synaptic weight
def mw_for_seq_npb_fCR_d(seq, npb, fCR, d, M, T, Jvalues, dic_seq_phase_lags):
    
    TTestStim = 20 # sec

    # distance between adjacent stimulation sites
    ds = 1.25 # mm
    l = ds/d

    # initialize mean weights
    meanWeight = 0
    meanWeight_intra = 0
    meanWeight_inter = 0
    # .. and numbers of connections
    relNumber_intra = 0
    relNumber_inter = 0

    # initialize matrices
    estMmatrix = np.zeros( (M,M) )
    conDiagram = np.zeros( (M,M) )
    synDiagram = np.zeros( (M,M) )
    
    for xpre in range(M):
        for ypost in range(M):
            # get phase lag
            phi = dic_seq_phase_lags[ seq ][xpre,ypost]
            # get weight change in TTestStim and weight prior to stimulation
            Sest, wT0 = Deltaw(d,fCR,phi,npb, Jvalues)
            # get estimated rate of weight change
            EstRateOfWeightChange = Sest/TTestStim
            # estimate weight after given time T
            wT = np.clip( wT0 + EstRateOfWeightChange*T , a_min=0.0, a_max=1.0 )
            # save estimated weight after time T in "estMmatrix"
            estMmatrix[xpre,ypost]=wT
            # update estimated of mean synaptic weight
            if ( xpre != ypost ):
                conDiagram[xpre,ypost]=b(xpre,ypost,l,M)*wT
                meanWeight+=b(xpre,ypost,l,M)*wT
                meanWeight_inter+=b(xpre,ypost,l,M)*wT
                relNumber_inter+=b(xpre,ypost,l,M)
            else:
                if npb == 3:
                    conDiagram[xpre,ypost]=b(xpre,ypost,l,M)*wT
                    meanWeight+=b(xpre,ypost,l,M)*wT
                else:
                    conDiagram[xpre,ypost]=b(xpre,ypost,l,M)*wT
                    meanWeight+=b(xpre,ypost,l,M)*wT
                meanWeight_intra+=b(xpre,ypost,l,M)*wT
                relNumber_intra+=b(xpre,ypost,l,M)
            # print(xpre,ypost,wxy,Sest)
            synDiagram[xpre,ypost]=b(xpre,ypost,l,M)
            
    return meanWeight, estMmatrix, conDiagram, synDiagram, meanWeight_intra/relNumber_intra, meanWeight_inter/relNumber_inter

# calculates the approximations of the mean synaptic weights of intra- and inter-population synapses
# after an evaluation time "Teval" for a given estimated of the rate of weight change "Jvalues"
def get_approximation( Jvalues, Teval ):
    
    # phase lags for different stimulation sequences]
    # dic_seq_phase_lags["I_II_III_IV"][xpre,ypost] is the phase lag between
    # stimuli delivered to subpopulations xpre and ypost
    dic_seq_phase_lags = {}
    # specify phase lags between subpopulations for different CR sequences
    dic_seq_phase_lags["I_II_III_IV"]=np.array([[0.0, 0.25, 0.5, 0.75],[0.75,0.0,0.25,0.5],[ 0.5, 0.75,0.0,0.25],[0.25,0.5, 0.75,0.0]])
    dic_seq_phase_lags["I_II_IV_III"]=np.array([[0.0, 0.25, 0.75,0.5], [0.75,0.0,0.5, 0.25],[0.25,0.5, 0.0,0.75],[0.5, 0.75,0.25,0.0]])
    dic_seq_phase_lags["I_III_II_IV"]=np.array([[0.0, 0.5,  0.25,0.75],[0.5, 0.0,0.75,0.25],[0.75,0.25,0.0,0.5], [0.25,0.75,0.5, 0.0]])
    dic_seq_phase_lags["I_III_IV_II"]=np.array([[0.0,0.75,0.25,0.5],[0.25,0.0,0.5,0.75],[0.75,0.5,0.0,0.25],[0.5,0.25,0.75,0.0]])
    dic_seq_phase_lags["I_IV_II_III"]=np.array([[0.0,0.5,0.75,0.25],[0.5,0.0,0.25,0.75],[0.25,0.75,0.0,0.5],[0.75,0.25,0.5,0.0]])
    dic_seq_phase_lags["I_IV_III_II"]=np.array([[0.0,0.75,0.5,0.25],[0.25,0.0,0.75,0.5],[0.5,0.25,0.0,0.75],[0.75,0.5,0.25,0.0]])
    
    seq_array = ["I_II_III_IV","I_II_IV_III","I_III_II_IV","I_III_IV_II","I_IV_II_III","I_IV_III_II"]
    dic_plt_results = {}

    # loop over numbers of pulses per stimulus
    for npb in [1,3]:
        dic_plt_results[npb] = {}
        # loop over synaptic length scales
        for d in [0.4, 2.0, 10.0]:
            dic_plt_results[npb][d] = {}
            M = 4

            results = []
            for kseq in range(len(seq_array)):
                seq = seq_array[kseq]
                # run over stimulation frequencies
                for fCR in np.arange(4.0,21.1,1.0): # Hz
                    # print(npb, d, kseq, fCR)
                    # try:
                    mw, CMW, conDiagram, synDiagram, meanWeight_intra, meanWeight_inter = mw_for_seq_npb_fCR_d(seq, npb, fCR, d, M, Teval-5000, Jvalues, dic_seq_phase_lags )
                    results.append( [fCR, kseq, mw, meanWeight_intra, meanWeight_inter])
                    # except:
                    #     continue

            dic_plt_results[npb][d] = np.array(results)
            
    return dic_plt_results

# the following function generates Figure 7
#  PathToDataNumbersOfConnections points to directory in which data on numbers of connections are saved
#  default value is "data/data_numbers_of_connections"/
def generateFigureRelativeNumbersOfConnections( pathToDataNumbersOfConnections ):
    
    M = 4
    label_fontsize = 20
    ticks_fontsize = 13

    seed_array = [10,12,14,16,18]

    fig = plt.figure( figsize = (12,2) )

    ax1 = fig.add_subplot(131)

    theoryData = np.load( pathToDataNumbersOfConnections + '/theoryData_d_0.4.npy' )
    # simulationData = np.load( pathToDataNumbersOfConnections + '/simulationData_d_0.4.npy' )
    simulationData = np.load( pathToDataNumbersOfConnections + '/simulationData_d_0.4_individualData.npy' )

    # print(simulationData)

    for kdataPoint in range(M*M):
        ax1.plot( [ theoryData[kdataPoint,0].astype(float)-0.5 , theoryData[kdataPoint,0].astype(float)+0.5 ], [ theoryData[kdataPoint,1].astype(float) , theoryData[kdataPoint,1].astype(float) ] , ls = '-', color = 'black', zorder = 1, lw=2 )
        #ax1.scatter( simulationData[kdataPoint,0].astype(float) ,simulationData[kdataPoint,1].astype(float), marker = 'o', edgecolor = 'red', facecolor='white'  )
        # ax1.errorbar( simulationData[kdataPoint,0].astype(float)+0.1 , simulationData[kdataPoint,1].astype(float) , yerr = simulationData[kdataPoint,2].astype(float), ecolor='red', zorder =0, lw=5 )
        # ax1.bar( simulationData[kdataPoint,0].astype(float) , simulationData[kdataPoint,1].astype(float) , yerr = simulationData[kdataPoint,2].astype(float), color='0.8', ecolor='black', zorder =0, lw=5 )

        for kseed in range( len( seed_array ) ):
            seed = seed_array[ kseed ]
            # print( kdataPoint ,  2+kseed, len(simulationData[kdataPoint]), len(simulationData) )
            ax1.scatter( simulationData[kdataPoint,0].astype(float),  simulationData[kdataPoint,3+kseed].astype(float), color="gray", marker="x"  )

    ax2 = fig.add_subplot(132)

    theoryData = np.load( pathToDataNumbersOfConnections + '/theoryData_d_2.0.npy' )
    # simulationData = np.load( pathToDataNumbersOfConnections + '/simulationData_d_2.0.npy' )
    simulationData = np.load( pathToDataNumbersOfConnections + '/simulationData_d_2.0_individualData.npy' )

    for kdataPoint in range(M*M):
        ax2.plot( [ theoryData[kdataPoint,0].astype(float)-0.5, theoryData[kdataPoint,0].astype(float)+0.5 ], [ theoryData[kdataPoint,1].astype(float),theoryData[kdataPoint,1].astype(float)] , ls = '-', color = 'black', zorder = 1, lw=2 )
        #ax2.scatter( simulationData[kdataPoint,0].astype(float) ,simulationData[kdataPoint,1].astype(float), marker = 'o', edgecolor = 'red', facecolor='white'  )
        # ax2.errorbar( simulationData[kdataPoint,0].astype(float)+0.1 , simulationData[kdataPoint,1].astype(float) , yerr = simulationData[kdataPoint,2].astype(float), color='red', zorder =0, lw=5 )
        # ax2.bar( simulationData[kdataPoint,0].astype(float) , simulationData[kdataPoint,1].astype(float) , yerr = simulationData[kdataPoint,2].astype(float), color='0.8', ecolor='black', zorder =0, lw=5 )
        for kseed in range( len( seed_array ) ):
            seed = seed_array[ kseed ]
            ax2.scatter( simulationData[kdataPoint,0].astype(float),  simulationData[kdataPoint,3+kseed].astype(float), color="gray", marker="x"  )

    ax3 = fig.add_subplot(133)

    theoryData = np.load( pathToDataNumbersOfConnections + '/theoryData_d_10.0.npy' )
    # simulationData = np.load( pathToDataNumbersOfConnections + '/simulationData_d_10.0.npy' )
    simulationData = np.load( pathToDataNumbersOfConnections + '/simulationData_d_10.0_individualData.npy' )

    for kdataPoint in range(M*M):
        # ax3.scatter( theoryData[kdataPoint,0].astype(float)-0.2     ,theoryData[kdataPoint,1].astype(float)    , marker = 'x', color = 'red', s=40, zorder = 1 )
        ax3.plot( [ theoryData[kdataPoint,0].astype(float)-0.5, theoryData[kdataPoint,0].astype(float)+0.5 ]  , [ theoryData[kdataPoint,1].astype(float) , theoryData[kdataPoint,1].astype(float) ]   , ls = '-', color = 'black', zorder = 1, lw=2 )
        #ax3.scatter( simulationData[kdataPoint,0].astype(float) ,simulationData[kdataPoint,1].astype(float), marker = 'o', edgecolor = 'red', facecolor='white'  )
        # ax3.errorbar( simulationData[kdataPoint,0].astype(float)+0.1 , simulationData[kdataPoint,1].astype(float) , yerr = simulationData[kdataPoint,2].astype(float), color='red', zorder =0, lw=5 )
        # ax3.bar( simulationData[kdataPoint,0].astype(float) , simulationData[kdataPoint,1].astype(float) , yerr = simulationData[kdataPoint,2].astype(float), color='0.8', ecolor='black', zorder =0, lw=5 )
        for kseed in range( len( seed_array ) ):
            seed = seed_array[ kseed ]
            ax3.scatter( simulationData[kdataPoint,0].astype(float),  simulationData[kdataPoint,3+kseed].astype(float), color="gray", marker="x"  )


    for ax in [ax1,ax2,ax3]:
        #plt.legend()
        ax.set_xticks( theoryData[:,0].astype(float) )
        ax.set_xticklabels( ['' for x in theoryData[:,2]], fontsize = ticks_fontsize )
        ax.set_yticks([0,0.05,0.1,0.15,0.2,0.25,0.3])
        ax.set_yticklabels(['','','','','','',''], fontsize = ticks_fontsize)

        ax.set_xticklabels(['I-I','I-II','I-III','I-IV','II-I','II-II','II-III','II-IV','III-I','III-II','III-III','III-IV','IV-I','IV-II','IV-III','IV-IV'], rotation ='vertical')

        ax.set_ylim(0,0.25)
        ax.set_xlim(-0.5,15.5)

        ax.set_xlabel('xy', fontsize = label_fontsize)

    ax1.set_yticklabels(['$0$','','$0.1$','','$0.2$','','$0.3$'], fontsize = ticks_fontsize)

    ax1.set_ylabel('$b_{\mathrm{xy}}$', fontsize = label_fontsize)
    ax3.set_xticklabels( theoryData[:,2], fontsize = ticks_fontsize )
    # ax3.set_xticklabels( ['I\n I'], fontsize = ticks_fontsize )
    ax3.set_xticklabels(['I-I','I-II','I-III','I-IV','II-I','II-II','II-III','II-IV','III-I','III-II','III-III','III-IV','IV-I','IV-II','IV-III','IV-IV'], rotation ='vertical')
    # plt.xticks(x, ['I->I','I->II','I->III','I->IV','II->I','II->II','II->III','II->IV','III->I','III->II','III->III','III->IV','IV->I','IV->II','IV->III','IV->IV'], rotation ='vertical')

    # ax1.set_title(0.6,0.23,'$s = 0.08 L$', fontsize = ticks_fontsize )
    ax1.set_title('$s = 0.08 L$ $(0.32 d)$', fontsize = 1.2*ticks_fontsize )
    ax2.set_title('$s = 0.4 L$ $(1.6 d)$', fontsize = 1.2*ticks_fontsize )
    ax3.set_title('$s = 2 L$ $(8 d)$', fontsize = 1.2*ticks_fontsize )

    ax1.text(-3.5,0.25,'A', fontsize = 1.1*label_fontsize )
    ax2.text(-3.,0.25,'B', fontsize = 1.1*label_fontsize )
    ax3.text(-3.,0.25,'C', fontsize = 1.1*label_fontsize )

    
    
    return fig




