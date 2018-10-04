from Prob2Line import *
pmap = np.load('./Results/NewTrainingRandom_strip_mixed/Pmamp_Fault_Australia.hdf5Australia_strip.mat.npz')
pmap = pmap['matrix']
p2l = prob2map(pmap)
p2l.runMethod(coeff=0.66, eps = 3, iteration=350)














