__author__ = "Amin Aghaee"
__copyright__ = "Copyright 2018, Amin Aghaee"

#from Prob2Line import *
#pmap = np.load('./Results/NewTrainingRandom_strip_mixed/Pmamp_Fault_Australia.hdf5Australia_strip.mat.npz')
#pmap = pmap['matrix']
#p2l = prob2map(pmap)
#p2l.runMethod(coeff=0.66, eps = 3, iteration=350)

from DATASET import *
import scipy.io as sio

# --------------------------------------
testList = ['Australia_strip.mat', 'QUEST_strip.mat']
for T in testList:
    ds_fname = DSDIR + T
    ds = DATASET(ds_fname)

    Z = np.zeros_like(ds.OUTPUT)
    O = np.ones_like(ds.OUTPUT)
    R = np.random.random(ds.OUTPUT.shape)

    z = {}
    o = {}
    r = {}

    pmap_list = [Z,R,O]
    output_list = [z,r,o]
    for i in range(3):
        m = output_list[i]

        m['Train_p'] = []
        m['Train_n'] = []
        m['Test_p'] = []
        m['Test_n'] = []
        m['All_p'] = []
        m['All_n'] = []

        pmap = pmap_list[i]

        for w in range(9, 57, 4):
            print("Teste: {} ----- W = {}".format(T, w))

            [pos, neg] = ds.evaluate(pmap, w, 'train', etype='our')
            m['Train_p'] += [pos]
            m['Train_n'] += [neg]

            [pos, neg] = ds.evaluate(pmap, w, 'test', etype='our')
            m['Test_p'] += [pos]
            m['Test_n'] += [neg]


            [pos, neg] = ds.evaluate(pmap, w, 'all', etype='our')
            m['All_p'] += [pos]
            m['All_n'] += [neg]


    sio.savemat(T[0:5]+'_extreme.mat' , [z,r,o] )


pmapname = '45_Pmamp_45_Fault_Quest.hdf5_on_QUEST_.npz'
outputname = 'Quest_on_Quest'

pmap = np.load(pmapname)['matrix']

png = np.uint8(pmap * 255)
im = Image.fromarray(png)
im.save(outputname + '.png')

im = Image.fromarray(pmap)
im.save(outputname + '.tiff')
