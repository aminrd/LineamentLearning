
__author__ = "Amin Aghaee"
__copyright__ = "Copyright 2018, Amin Aghaee"

import os
import sys
import numpy as np
import random
import argparse

# Loading Related Modules:
# --------------------------------------
from globalVariables import *
from Utility import *
from DATASET import *
#from MODEL import *
from FILTER import *
#from PmapViewer import *
from Logger import *
# --------------------------------------


def GET_PARSER():
    parser = argparse.ArgumentParser()
    parser.add_argument('work', default='test-choosy')
    parser.add_argument('-W', '--WSIZE', type=int, default=45)
    parser.add_argument('-it', '--iterations', type=int, default=ITERATIONS)
    parser.add_argument('-prefix', '--prepprefix', default='ANG_')
    parser.add_argument('-nprep', '--prefnumber', type=int, default=15)
    parser.add_argument('-CB', '--callback', default='FaultDetection.hdf5')
    return parser


def SET_DEFAULT_ARGUMENTS(args):
    print('#'*30)
    print('Setting up global variables:')
    print(args)

    global ITERATIONS
    ITERATIONS = args.iterations
    global WindowSize
    WindowSize = args.WSIZE
    print('#' * 30)



if __name__== "__main__":
    parser = GET_PARSER()
    args = parser.parse_args()
    work = args.work

    #SET_DEFAULT_ARGUMENTS(args)
    # ------------------ Training model only on faulty areas ------------------------------------------------------------
    if work.__eq__("train-choosy"):

        step = np.pi / NUMBER_OF_DEGREE_MODELS

        for d in range(NUMBER_OF_DEGREE_MODELS):

            # Working on degree:
            baseDegree = np.pi / 2.0 - step * d - 0.00001

            fname = CB + 'Rotate_choosy_{}.hdf5'.format(d)
            model = MODEL(checkpoint=fname)

            for it in range(ITERATIONS):
                numberOfFiles = 3 + (it*33)//ITERATIONS

                if DEBUG_MODE:
                    print("=" * 30)
                    print("Iteration number is : {}".format(it))
                    print("---- working of {} number of files".format(numberOfFiles))
                    print("---- working on degree: {} degrees".format(baseDegree * 180 / np.pi))
                    print("=" * 30)

                X = np.zeros((1, WindowSize, WindowSize, Layers))
                Y = np.zeros((1, 1))

                myRatio = 1 / (6*numberOfFiles)

                for i in random.sample(list(range(fileNumber)), numberOfFiles):

                    idx = np.int((i+1) * 10)
                    ds_fname = DSDIR + "Australia_{}.mat".format(idx)
                    ds = DATASET(ds_fname)

                    [Xb, Yb, IDXb] = ds.generateDS(ds.DEGREES, ds.trainMask, ratio=myRatio, choosy=True, output_type = baseDegree)

                    X = np.concatenate((X, Xb), axis=0)
                    Y = np.concatenate((Y, Yb), axis=0)

                model.train(X, Y, epochs=1)

    # ------------------ Testing model only on faulty areas ------------------------------------------------------------
    elif work.__eq__("test-choosy"):
        #testList = list(range(36))     # See Results on all different rotations
        testList = list([23])           # See Results only on main file (because 36 = 360 degrees rotation = main file)

        step = np.pi / NUMBER_OF_DEGREE_MODELS

        for i in testList:

            if DEBUG_MODE:
                print("Working on rotation number : {}".format(i+1))

            idx = np.int((i+1)*10)
            ds_fname = DSDIR + "Australia_{}.mat".format(idx)
            ds = DATASET(ds_fname)
            [X, Y, IDX] = ds.generateDS(ds.DEGREES, ds.MASK, ratio=0.99, choosy=True)

            MaxProb = -np.ones(len(Y))
            MaxSlope = np.full(len(Y), np.pi / 2.0)

            slopes = np.zeros(NUMBER_OF_DEGREE_MODELS)

            for d in range(NUMBER_OF_DEGREE_MODELS):

                slopes[d] = (np.pi / 2.0 - step * d)

                fname = CB + 'Rotate_choosy_{}.hdf5'.format(d)
                model = MODEL(param_dir=fname)
                baseDegree = (np.pi / 2.0 - step * d) * 180 // np.pi
                Yh = model.predict(X)

                empty_matrix = np.zeros((ds.x, ds.y, 3))
                O = np.array(empty_matrix)
                O[:, :, 0] = ds.OUTPUT * 255
                O[:, :, 1] = ds.OUTPUT * 255
                O[:, :, 2] = ds.OUTPUT * 255
                O = np.uint8(O)

                tmp = drawLines(O, IDX, Yh, WIDTH=3, FILL=128, ws=1, fname=FG + "Degree_{}_Line_{}_overlay.png".format(baseDegree, idx))
                empty_matrix = np.uint8(empty_matrix)
                tmp = drawLines(empty_matrix, IDX, Yh, WIDTH=3, FILL=128, ws=3, fname=FG + "Degree_{}_Line_{}_alone.png".format(baseDegree,idx))

                YhNormal = np.ndarray.flatten(Yh)
                mIdx = np.where(MaxProb < YhNormal)
                MaxProb[mIdx] = YhNormal[mIdx]
                MaxSlope[mIdx] = slopes[d]


            empty = np.uint8(np.zeros((ds.x, ds.y, 3)))
            tmp = drawLinesSlope(empty, IDX, MaxSlope, ws=6 ,fname=FG + 'Predictions_Alone_{}.png'.format(i + 1))

            for r in range(3):
                empty[:,:,r] = np.uint8(ds.OUTPUT)

            tmp = drawLinesSlope(empty, IDX, MaxSlope, ws=6, fname=FG + 'Predictions_Overlay_{}.png'.format(i + 1))


    # ------------------ Train Fault detection method on all area, Not break mask, instead Bootstrapping on all input images -----------------------------
    elif work.__eq__("train-fault-all"):

        windowList = [35,45]

        for W in windowList:
            fname = CB + 'FaultDetection_{}.hdf5'.format(W)
            model = MODEL(w = W, checkpoint=fname)


            for it in range(ITERATIONS):

                numberOfFiles = 3 + (it * (fileNumber - 3)) // ITERATIONS

                if DEBUG_MODE:
                    print("=" * 30)
                    print("Iteration number is : {}".format(it))
                    print("---- working of {} number of files".format(numberOfFiles))
                    print("=" * 30)

                X = np.zeros((1, W, W, Layers))
                Y = np.zeros((1, 1))

                myRatio = 1 / (9*numberOfFiles)

                for i in random.sample(list(range(fileNumber)), numberOfFiles):

                    idx = np.int((i+1) * 10)
                    ds_fname = DSDIR + "Australia_{}.mat".format(idx)
                    ds = DATASET(ds_fname)

                    t_mask_small = ds.shrinkMask('train')
                    ds.expandBy(width=35, epsilon=0.9)

                    [Xb, Yb, IDXb] = ds.generateDS(ds.OUTPUT, ds.trainMask, w = W, ratio=myRatio, output_type=0)

                    X = np.concatenate((X,Xb), axis=0)
                    Y = np.concatenate((Y, Yb), axis=0)

                model.train(X,Y,epochs=1)


    # ------------------ Test Fault detection method on all area, break mask -----------------------------
    elif work.__eq__("test-fault-all"):

        windowList = [35,45]

        #testList = list(range(36))
        testList = list([35])

        for i in testList:

            idx = np.int((i+1) * 10)
            ds_fname = DSDIR + "Australia_{}.mat".format(idx)
            #ds_fname = DSDIR + "QUEST_0.mat"

            ds = DATASET(ds_fname)

            O = np.zeros((ds.x, ds.y, 3))
            O[:, :, 0] = ds.OUTPUT * 255
            O[:, :, 1] = ds.OUTPUT * 255
            O[:, :, 2] = ds.OUTPUT * 255
            O = np.uint8(O)

            mergeAll = np.zeros((ds.x, ds.y, 3))

            for W in windowList:

                fname = CB + 'FaultDetection_{}.hdf5'.format(W)
                model = MODEL(w = W, param_dir=fname)

                if DEBUG_MODE:
                    print("-"*30)
                    print("Loading Model W = {}".format(W))
                    print("Drawing output for rotation number : {}".format(i+1))


                [X, Y, IDX] = ds.generateDS(ds.OUTPUT, ds.MASK, ratio=0.25, output_type=0, w =W)
                Yh = model.predict(X)

                #d = drawLines(O, IDX , Yh, WIDTH = 1 , FILL = 128, ws = 1, fname = FG+'Map_allarea_w{}_{}.png'.format(W,i+1))
                d = drawLines(O, IDX, Yh, WIDTH=1, FILL=128, ws=1, fname=FG + 'Map_r_allarea_w{}_{}.png'.format(W, i + 1))

                pmap = probMap(ds.OUTPUT.shape, IDX, Yh)
                #P = PmapViewer(matrix=pmap, bg=ds.OUTPUT)
                #P.save(FG+'Map_allarea_w{}_{}.npz'.format(W,i+1))
                #P.save(FG + 'Map_allarea_QUEST_w{}_{}.npz'.format(W, i + 1))

                del X
                del Y
                del IDX

            mergeAll = mergeAll / len(windowList)
            showMatrix(mergeAll, dim=3, fname=FG + 'MergeAll_{}.png'.format(i+1), show=False)
            #showMatrix(mergeAll, dim=3, fname=FG + 'MergeAll_QUEST_{}.png'.format(i + 1), show=False)

    # ------------------ Test Fault detection method on all area, break mask -----------------------------
    elif work.__eq__("test-fault-all-derotate"):

        fname = CB + 'FaultDetection.hdf5'
        model = MODEL(param_dir=fname)
        windowList = [15, 21, 27, 35, 45]

        #testList = list(range(36))
        testList = list([35])

        for i in testList:

            if DEBUG_MODE:
                print("Drawing output for rotation number : {}".format(i+1))

            idx = np.int((i+1) * 10)
            ds_fname = DSDIR + "Australia_{}.mat".format(idx)

            ds = DATASET(ds_fname)

            O = np.zeros((ds.x, ds.y, 3))
            O[:, :, 0] = ds.OUTPUT * 255
            O[:, :, 1] = ds.OUTPUT * 255
            O[:, :, 2] = ds.OUTPUT * 255
            O = np.uint8(O)


            [X, Y, IDX] = ds.generateDS(ds.OUTPUT, ds.testMask, raio = 0.99, output_type=0)
            Yh = model.predict(X)
            d = drawLines(O, IDX , Yh, WIDTH = 1 , FILL = 128, ws = 1, fname = FG+'Map_testarea_{}.png'.format(i+1))
            del X
            del Y
            del IDX
            d = rotateWithMap(d, ds.M2R, map_type = 'm2r', dim = 3)

            if i == 0:
                mergeTest = np.zeros(d.shape)
                mergeAll = np.zeros(d.shape)

            mergeTest = mergeTest + d

            [X, Y, IDX] = ds.generateDS(ds.OUTPUT, ds.MASK, raio=0.25, output_type=0)
            Yh = model.predict(X)
            d = drawLines(O, IDX , Yh, WIDTH = 1 , FILL = 128, ws = 1, fname = FG+'Map_allarea_{}.png'.format(i+1))
            del X
            del Y
            del IDX
            d = rotateWithMap(d, ds.M2R, map_type='m2r', dim=3)

            mergeAll = mergeAll + d

        mergeAll = np.uint8(mergeAll / 36)
        im = Image.fromarray(mergeAll)
        im.save(FG+'mergeAll.png')
        mergeTest = np.uint8(mergeTest / 36)
        im = Image.fromarray(mergeTest)
        im.save(FG+'mergeTest.png')


    elif work.__eq__("prepare-datasets-ang"):

        ds = DATASET(DSDIR + 'Australia_360.mat')

        NFILE = 30
        NREPEAT = 10
        RATIO = 0.5
        FNAME = DSREADY + 'ANG_'
        flt_name = FILTERDIR + 'Filters_w45_100.mat'


        for t1 in range(NFILE):

            X = np.zeros((1, WindowSize, WindowSize, Layers))
            Y = np.zeros((1, 1))
            for t2 in range(NREPEAT):

                [Xb, Yb, IDXb] = ds.generateDSwithFilter('train',ds.DEGREES, ds.trainMask, ratio=RATIO, choosy=True)
                X = np.concatenate((X, Xb), axis=0)
                Y = np.concatenate((Y, Yb), axis=0)

            np.savez(FNAME+'{}'.format(t1), X=X, Y=Y)


    elif work.__eq__("prepare-datasets-flt"):

        W = args.WSIZE

        ds1 = DATASET(DSDIR + 'Australia_strip.mat')
        ds2 = DATASET(DSDIR + 'QUEST_strip.mat')

        RATIO = [0.04, 0.005]
        oname = ['A_','Q_']

        ds1.expandBy(width=W, epsilon=0.9)
        ds2.expandBy(width=W, epsilon=0.9)

        ds = [ds1, ds2]

        NFILE = [100,100]

        for t2 in range(len(ds)):

            for t1 in range(NFILE[t2]):

                [X, Y, IDXb] = ds[t2].generateDSwithFilter('train',ds[t2].OUTPUT, ds[t2].trainMask, w=W , ratio=RATIO[t2], choosy=False)

                FNAME = DSREADY + oname[t2] + '{}'.format(t1)
                np.savez(FNAME, X=X, Y=Y)


    elif work == "train-prepared":

        W = args.WSIZE

        prefix = args.prepprefix
        nfile = args.prefnumber

        if prefix == "A_":
            fname = CB + '{}_Fault_Australia.hdf5'.format(W)
        elif prefix == "Q_":
            fname = CB + '{}_Fault_Quest.hdf5'.format(W)
        else:
            fname = CB + '{}_Fault_Mixed.hdf5'.format(W)

        model = MODEL(w=W, checkpoint=fname)


        for i in range(nfile):

            if DEBUG_MODE:
                print("******* Working of prepared file: {}".format(i+1) + '-'+ slideBar(i*100/args.prefnumber))

            if prefix == "A_" or prefix == "Q_":
                ds_fname = DSREADY+ prefix + "{}.npz".format(i)
                data = np.load(ds_fname)
                model.train(data['X'], data['Y'], epochs=1)

            else:
                mixed_list = ['A_','Q_']
                X = np.zeros((1, W, W, Layers))
                Y = np.zeros((1, 1))

                for p in mixed_list:
                    ds_fname = DSREADY + p + "{}.npz".format(i)
                    data = np.load(ds_fname)
                    X = np.concatenate((X, data['X']), axis=0)
                    Y = np.concatenate((Y, data['Y']), axis=0)

                model.train(X, Y, epochs=1)


    elif work.__eq__("test-fault-all-prep"):

        testList = ['Australia_strip.mat', 'QUEST_strip.mat']

        W = 45
        flt_name = FILTERDIR + 'Filters_0_w45.mat'

        fname = CB + args.callback
        model = MODEL(w=W, param_dir=fname)


        for T in testList:
            ds_fname = DSDIR + T
            ds = DATASET(ds_fname)

            O = np.zeros((ds.x, ds.y, 3))
            O[:, :, 0] = ds.OUTPUT * 255
            O[:, :, 1] = ds.OUTPUT * 255
            O[:, :, 2] = ds.OUTPUT * 255
            O = np.uint8(O)

            if DEBUG_MODE:
                print("-"*30)
                print("Loading Model W = {}".format(W))
                print("Drawing output for rotation number : {}".format(i+1))

            [X, Y, IDX] = ds.generateDSwithFilter('test', ds.OUTPUT, ds.MASK, ratio=0.1, w=W, choosy=False)
            Yh = model.predict(X)

            d = drawLines(O, IDX , Yh, WIDTH = 1 , FILL = 128, ws = 1, fname = FG+'Map_allarea_w{}_{}.png'.format(W,i+1), threshold=0.4)
            pmap = probMap(ds.OUTPUT.shape, IDX, Yh)
            del X
            del Y
            del IDX

            ofname = FG + 'Probmap_{}.png'.format(idx)
            showMatrix(pmap, dim=2, fname=ofname, show=False)

            P = PmapViewer(matrix=pmap, bg = ds.OUTPUT)
            P.save(dir = FG + 'Probmap_{}.npz'.format(idx))


    elif work.__eq__("test-choosy-prepared"):

        #testList = list(range(36))     # See Results on all different rotations
        testList = list([35])           # See Results only on main file (because 36 = 360 degrees rotation = main file)
        W = 45

        flt_name = FILTERDIR + 'Filters_w45_36.mat'
        FLT = FILTER(flt_name)

        model = MODEL(w=W, param_dir=CB + 'Rotate_choosy.hdf5')

        for i in testList:

            if DEBUG_MODE:
                print("-"*30)
                print("Drawing output for rotation number : {}".format(i+1))

            idx = np.int((i+1)*10)
            ds_fname = DSDIR + "Australia_{}.mat".format(idx)
            ds = DATASET(ds_fname)

            PMAPS = np.zeros((ds.OUTPUT.shape[0], ds.OUTPUT.shape[1], FLT.N))
            [X, Y, IDX] = ds.generateDSwithFilter('test', ds.DEGREES, ds.MASK, ratio=0.99, w=W, choosy=True)

            for r in range(FLT.N):

                [fnum,filter] = FLT.getFilterbyNumber(r)

                Xr = np.array(X)
                xr = np.zeros((W,W,Layers))

                # Rotate test test:
                for id in range(Xr.shape[0]):
                    xr = Xr[id, :,:,:]
                    Xr[id, :,:,:] = rotateWithMap(xr, filter, map_type = 'm2r', dim = 2)

                Yh = model.predict(Xr)
                PMAPS[:,:,r] = probMap(ds.OUTPUT.shape, IDX, Yh)

            P = PmapViewer(matrix=PMAPS, bg = ds.OUTPUT)
            P.save(dir=FG + 'Probmap_choosy_{}.npz'.format(idx))


    elif work.__eq__("apply-on-prediction"):
        Wf = 45 # Fault detection window size
        Wa = 45 # Angel detection window size
        threshold = 0.4

        flt_name = FILTERDIR + 'Filters_0_w45.mat'

        ds_fname = DSDIR + "Australia_360.mat"
        ds = DATASET(ds_fname)

        model_flt = MODEL(w=Wf, param_dir=CB + 'FaultDetection.hdf5')
        model_ang = MODEL(w=Wa, param_dir=CB + 'Rotate_choosy.hdf5')

        [X, Y, IDX] = ds.generateDSwithFilter('test', ds.OUTPUT, ds.MASK, ratio=0.2, w=Wf, choosy=False)
        Yh1 = model_flt.predict(X)
        pmap = probMap(ds.OUTPUT.shape, IDX, Yh1)
        newMask = pmapCutoff(pmap, threshold)


        flt_name = FILTERDIR + 'Filters_w45_36.mat'
        FLT = FILTER(flt_name)

        PMAPS = np.zeros((ds.OUTPUT.shape[0], ds.OUTPUT.shape[1], FLT.N))
        [X, Y, IDX] = ds.generateDSwithFilter('test', ds.DEGREES, newMask, ratio=0.99, w=Wa,
                                              choosy=True)

        MaxProb = -np.ones(len(Y))
        MaxSlope = np.full(len(Y), np.pi / 2.0)

        for r in range(FLT.N//2):

            [fnum, filter] = FLT.getFilterbyNumber(r)
            slope = 2*np.pi*r / FLT.N
            slope = np.arctan(np.tan(slope))


            Xr = np.array(X)
            xr = np.zeros((Wa, Wa, Layers))

            # Rotate test test:
            for id in range(Xr.shape[0]):
                xr = Xr[id, :, :, :]
                Xr[id, :, :, :] = rotateWithMap(xr, filter, map_type='m2r', dim=2)

            Yh = model_ang.predict(Xr)

            YhNormal = np.ndarray.flatten(Yh)
            mIdx = np.where(MaxProb < YhNormal)
            MaxProb[mIdx] = YhNormal[mIdx]
            MaxSlope[mIdx] = slope

        empty = np.uint8(np.zeros((ds.x, ds.y, 3)))
        tmp = drawLinesSlope(empty, IDX, MaxSlope, ws=10, fname=FG + 'Combined_Alone.png')

        for r in range(3):
            empty[:, :, r] = np.uint8(ds.OUTPUT*255)

        tmp = drawLinesSlope(empty, IDX, MaxSlope, ws=15, fname=FG + 'Combined_Ovelay.png')


    elif work.__eq__("prepare-pmap"):
        Wf = args.WSIZE
        ratio = 0.999

        testList = ['Australia_strip.mat', 'QUEST_strip.mat']

        for T in testList:
            ds_fname = DSDIR + T
            ds = DATASET(ds_fname)

            model_flt = MODEL(w=Wf, param_dir=CB + args.callback)

            masknumber = 80
            masks = ds.shrinkMask(maskName="all", number=masknumber)
            pmap = np.zeros(ds.OUTPUT.shape)

            for i in range(masknumber):
                [X, Y, IDX] = ds.generateDSwithFilter('test', ds.OUTPUT, masks[i], ratio=ratio, w=Wf, choosy=False)
                Yh1 = model_flt.predict(X)
                pmap_tmp = probMap(ds.OUTPUT.shape, IDX, Yh1)
                pmap = np.maximum(pmap, pmap_tmp)

            # Logging activity:
            L = Logger()
            L.addlog("-"*30)
            L.addlog(" W = {} ".format(Wf))
            L.addlog(" Callback = {}".format(args.callback))
            L.addlog(" Map = {}".format(T))

            ev_train = ds.evaluate(pmap, Wf, 'train')
            ev_test = ds.evaluate(pmap, Wf, 'test')
            ev_all = ds.evaluate(pmap, Wf, 'all')

            L.addlog(" Train Error = {} , {}".format(ev_train[0], ev_train[1]))
            L.addlog(" Test Error = {} , {}".format(ev_test[0], ev_test[1]))
            L.addlog(" All Error = {} , {}".format(ev_all[0], ev_all[1]))

            pmapname = PMAP_DIR + '{}_Pmamp_'.format(Wf)+ args.callback + '_on_{}_'.format(T[:5]) + '.npz'
            np.savez(pmapname, matrix=pmap)


    elif work == "evaluate_pmap":
        T = args.prepprefix
        ds_fname = DSDIR + T
        ds = DATASET(ds_fname)

        Train_E = [[],[]]
        Test_E = [[], []]
        All_E = [[], []]

        for Wf in range(9,57,4):

            if DEBUG_MODE:
                print("- Evaluating {} ------- W = {}".format(T, Wf))


            # if args.callback == 'zeros':
            #     pmap = np.zeros_like(ds.OUTPUT)
            # elif args.callback == 'ones':
            #     pmap = np.ones_like(ds.OUTPUT)
            # else:
            #     pmap = ds.expandBy(Wf, epsilon=0.9, set=False)

            pmapname = PMAP_DIR + '{}_Pmamp_'.format(Wf) + "{}_".format(Wf) + args.callback + '_on_{}_'.format(T[:5]) + '.npz'
            pmap = np.load(pmapname)['matrix']


            eval_type = 'loss'

            [pos, neg] = ds.evaluate(pmap, Wf, 'train', eval_type)
            Train_E[0] += [pos]
            Train_E[1] += [neg]

            [pos, neg] = ds.evaluate(pmap, Wf, 'test', eval_type)
            Test_E[0] += [pos]
            Test_E[1] += [neg]

            [pos, neg] = ds.evaluate(pmap, Wf, 'all', eval_type)
            All_E[0] += [pos]
            All_E[1] += [neg]

        errors = {'TrainE':Train_E , 'TestE':Test_E, 'AllE':All_E}
        sio.savemat('loss_'+ args.callback + "_" + T + "_eval.mat", errors)


    else:
        print(globals())
        print("No job is defined!")
