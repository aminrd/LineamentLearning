# Module : Dataset
# Loading dataset from MATLAB files , Expanding fault lines
__author__ = "Amin Aghaee"
__copyright__ = "Copyright 2018, Amin Aghaee"

import sys
import numpy as np
import random
import scipy.io as sio
import scipy.ndimage

from globalVariables import *
from Utility import *
#from FILTER import *


if DEBUG_MODE:
    print("### Importing DATASET Class ###")


def gaussian2DMatrix(width = 5, epsilon = 0.9, type = 'manhattan'):
    matrix = np.zeros((width,width))
    [x,y] = [np.int((width-1)/2),np.int((width-1)/2)]
    s = np.int((width-1)/2)

    if type.__eq__("manhattan"):
        for i in range(0-s,s+1):
            for j in range(0 - s, s + 2):
                D = np.abs(i) + np.abs(j)
                if D < s+1:
                    matrix[x+i][y+j] = pow(epsilon, D)
    elif type.__eq__("gaussian"):
        for i in range(0-s,s+1):
            for j in range(0 - s, s + 2):
                D = np.abs(i) + np.abs(j)
                if D < s+1:
                    matrix[x+i][y+j] = np.exp( (0 - (i*i + j*j))/(2 * epsilon * epsilon) )

    else: # Simply expand by _width pixels
        for i in range(0-s,s+1):
            for j in range(0 - s, s + 2):
                D = np.abs(i) + np.abs(j)
                if D < s+1:
                    matrix[x+i][y+j] = 1.0
    return matrix


def labelAngel(radian, base = np.pi / 2.0):

    if base == np.pi / 2.0:
        return np.abs(np.abs(radian) - base) <= radianTH

    return np.abs(radian - base) <= radianTH

class DATASET:
    """ Dataset Class, Loads dataset from MATLAB file. Have some function to expand fault lines, ...."""

    def __init__(self , directory, mode = 'normal'):

        DS = sio.loadmat(directory)
        self.x = DS['I1'].shape[0]
        self.y = DS['I1'].shape[1]

        self.INPUTS = np.zeros((self.x, self.y, Layers))

        for i in range(Layers):
            self.INPUTS[:,:,i] = np.array(DS['I{}'.format(i+1)])

        self.MASK = np.array(DS['mask'])
        self.trainMask = np.array(DS['train_mask'])

        if mode.__eq__('normal'):
            self.testMask = np.array(DS['test_mask'])
            self.OUTPUT = np.array(DS['output'])
            self.R2M = np.array(DS['R2M'])
            self.M2R = np.array(DS['M2R'])

        self.DEGREES = np.array(DS['DEGREES'])

        for i in range(Layers):
            self.INPUTS[:, :, i] = myNormalizer(self.INPUTS[:, :, i])


    def expandBy(self, width=3, epsilon = 1.0, type = 'manhattan', set = True):

        if width==0 and set==False:
            return self.OUTPUT

        matrix = np.array(self.OUTPUT).astype(float)
        [a,b] = np.where(matrix == 1)

        GMAT = gaussian2DMatrix(width, epsilon, type)
        s = np.int((width-1)/2)

        for k in range(len(a)):
            [i,j] = [a[k], b[k]]

            if i<s+1 or i>self.x-s-1 or j<s+1 or j>self.y-s-s:
                continue

            submat = matrix[i - s:i + s+1, j - s:j + s+1]
            matrix[i - s:i + s+1, j - s:j + s+1] = np.maximum(GMAT, submat)

        if set:
            self.OUTPUT = matrix
        else:
            return matrix

    def generateDS(self, output, mask, w = WindowSize, choosy = False, ratio = 1.0, output_type = np.pi / 2.0):
        # When choosy = TRUE : it only picks the fault locations
        # ratio coresponds to randomly selecting all possible locations
        # output_type: 1 --> Degree , Otherwise ---> Binary
        input = np.array(self.INPUTS)

        s = np.uint32((w-1)/2)
        O = np.array(output)
        O[np.where(mask == 0)] = 0

        if output_type == 0:
            O[O.nonzero()] = 1

        if choosy == True:
            IDX = np.where(O > 0) # Find where there is a fault
        else:
            IDX = np.where(mask == 1) # Use whole of mask

        # Choosing samples randomly : Shuffling data and randomly choosing them usnig "ratio"
        subset = random.sample(range(len(IDX[0])), np.uint32(np.floor(ratio * len(IDX[0]))))
        subset = np.uint32(subset)
        IDX = np.array(IDX)
        IDX = IDX[:,subset]
        IDX = tuple(IDX)

        w = np.uint32(w)
        X = np.zeros([len(IDX[0]), w, w, Layers])
        Y = np.zeros([len(IDX[0]), 1])

        for k in range(len(IDX[0])):

            if DEBUG_MODE and np.random.rand() < 0.01:
                pct = k * 100 / len(IDX[0])
                print(slideBar(pct) + '-- Preparing dataset, about ' + '{}'.format(pct) + 'done!')

            [i,j] = [IDX[0][k],IDX[1][k]]
            X[k,:,:,:] = np.reshape(input[i-s:i+s+1, j-s:j+s+1, :] , (1, w, w, Layers))


            if output_type == 0: # All areas, not only faults
                Y[k] = O[i, j]
            else:
                Y[k] = labelAngel(O[i, j], output_type)


        return [X,Y, IDX]



    def generateDSwithFilter(self, dstype, output, mask, w = WindowSize, choosy = False, ratio = 1.0):
        # When choosy = TRUE : it only picks the fault locations and labels are based on fault angels
        # ratio coresponds to randomly selecting all possible locations

        input = np.array(self.INPUTS)

        s = np.uint32((w-1)/2)
        O = np.array(output)
        O[np.where(mask == 0)] = 0

        if choosy == True:
            IDX = np.where(O > 0) # Find where there is a fault
        else:
            IDX = np.where(mask == 1) # Use whole of mask

        # Choosing samples randomly : Shuffling data and randomly choosing them usnig "ratio"
        subset = random.sample(range(len(IDX[0])), np.uint32(np.floor(ratio * len(IDX[0]))))
        subset = np.uint32(subset)
        IDX = np.array(IDX)
        IDX = IDX[:,subset]
        IDX = tuple(IDX)

        w = np.uint32(w)
        X = np.zeros([len(IDX[0]), w, w, Layers])
        Y = np.zeros([len(IDX[0]), 1])

        inverted_mask = ~circular_mask(w)

        for k in range(len(IDX[0])):

            if DEBUG_MODE and np.random.rand() < 0.01:
                pct = k * 100 / len(IDX[0])
                print(slideBar(pct) + '-- Preparing dataset, about ' + '{}'.format(pct) + 'done!')

            [i,j] = [IDX[0][k],IDX[1][k]]
            xr = np.array(input[i-s:i+s+1, j-s:j+s+1, :])

            if dstype == 'train':
                X[k,:,:,:] = scipy.ndimage.rotate(xr, random.randrange(0, 360, 6), reshape=False, order=0)
            else:
                X[k, :, :, :] = scipy.ndimage.rotate(xr, 0, reshape=False, order=0)

            X[k,:,:,:][inverted_mask] = 0


            if choosy == False: # All areas, not only faults
                Y[k] = O[i, j]
            else:#TODO: Non choosy not supported yet
                Y[k] = O[i, j]

        return [X,Y, IDX]




    def shrinkMask(self, maskName = 'train', number = 9):
        # Shrink mask into 1/9 and return 9 masks:

        if maskName.__eq__('train'):
            M = np.array(self.trainMask)
        elif maskName.__eq__('all'):
            M = np.array(self.MASK)
        elif maskName.__eq__('whole'):
            M = np.ones(self.MASK.shape)
            offset = 100
            M[:, 0:offset] = 0
            M[0:offset, :] = 0
            M[:, 0 - offset:] = 0
            M[0 - offset:, :] = 0
        else:
            M = np.array(self.testMask)

        m = np.zeros((number, self.x, self.y))
        idx = np.where(M == 1)
        idx = np.array(idx)

        cnt = idx.shape[1] // number

        for i in range(number):
            mprim = m[i]
            subidx = idx[:, cnt*i : cnt*(i+1)]
            subidx = tuple(subidx)
            mprim[subidx] = 1
            m[i] = mprim

        return m




    def evaluate(self, _pmap, expand=0, mask = 'all', etype = 'our'):
        pmap = np.array(_pmap)
        labels = self.expandBy(width=expand, epsilon=0.9 ,type='normal', set=False)


        if mask.__eq__('train'):
            maskFilter = self.trainMask
            labels[np.where(self.trainMask == 0)] = 0
            pmap[np.where(self.trainMask == 0)] = 0
        elif mask.__eq__('test'):
            maskFilter = self.testMask
            labels[np.where(self.testMask == 0)] = 0
            pmap[np.where(self.testMask == 0)] = 0
        else:
            maskFilter = self.MASK
            labels[np.where(self.MASK == 0)] = 0
            pmap[np.where(self.MASK == 0)] = 0



        if etype == 'our':
            IDX_pos = labels > 0
            differror = np.square(labels - pmap)
            differror[~IDX_pos] = 0
            pos_score = differror.sum() / IDX_pos.sum()



            IDX_neg = labels <= 0
            differror = np.square(labels - pmap)
            differror[~IDX_neg] = 0
            neg_score = differror.sum() / max(1, (pmap[IDX_neg] >0 ).sum())

            IDXa = np.where(pmap > 0)


            return [pos_score, neg_score]


        else:
            EPS = np.finfo(float).eps

            yh = np.copy(pmap)
            yh[ yh == 1.0 ] = 1 - EPS
            yh[ yh == 0.0 ] = EPS

            y = np.copy(labels)
            y[ y == 1.0 ] = 1 - EPS
            y[ y == 0.0 ] = EPS


            loss = np.multiply(yh, np.log(yh)) + np.multiply((1.0 - y), np.log( 1-yh ))

            err = -np.sum( loss[maskFilter == 1] ) / np.sum(maskFilter)
            return [err,err]
