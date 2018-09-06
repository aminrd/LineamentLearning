import os
import sys
import numpy as np
import random
from globalVariables import *
from Utility import *
from DATASET import *

class FILTER:
    def __init__(self, directory = FILTERDIR + 'Default.mat'):
        FDS = sio.loadmat(directory)

        self.F = FDS['filters']
        self.N = self.F.shape[0]
        self.rlist = FDS['rotations']

    def getFilter(self, n = 1):
        fnum = random.sample(range(self.N), n)
        return [fnum, self.F[fnum, :, :]]

    def getRadian(selfs, index):
        return (selfs.rlist[index]*np.pi) / 180.0

    def getFilterbyNumber(self, fnum = 0):
        return [fnum, self.F[fnum, :, :]]

