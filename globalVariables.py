# Define all global variables here:
import numpy as np

DEBUG_MODE = True   # [DB=T][DB=F]

# --------------------------------------------------
WindowSize = 45     # Size of window for learning slopes
Layers = 8          # Layer number which running our model on
fileNumber = 36    # 36 for only using the rotations, 108 for flipping as well



maskTh = 0.9        # For TH = 0.9 , means when 90% of a window is inside a mask, it is acceptable
radianTH = np.pi / 12.0
ITERATIONS = 150   # Maximum number of Iterations on learning procedures
# Number of different models for Degrees
# E.g. each of those models are designed to predict different angels
NUMBER_OF_DEGREE_MODELS = 6



MATLAB_DATASET_FILE = 'PYDataset.mat'


# --------------------------------------------------
# Directories:

CB = './CallBacks/Rotate/'
FG = './Figures/Rotate/'
DSDIR = './Dataset/Australia/Rotations/'
DSREADY = './Dataset/DSREADY/'
FILTERDIR = './Filters/'
PMAP_DIR = './Pmaps/'

# --------------------------------------------------


# APPLET global variables:
MAX_WINDOW_SIZE = 1000
LOAD_MODELS = False
