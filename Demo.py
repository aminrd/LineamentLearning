
__author__ = "Amin Aghaee"
__copyright__ = "Copyright 2018, Amin Aghaee"

from DATASET import *
from PmapViewer import *

LOAD_BACKGROUND = True
AUSTRALIA = True

dir1 = './Results/TrainOnRandomSeleciton_W45_fault_QUEST/Pmap_exist_aust.npz'
dir2 = './Results/TrainOnRandomSeleciton_W45_fault_QUEST/Pmap_exist_quest.npz'
dir3 = './Results/TrainOnRandomSelection_w45_fault/Pmap_exist_quest.npz'
dir4 = './Results/TrainOnRandomSelection_w45_fault/PMAP_exist.npz'

dir5 = './Results/NewTrainingRandom_strip_mixed/Pmamp_Fault_Australia.hdf5Australia_strip.mat.npz'
dir6 = './Results/NewTrainingRandom_strip_mixed/Pmamp_Fault_Australia.hdf5QUEST_strip.mat.npz'
dir7 = './Results/NewTrainingRandom_strip_mixed/Pmamp_Fault_Mixed.hdf5Australia_strip.mat.npz'
dir8 = './Results/NewTrainingRandom_strip_mixed/Pmamp_Fault_Mixed.hdf5QUEST_strip.mat.npz'
dir9 = './Results/NewTrainingRandom_strip_mixed/Pmamp_Fault_Quest.hdf5Australia_strip.mat.npz'
dir10 = './Results/NewTrainingRandom_strip_mixed/Pmamp_Fault_Quest.hdf5QUEST_strip.mat.npz'

dir11 = './Results/First3Layers/Pmamp_Fault_Australia.hdf5Australia_strip.mat.npz'


jdir = './applet.json'
# Load and run application:
p = PmapViewer(dir=jdir)
p.run()

