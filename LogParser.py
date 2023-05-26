import re
import numpy as np
import scipy.io as sio

model_australia = {}
model_quest = {}
model_mixed = {}

models = [model_australia, model_quest, model_mixed]

for m in models:
    m['Quest'] = {}
    m['Australia'] = {}

    for mp in ['Quest', 'Australia']:
        m[mp]['Train_p'] = []
        m[mp]['Train_n'] = []
        m[mp]['Test_p'] = []
        m[mp]['Test_n'] = []
        m[mp]['All_p'] = []
        m[mp]['All_n'] = []


with open("log.txt", "r") as f:
    for it in range(72):

        l1 = f.readline()[29:]
        l2 = f.readline()[29:]
        l3 = f.readline()[29:]
        l4 = f.readline()[29:]
        l5 = f.readline()[29:]
        l6 = f.readline()[29:]
        l7 = f.readline()[29:]

        idx = int(re.findall("\d+", l2)[0])
        idx = (idx - 9) // 4

        if 'Quest' in l3:
            model = model_quest
        elif 'Mixed' in l3:
            model = model_mixed
        else:
            model = model_australia

        if 'QUEST' in l4:
            map_name = 'Quest'
        else:
            map_name = 'Australia'

        [p,n] = re.findall("\d+\.\d+", l5)
        model[map_name]['Train_p'] += [float(p)]
        model[map_name]['Train_n'] += [float(n)]

        [p, n] = re.findall("\d+\.\d+", l6)
        model[map_name]['Test_p'] += [float(p)]
        model[map_name]['Test_n'] += [float(n)]

        [p, n] = re.findall("\d+\.\d+", l7)
        model[map_name]['All_p'] += [float(p)]
        model[map_name]['All_n'] += [float(n)]

sio.savemat('Australia.mat', model_australia)
sio.savemat('Quest.mat', model_quest)
sio.savemat('Mixed.mat', model_mixed)