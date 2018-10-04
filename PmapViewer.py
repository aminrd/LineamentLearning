#Pmap Viewer
import numpy as np
__author__ = "Amin Aghaee"
__copyright__ = "Copyright 2018, Amin Aghaee"

from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
import json


# Loading Related Modules:
# --------------------------------------
from globalVariables import *
from Utility import *
from DATASET import *
from FILTER import *
import matplotlib.pyplot as plt

if LOAD_MODELS:
    from MODEL import *
# --------------------------------------
from Prob2Line import *


class PmapViewer:
    def __init__(self, matrix=None, bg = None, dir = None):

        self.RUN = False

        sz = MAX_WINDOW_SIZE

        self.FILL = 'red'
        self.ErrorIsReady = False


        if matrix is None and dir is None:
            self.matrix = np.zeros((sz,sz))
            self.width = sz
            self.height = sz
            self.width2 = sz
            self.height2 = sz

        elif not matrix is None:

            if len(matrix.shape) < 3:
                self.matrix = np.zeros((matrix.shape[0], matrix.shape[1]))
                self.matrix = matrix
            else:
                self.matrix = matrix

            self.width = matrix.shape[0]
            self.height = matrix.shape[1]

            if self.width > sz or self.height > sz:
                if self.width > sz:
                    self.height2 = (sz * self.height) // self.width
                    self.width2 = sz
                else:
                    self.width2 = (sz * self.width) // self.height
                    self.height2 = sz
            else:
                self.height2 = self.height
                self.width2 = self.width

        elif not dir is None:
            self.load(dir)



        if not bg is None:
            if len(bg.shape) >= 3:
                self.bg = bg
            else:
                self.bg = np.zeros((bg.shape[0], bg.shape[1], 3))
                self.bg[:, :, 0] = bg
                self.bg[:, :, 1] = bg
                self.bg[:, :, 2] = bg
                self.bg = np.uint8(self.bg)
        else:
            self.bg = np.zeros((self.ds.OUTPUT.shape[0], self.ds.OUTPUT.shape[1], 3))

            oex = self.ds.expandBy(width=45, epsilon=0.9, set=False)
            #oex = self.ds.expandBy(width=3, epsilon=0.9, set=False)

            #self.bg[:, :, 0] = self.ds.OUTPUT
            #self.bg[:, :, 1] = self.ds.OUTPUT
            #self.bg[:, :, 2] = self.ds.OUTPUT

            self.bg[:, :, 0] = oex
            self.bg[:, :, 1] = oex
            self.bg[:, :, 2] = oex

            self.bg = np.uint8(self.bg * 255)


        self.master = tk.Tk()





    def load(self, dir = './applet.json'):

        with open(dir) as f:
            self.jfile = json.load(f)

        self.ds = DATASET(self.jfile["dataset"]["link"])


        if LOAD_MODELS:
            self.wf = int(self.jfile["model1"]["w"])
            self.model_flt = MODEL(w=self.wf, param_dir=self.jfile["model1"]["link"])

            self.wa = int(self.jfile["model2"]["w"])
            self.model_ang = MODEL(w=self.wa, param_dir=self.jfile["model2"]["link"])


        ds = np.load(self.jfile["pmap"]["plink"])
        self.lnumber = int(self.jfile["pmap"]["lnumber"])
        self.matrix = ds['matrix']
        self.width = self.matrix.shape[0]
        self.height = self.matrix.shape[1]

        if int(self.jfile["pmap"]["trained"]) == 1:
            h = np.load(self.jfile["pmap"]["alink"])
            #self.angels = h['matrix']
            self.angels = np.zeros((self.width, self.height, 36))




        sz = MAX_WINDOW_SIZE
        if self.width > sz or self.height > sz:
            if self.width > sz:
                self.height2 = (sz * self.height) // self.width
                self.width2 = sz
            else:
                self.width2 = (sz * self.width) // self.height
                self.height2 = sz
        else:
            self.height2 = self.height
            self.width2 = self.width




    # --------------------------------------------------------------------------

    def getBackground(self, showLines=False, c1 = 1, c2 = 254, layer=-1):

        BG = np.ones((self.width, self.height, 3))

        if layer == -1:
            BG = BG * c1
        else:
            ll = self.ds.INPUTS[:,:,layer] * 255
            BG[:, :, 0] = ll
            BG[:, :, 1] = ll
            BG[:, :, 2] = ll

        if showLines:
            ll = np.array(self.bg)
            BG[ll >= 200] = np.floor(c2 * 0.81)
            BG[ll >= 220] = np.floor(c2 * 0.9)
            BG[ll >= 250] = c2
            #BG = BG + c2 * self.bg
            #BG[BG > 255] = 255

        return np.uint8(BG)



    def getImage(self, showLines = False, angels=False, pct = 0.9, onlyMax = True, threshold = 0.5, cb=1, cl=254, sheet=-1, prob=False):

        p = pmapCutoff(self.matrix, threshold = threshold)

        BG = self.getBackground(showLines, cb, cl, sheet)

        if angels == False:

            if prob == False:
                cvec = colour2vec(self.FILL)

                ptrans = np.uint8(p * 255)
                map = np.zeros(BG.shape)

                for d in range(3):
                    cmap = self.ds.trainMask * cvec[d] + self.ds.testMask*(1-cvec[d])
                    cmap = cmap*255
                    map[:,:,d] = np.multiply(BG[:,:,d], (1-p)) + np.multiply(p,cmap)

            else:

                cvec = colour2vec(self.FILL)

                ptrans = np.uint8(self.matrix * 255)
                map = np.zeros(BG.shape)

                for d in range(3):
                    cmap = self.ds.trainMask * cvec[d] + self.ds.testMask*(1-cvec[d])
                    cmap = cmap*255
                    map[:,:,d] = np.multiply(BG[:,:,d], (1-self.matrix)) + np.multiply(self.matrix,cmap)


            # ------ Apply mask? ---------------
            APPLY_MASK = True

            if APPLY_MASK:
                for i in range(3):
                    mp = np.array(map[:,:,i])
                    mp[np.where(self.ds.MASK == 0)] = 0
                    map[:,:,i] = mp

            #-----------------------------------

            im = Image.fromarray(np.uint8(map))
            return im



        else:

            if int(self.jfile["pmap"]["trained"]) == 1:
                n = self.angels.shape[2]

                IDX = np.where(p >= 1)
                subset = np.random.choice(range(len(IDX[0])), np.uint32(np.floor(pct * len(IDX[0]))), replace=False)
                subset = np.uint32(subset)
                IDX = np.array(IDX)
                IDX = IDX[:, subset]
                IDX = tuple(IDX)

                ang_predictions = np.zeros((len(IDX[0]),n))

                k = len(IDX[0])

                MaxProb = -np.ones(k)
                MaxSlope = np.full(k, np.pi / 2.0)

                for i in range(n):
                    slope = 2 * np.pi * i / n
                    slope = np.arctan(np.tan(slope))

                    pm = self.angels[:,:,i]
                    ang_predictions[:,i] = np.ndarray.flatten(pm[IDX])

                    YhNormal = np.ndarray.flatten(pm[IDX])

                    mIdx = np.where(MaxProb < YhNormal)
                    MaxProb[mIdx] = YhNormal[mIdx]
                    MaxSlope[mIdx] = slope


                if onlyMax==False:

                    for r in range(n):
                        slope = 2 * np.pi * r / n
                        slope = np.arctan(np.tan(slope))
                        slope_m = np.repeat(slope, k)

                        BG = drawLinesSlope(BG, IDX, slope_m, ws=30, prelative=True, parray=ang_predictions[:,r], WIDTH=1, FILL=self.FILL)

                    return Image.fromarray(BG)


                else:
                    tmp = drawLinesSlope(BG, IDX, MaxSlope, ws=15, FILL=self.FILL)
                    return Image.fromarray(tmp)





            else:

                flt_name = self.jfile["filter"]["link"]
                FLT = FILTER(flt_name)


                [X, Y, IDX] = self.ds.generateDSwithFilter(FILTERDIR + 'Filters_0_w45.mat', self.ds.DEGREES, p, ratio=pct,
                                                      w=self.wa,
                                                      choosy=True)

                ang_predictions = np.zeros((len(Y), FLT.N))

                MaxProb = -np.ones(len(Y))
                MaxSlope = np.full(len(Y), np.pi / 2.0)

                for r in range(FLT.N):

                    if DEBUG_MODE:
                        print("++Preparing prediction for angel number {}".format(r))

                    [fnum, filter] = FLT.getFilterbyNumber(r)
                    slope = 2 * np.pi * r / FLT.N
                    slope = np.arctan(np.tan(slope))

                    Xr = np.array(X)
                    xr = np.zeros((self.wa, self.wa, Layers))

                    # Rotate test test:
                    for id in range(Xr.shape[0]):
                        xr = Xr[id, :, :, :]
                        Xr[id, :, :, :] = rotateWithMap(xr, filter, map_type='m2r', dim=2)

                    Yh = self.model_ang.predict(Xr)
                    YhNormal = np.ndarray.flatten(Yh)

                    ang_predictions[:,r] = YhNormal

                    mIdx = np.where(MaxProb < YhNormal)
                    MaxProb[mIdx] = YhNormal[mIdx]
                    MaxSlope[mIdx] = slope


                if showLines == False:
                    BG = np.uint8(np.zeros((self.ds.x, self.ds.y, 3)))
                else:
                    BG = self.bg


                if onlyMax == True:
                    tmp = drawLinesSlope(BG, IDX, MaxSlope, ws=15, FILL=self.FILL)
                    return Image.fromarray(tmp)

                else:

                    for r in range(FLT.N):
                        slope = 2 * np.pi * r / FLT.N
                        slope = np.arctan(np.tan(slope))
                        slope_m = np.repeat(slope, len(Y))

                        BG = drawLinesSlope(BG, IDX, slope_m, ws=15, prelative=True, parray=ang_predictions[:,r], FILL=self.FILL)

                    return Image.fromarray(BG)



    def plotEvaluation(self):

        fname = './applet_images/plot.png'

        if self.ErrorIsReady:
            im = Image.open(fname)
            im.show()
            return

        max_expand = 45
        xaxis = list(range(1,max_expand,2))

        train_err = np.zeros((len(xaxis), 2))
        test_err = np.zeros((len(xaxis), 2))
        all_err = np.zeros((len(xaxis), 2))

        for i in range(len(xaxis)):

            if DEBUG_MODE:
                print("<-> Expanding lines by {} pixels".format(xaxis[i]))

            train_err[i] = self.ds.evaluate(self.matrix, xaxis[i], 'train')
            test_err[i] = self.ds.evaluate(self.matrix, xaxis[i], 'test')
            all_err[i] = self.ds.evaluate(self.matrix, xaxis[i], 'all')



        f, axarr = plt.subplots(3, sharey=True)

        axarr[0].plot(xaxis, train_err[:,0], '+', xaxis, train_err[:,1], 'r--')
        axarr[0].set_title('Training errors')
        str = 'pos: {:10.3f}\n neg:{:10.3f}'.format(np.mean(train_err[:,0]), np.mean(train_err[:,1]))
        axarr[0].text(4, 0.5,str, horizontalalignment='right', verticalalignment='center')



        axarr[1].plot(xaxis, test_err[:, 0], '+', xaxis, test_err[:, 1], 'r--')
        axarr[1].set_title('Test errors')
        str = 'pos: {:10.3f}\n neg:{:10.3f}'.format(np.mean(test_err[:,0]), np.mean(test_err[:,1]))
        axarr[1].text(4, 0.5,str, horizontalalignment='right', verticalalignment='center')


        axarr[2].plot(xaxis, all_err[:, 0], '+', xaxis, all_err[:, 1], 'r--')
        axarr[2].set_title('Total errors')
        str = 'pos: {:10.3f}\n neg:{:10.3f}'.format(np.mean(all_err[:,0]), np.mean(all_err[:,1]))
        axarr[2].text(4, 0.5,str, horizontalalignment='right', verticalalignment='center')


        plt.savefig(fname)
        self.ErrorIsReady = True
        im = Image.open(fname)
        im.show()


    def requestImage(self):
        # Getting values from Applet:
        _showLines = self.CheckVar1.get() == 1
        _onlyMax = self.CheckVarMode.get() == 1
        _threshold = self.th.get() / 100

        #_pct = self.pcth.get()/100
        _pct = 0.1

        _angels = False
        _prob = self.CheckVarpmap.get() == 1

        _c1 = self.bgcol.get()
        _c2 = self.linecol.get()
        _sheet = self.lselect.index(tk.ACTIVE) - 1


        self.FILL = self.pcol.get(tk.ACTIVE)


        im = self.getImage(_showLines, _angels, _pct, _onlyMax, _threshold, _c1, _c2, _sheet, _prob)

        return im

    def showValues(self):

        im = self.requestImage()
        im2 = im.resize((self.width2, self.height2))
        photo = ImageTk.PhotoImage(im2)
        self.panel.configure(image = photo)
        self.panel.image = photo

        im.save('./Temp.png')


    def updateImage(self, im):
        im2 = im.resize((self.width2, self.height2))
        photo = ImageTk.PhotoImage(im2)
        self.panel.configure(image = photo)
        self.panel.image = photo



    def openImage(self):
        im = self.requestImage()
        im.save('./Temp.png')
        im.show()




    def close_window(self):
        self.master.destroy()



    def showclusters(self):
        p2l = prob2map(self.matrix)

        _coff = self.th.get() / 100
        _eps =  self.pcth.get()
        cmap = p2l.getClusters(cutoff=_coff, eps=_eps)
        img = p2l.showClusters(cmap)
        im = Image.fromarray(img)
        self.updateImage(im)

        im.save('./Temp.png')

        return im



    def convert2lines(self):
        p2l = prob2map(self.matrix)

        _coff = self.th.get() / 100
        _eps =  self.pcth.get()
        lines = p2l.makeConversion(_coff, _eps)

        _showLines = self.CheckVar1.get() == 1
        _cb = self.bgcol.get()
        _cl = self.linecol.get()
        _sheet = self.lselect.index(tk.ACTIVE) - 1
        bg = self.getBackground(_showLines, _cb, _cl, _sheet)


        im = drawLinesWithEndingPoints(bg, lines)
        self.updateImage(im)

        im.save('./Temp.png')

        return im







    def run(self):

        self.master.title("Probability Map Viewer")



        # FAULT EXISTENSE:
        # Scale bar to set threshold
        mainframe = tk.Frame(self.master)
        mainframe.pack()

        frame1 = tk.Frame(mainframe, borderwidth=2)
        frame1.pack(side=tk.LEFT)

        frame3 = tk.Frame(mainframe)
        frame3.pack(side=tk.LEFT)


        frame2 = tk.Frame(mainframe)
        frame2.pack(side=tk.LEFT)

        frame4 = tk.Frame(mainframe)
        frame4.pack(side = tk.LEFT)

        frame5 = tk.Frame(mainframe)
        frame5.pack(side = tk.RIGHT)



        checkFrame = tk.Frame(self.master)
        checkFrame.pack()

        buttonFrame = tk.Frame(self.master)
        buttonFrame.pack()



        # ========================================= #
        # ===============   FRAME 1  ============== #
        # ========================================= #

        self.th = tk.Scale(frame1, from_=1, to=100, orient=tk.HORIZONTAL, label='Labeling threshold', length=200)
        self.th.set( 50 )
        self.th.pack()

        self.pcth = tk.Scale(frame1, from_=3, to=10, resolution=0.2 , orient=tk.HORIZONTAL, label='Epsilon ', length=200)
        self.pcth.set( 1 )
        self.pcth.pack()



        # ========================================= #
        # ===============   FRAME 2  ============== #
        # ========================================= #


        self.bgcol = tk.Scale(frame2, from_=0, to=254, orient=tk.HORIZONTAL, length=100)
        self.bgcol.set(0)
        self.bgcol.pack()

        self.linecol = tk.Scale(frame2, from_=1, to=254, orient=tk.HORIZONTAL,length=100)
        self.linecol.set(254)
        self.linecol.pack()

        barimage = Image.open('./applet_images/graybar.png')
        img_bar = ImageTk.PhotoImage(barimage)
        pnl_bar = tk.Label(frame2, image=img_bar)
        pnl_bar.pack()



        # ========================================= #
        # ===============   FRAME 3  ============== #
        # ========================================= #


        tbg = tk.Text(frame3, height=2, width=15)
        tbg.pack()
        tbg.insert(tk.END, "Background's colour")

        tl = tk.Text(frame3, height=2, width=15)
        tl.pack()
        tl.insert(tk.END, "Line's colour")




        # ========================================= #
        # ===============   FRAME 4  ============== #
        # ========================================= #


        tpc = tk.Text(frame4, height=2, width=30)
        tpc.pack()
        tpc.insert(tk.END, "Prediction colour:")


        self.pcol = tk.Listbox(frame4, height=6)
        self.pcol.insert(1, 'red')
        self.pcol.insert(2, 'green')
        self.pcol.insert(3, 'blue')
        self.pcol.insert(4, 'yellow')
        self.pcol.insert(5, 'white')
        self.pcol.insert(6, 'black')
        self.pcol.pack()

        self.pcol.itemconfig(0, {'bg':'red'})
        self.pcol.itemconfig(1, {'bg': 'green'})
        self.pcol.itemconfig(2, {'bg': 'blue'})
        self.pcol.itemconfig(3, {'bg': 'yellow'})
        self.pcol.itemconfig(4, {'bg': 'white'})
        self.pcol.itemconfig(5, {'bg': 'black', 'fg':'white'})




        # ========================================= #
        # ===============   FRAME 5  ============== #
        # ========================================= #


        tls = tk.Text(frame5, height=2, width=30)
        tls.pack()
        tls.insert(tk.END, "Underlying map/sheet:")


        self.lselect = tk.Listbox(frame5, height=5)
        self.lselect.insert(1, 'Empty')
        self.lselect.insert(2, '1vd_TMI_RTP')
        self.lselect.insert(3, 'TMI_RTP')
        self.lselect.insert(4, 'Digital Elevation')
        self.lselect.insert(5, 'Isostatic Gravity')
        self.lselect.insert(6, 'RTP_HGM')
        self.lselect.insert(7, 'RTP_RS_HGM')
        self.lselect.insert(8, 'RTP_RD_HGM')
        self.lselect.insert(9, 'RTP_RI_HGM')
        self.lselect.pack(side=tk.RIGHT)




        # ========================================= #
        # ============== CHECK FRAME ============== #
        # ========================================= #


        self.CheckVar1 = tk.IntVar()
        check = tk.Checkbutton(checkFrame , text="Show interpreted lines", variable = self.CheckVar1)
        check.pack(side=tk.LEFT)


        self.CheckVarMode = tk.IntVar()
        showMax = tk.Checkbutton(checkFrame, text="Maximum/Mode?", variable = self.CheckVarMode)
        showMax.pack(side=tk.RIGHT)



        self.CheckVarpmap = tk.IntVar()
        angTik = tk.Checkbutton(checkFrame, text="Show prob map?", variable = self.CheckVarpmap)
        angTik.pack(side=tk.LEFT)


        # ========================================= #
        # ============== BUTTON FRAME ============= #
        # ========================================= #

        tk.Button(buttonFrame, text='Show', command=self.showValues, bg="blue", bd=4, fg="blue").pack(side=tk.LEFT)
        tk.Button(buttonFrame, text='Open', command=self.openImage, bg="red", bd=4, fg="red").pack(side=tk.LEFT)
        tk.Button(buttonFrame, text='Compute Error', command=self.plotEvaluation, bg="black", bd=4, fg="black").pack(side=tk.LEFT)
        tk.Button(buttonFrame, text='Run DBSCAN', command=self.showclusters, bg="green", bd=4, fg="green").pack(
            side=tk.LEFT)
        tk.Button(buttonFrame, text='Convert to Lines', command=self.convert2lines, bg="yellow", bd=4, fg="yellow").pack(
            side=tk.RIGHT)



        # ========================================= #
        # ============== IMAGE FRAME ============== #
        # ========================================= #


        im = Image.fromarray(self.bg)
        im = im.resize((self.width2, self.height2))
        img = ImageTk.PhotoImage(im)
        self.panel = tk.Label(self.master, image=img)
        self.panel.pack()


        self.RUN = True
        self.master.mainloop()





