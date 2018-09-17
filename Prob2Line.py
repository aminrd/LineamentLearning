# Prob2Map
# Is a class that gets a probability map as an input and convert it to lines

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from Utility import *
import numpy.matlib



METHOD_OPTIONS = ['Linear', 'Curve', 'BestCurve']
METHOD = METHOD_OPTIONS[2]
DEGREELIST = [1,3,5]
DEGREE = 3


class prob2map:
    def __init__(self, pmap=None):

        if pmap is None:
            self.pmap = np.random.random((500,500))
        else:
            self.pmap = pmap

        self.SpecialColor = -1


    def window2line(self, patch, s):
        [x,y] = np.where(patch > 0)

        x -= s
        y -= s

        x = np.reshape(x,[x.shape[0],1])
        weight = patch[np.where(patch > 0)]

        lr = linear_model.RANSACRegressor()
        lr.fit(x,y,weight)

        return lr.estimator_.coef_[0]


    def getLines(self, pachsize = 17, cutoff = 0.3, mincut = 0.2):
        PMAP = np.array(self.pmap)
        PMAP[PMAP < mincut] = 0
        matrix = np.array(PMAP)

        s = (pachsize - 1) // 2

        lines = []
        while matrix.max() > cutoff:
            tmp = np.where( matrix >= matrix.max())
            ox = tmp[0][0]
            oy = tmp[1][0]

            matrix[ox-s:ox+s+1,oy-s:oy+s+1] = 0
            coeff = self.window2line(PMAP[ox-s:ox+s+1,oy-s:oy+s+1], s)

            lines.append([ox,oy, coeff])

        return lines


    def getClusters(self, cutoff = 0.3, eps = 0.3):
        PMAP = pmapCutoff(self.pmap, cutoff)
        X = np.transpose(np.where(PMAP > 0))
        db = DBSCAN(eps=eps, min_samples=20).fit(X)
        labels = db.labels_

        cmap = np.zeros(PMAP.shape)
        for i in range(len(labels)):
            cmap[ X[i][0] , X[i][1]] = labels[i]
        return np.int32(cmap)


    def showClusters(self, cmap, specnumber = -1):
        imageMap = np.zeros((cmap.shape[0], cmap.shape[1], 3))
        nclass = np.max(cmap)

        for c in range(1,nclass):

            if specnumber > 0 and c == specnumber:
                col = [255,0,0]
            else:
                col = getRandomColour(3)

            ind = np.where(cmap == c)
            for i in range(len(ind[0])):
                for j in range(3):
                    imageMap[ ind[0][i], ind[1][i], j ] = col[j]

        return np.uint8(imageMap)

    def mergeClusters(self, cmap, c1, c2):
        cmap2 = np.array(cmap)
        cmin = np.min([c1,c2])

        I1 = np.where(cmap2 == c1)
        cmap2[I1] = cmin

        I2 = np.where(cmap2 == c2)
        cmap2[I2] = cmin

        return cmap2

    def getClusterCentroid(self, cmap, c):
        [i,j] = np.where(cmap == c)

        I = np.mean(i)
        J = np.mean(j)

        return [I,J]

    def getClusterDistance(self, cmap, c1, c2, center=False):

        if center:
            # Compute distance of center points of each cluster
            [i1,j1] = self.getClusterCentroid(cmap, c1)
            [i2, j2] = self.getClusterCentroid(cmap, c2)

            return np.sqrt( (i1-i2)*(i1-i2) + (j1-j2)*(j1-j2) )
        else:
            # Compute distance of nearest points in two clusters:
            D = np.inf
            [I1, J1] = np.where(cmap == c1)
            [I2, J2] = np.where(cmap == c2)

            k = len(I1)
            l = len(I2)

            I1 = np.matlib.repmat(np.array(I1), l, 1)
            J1 = np.matlib.repmat(np.array(J1), l, 1)
            I2 = np.matlib.repmat(np.array(I2), k, 1).transpose()
            J2 = np.matlib.repmat(np.array(J2), k, 1).transpose()

            Dmatrix = np.square((I1-I2)) + np.square((J1-J2))
            D = np.min(Dmatrix)
            return np.sqrt(D)



    def sortClustesrsByDistance(self, cmap, cbase):
        clist = np.unique(cmap)[1:]
        d = np.zeros_like(clist)

        for c in range(len(clist)):
            d[c] = self.getClusterDistance(cmap, cbase, clist[c], center=True)

        args = np.argsort(d)
        return clist[args]




    def getClusterLinearError(self, cmap, c):
        centroid = self.getClusterCentroid(cmap, c)
        centroid = np.uint64(centroid)

        ind = np.where(cmap == c)
        return self.convertCluster2Line(centroid, ind, getError=True)

    def getClusterCurveError(self, cmap, c, degree=3):
        centroid = self.getClusterCentroid(cmap, c)
        centroid = np.uint64(centroid)

        ind = np.where(cmap == c)
        return self.convertCluster2Curve(centroid, ind, degree, getError=True)

    def getClusterBestCurveError(self, cmap, c, degree=None):

        if degree is None:
            degree = [1,3,5]

        centroid = self.getClusterCentroid(cmap, c)
        centroid = np.uint64(centroid)

        ind = np.where(cmap == c)
        return self.convertCluster2BestCurve(centroid, ind, degree, getError=True)



    def convertCluster2Curve(self, center, cluster, degree=3, getError=False):
        # Cluster : 2xN array [[x1,x2,...],[y1,y2,...]]
        # Center  : [X0,Y0]
        X = np.array(cluster[0])
        Y = np.array(cluster[1])

        X -= center[0]
        Y -= center[1]
        X = np.reshape(X, [X.shape[0], 1])


        model = make_pipeline(PolynomialFeatures(degree), Ridge())
        model.fit(X,Y)


        if getError:
            yhat = model.predict(X)
            return mean_squared_error(Y, yhat)

        else:
            xset = np.unique(X)
            X1 = np.reshape(xset, [xset.shape[0], 1])
            yset = model.predict(X1)

            return [ xset + center[0], yset + center[1] ]

    def convertCluster2BestCurve(self, center, cluster, degree=None, getError=False):
        # Cluster : 2xN array [[x1,x2,...],[y1,y2,...]]
        # Center  : [X0,Y0]

        if degree is None:
            degree = [1,3,5]

        X = np.array(cluster[0])
        Y = np.array(cluster[1])

        X -= center[0]
        Y -= center[1]
        X = np.reshape(X, [X.shape[0], 1])


        Emin = np.inf
        BestModel = None

        for d in degree:
            model = make_pipeline(PolynomialFeatures(d), Ridge())
            model.fit(X,Y)

            yhat = model.predict(X)
            err = mean_squared_error(Y, yhat)

            if err < Emin:
                BestModel = model
                Emin = err


        if getError:
            return Emin

        else:
            xset = np.unique(X)
            X1 = np.reshape(xset, [xset.shape[0], 1])
            yset = model.predict(X1)

            return [ xset + center[0], yset + center[1] ]



    def convertCluster2Line(self, center, cluster, getError=False):
        # Cluster : 2xN array [[x1,x2,...],[y1,y2,...]]
        # Center  : [X0,Y0]

        X = np.array(cluster[0])
        Y = np.array(cluster[1])

        X -= center[0]
        Y -= center[1]
        X = np.reshape(X, [X.shape[0], 1])

        lr = linear_model.RANSACRegressor()
        lr.fit(X,Y)

        if getError:
            yhat = lr.predict(X)
            return mean_squared_error(Y, yhat)

        else:

            Xmin = [X.min() + center[0], lr.predict(X.min()) + center[1]]
            Xmax = [X.max() + center[0], lr.predict(X.max()) + center[1]]

            return [center, Xmin, Xmax]



    def doIteration(self, cmap, crange = 5, threshold = 0.8):
        clusterList = np.unique(cmap)[1:]

        cnt = self.getClusterSizes(cmap, clusterList)
        cnt_max = np.max(cnt)
        cnt = cnt_max - cnt
        _p = cnt / np.sum(cnt)


        cbase = np.random.choice(clusterList, p=_p)


        if DEBUG_MODE:
            print("===================")
            print("Chose cluster {} with size {}".format(cbase, len(np.where(cmap == cbase)[0])))

        if cbase <= 0:
            return cmap

        cnearby = self.sortClustesrsByDistance(cmap, cbase)


        if METHOD.__eq__("Linear"):
            E1 = self.getClusterLinearError(cmap, cbase)
        elif METHOD.__eq__("BestCurve"):
            E1 = self.getClusterBestCurveError(cmap, cbase, degree=DEGREELIST)
        else:
            E1 = self.getClusterCurveError(cmap, cbase, degree=DEGREE)



        EMIN = np.inf
        BestMerge = cmap
        self.SpecialColor = -1
        Best_Desc = "No Merge!"




        for i in range(crange):
            cprim = cnearby[i+1]

            if cprim <= 0:
                continue



            # Computing Error for other cluster
            if METHOD.__eq__("Linear"):
                E2 = self.getClusterLinearError(cmap, cprim)
            elif METHOD.__eq__("BestCurve"):
                E2 = self.getClusterBestCurveError(cmap, cprim, degree=DEGREELIST)
            else:
                E2 = self.getClusterCurveError(cmap, cprim, degree=DEGREE)



            cmerge = self.mergeClusters(cmap, cbase, cprim)



            # Computing Error if merge these two clusters
            if METHOD.__eq__("Linear"):
                Emerge = self.getClusterLinearError(cmerge, np.min([cprim, cbase]))
            elif METHOD.__eq__("BestCurve"):
                Emerge = self.getClusterBestCurveError(cmerge, np.min([cprim, cbase]), degree=DEGREELIST)
            else:
                Emerge = self.getClusterCurveError(cmerge, np.min([cprim, cbase]), degree=DEGREE)




            if Emerge < EMIN and E1+E2 >= Emerge * threshold:
                EMIN = Emerge
                BestMerge = cmerge
                self.SpecialColor = np.min([cbase, cprim])

                if DEBUG_MODE:
                    Best_Desc = "--- Merged {} and {}".format(cbase, cprim)



        if DEBUG_MODE:
            print(Best_Desc)
            print("--- Total number of clusters = {}".format(len(np.unique(BestMerge))))

        return BestMerge






    def makeConversion(self, cutoff = 0.3, eps = 0.3):
        cmap = self.getClusters(cutoff, eps)
        return self.convertClustersToLines(cmap)



    def convertClustersToLines(self, cmap):
        nclass = np.unique(cmap)
        lines = []

        # Each line consist of two pairs [[x1,y1],[x2,y2]]
        for c in nclass:

            if c <= 0:
                continue

            ind = np.where(cmap == c)
            center = [np.uint64(np.mean(ind[0])) , np.uint64(np.mean(ind[1]))]
            L = self.convertCluster2Line(center, ind)
            L = np.uint64(L)

            lines += [ [ L[1] , L[2]] ]

        return lines


    def convertClustersToCurves(self, cmap, degree = 3):
        nclass = np.unique(cmap)
        curves = []

        for c in nclass:
            if c<=0:
                continue

            ind = np.where(cmap == c)
            center = [np.uint64(np.mean(ind[0])) , np.uint64(np.mean(ind[1]))]
            C = self.convertCluster2Curve(center, ind, degree)
            C = np.uint64(C)

            curves += [C]

        return curves


    def convertClustersToBestCurves(self, cmap, degree = None):

        if degree is None:
            degree = [1,3,5]

        nclass = np.unique(cmap)
        curves = []

        for c in nclass:

            if c <= 0:
                continue

            ind = np.where(cmap == c)
            center = [np.uint64(np.mean(ind[0])) , np.uint64(np.mean(ind[1]))]
            C = self.convertCluster2BestCurve(center, ind, degree)
            C = np.uint64(C)

            curves += [C]

        return curves





    def drawLines(self, pachsize = 17, cutoff = 0.3, mincut = 0.2):

        lines = self.getLines(pachsize, cutoff, mincut)
        lines = np.array(lines)

        self.LINES = lines

        bg = np.zeros([self.pmap.shape[0], self.pmap.shape[1], 3])
        bg[:,:,1] = np.floor(self.pmap * 255)
        bg = np.uint8(bg)

        IDX = np.transpose(lines[:,0:2])
        slopes = np.arctan( -lines[:,2] )
        limage = drawLinesSlope(bg , IDX,slopes, ws = pachsize)

        Image.fromarray(limage).show()



    def getClusterSizes(self, cmap, nclass):

        cnt = np.zeros_like(nclass)

        for i in range(len(nclass)):
            c = nclass[i]
            ind = np.where( cmap == c )
            cnt[i] = len(ind[0])

        return cnt


    def runMethod(self, coeff = 0.5, eps = 1, iteration = 100):

        BG1 = Image.open("TMP1.png")
        BG2 = Image.open("TMP2.png")

        CurveList = []

        cmap = self.getClusters(coeff, eps)

        for t in range(iteration):

            if len(np.unique(cmap)) < 20:
                break

            if DEBUG_MODE:
                print("Iteration = {}".format(t))


            cmap = self.doIteration(cmap, crange=8, threshold=1)

            img = self.showClusters(cmap, self.SpecialColor)
            im = Image.fromarray(img)
            im.save('./applet_images/cluster/{}.png'.format(t))


            #Convert this state to lines:
            if METHOD.__eq__("Linear"):
                lines = self.convertClustersToLines(cmap)
                CurveList += [lines]

            elif METHOD.__eq__("BestCurve"):
                curves = self.convertClustersToBestCurves(cmap, DEGREELIST)
                CurveList += [curves]
            else:
                curves = self.convertClustersToCurves(cmap, DEGREE)
                CurveList += [curves]


            bg = np.zeros((self.pmap.shape[0], self.pmap.shape[1], 3))
            bg = np.uint8(bg)


            bgList = [bg, BG1, BG2]
            for i in range(len(bgList)):
                B = bgList[i]
                if METHOD.__eq__("Linear"):
                    im = drawLinesWithEndingPoints(B, lines)
                else:
                    im = drawCurves(B, curves)

                im.save('./applet_images/cluster/{}_{}_l.png'.format( i,t))

        np.save("Curves.npy" , CurveList)



