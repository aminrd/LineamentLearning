
# Utility file: Contains functions to show matrices, normalizations, ....
__author__ = "Amin Aghaee"
__copyright__ = "Copyright 2018, Amin Aghaee"
import numpy as np
from PIL import Image, ImageDraw
from globalVariables import *



def slideBar(pct = 10.0, totalLength = 30):
    [p1,p2] = [(pct*totalLength)//100 , ((100-pct)*totalLength)//100]
    return '{'+ '='*int(p1) +'#'+ '-'*int(p2) +'}'



def myNormalizer(matrix):
    xmax, xmin = matrix.max(), matrix.min()

    if xmax == xmin:
        if xmax == 0:
            return np.zeros(np.array(matrix).shape)
        else:
            return np.ones(np.array(matrix).shape)

    [XMAX, XMIN] = [xmax, xmin]

    if xmin < -10000.0:
        idxMin = matrix == xmin
        matrix[idxMin] = 0.0
        XMIN = matrix.min()
    if xmax > 10000.0:
        idxMax = matrix == xmax
        matrix[idxMax] = 0.0
        XMAX = matrix.max()

    matrix = (matrix - XMIN) / (XMAX - XMIN)

    if xmin < -10000.0:
        matrix[idxMin] = -100.0
    if xmax > 10000.0:
        matrix[idxMax] = 100.0

    return matrix



def rotateWithMap(mat, rmap, map_type = 'r2m', dim = 1):

    MODE_VALUE = 100000
    matrix = np.array(mat)
    newMat = np.zeros(matrix.shape)

    if dim == 1:
        [_x, _y] = matrix.shape
    else:
        [_x, _y] = matrix[:,:,0].shape


    if map_type.__eq__('r2m'):

        if dim == 1:
            flagMat = np.zeros(matrix.shape)
        else:
            flagMat = np.zeros(matrix[:, :, 1].shape)


        for i in range(_x):
            for j in range(_y):
                val = rmap[i][j]
                x0 = val // MODE_VALUE
                y0 = val %  MODE_VALUE

                if x0 >= _x or y0 >= _y or x0 < 0 or y0 <0:
                    continue

                if dim == 1:
                    newMat[x0][y0] = matrix[i][j]
                else:
                    newMat[x0,y0,:] = matrix[i,j,:]

                flagMat[x0][y0] = 1

        for i in range(1,_x-1):
            for j in range(1, _y - 1):
                if flagMat[i][j] == 0:
                    if dim == 1:
                        newMat[i][j] = (newMat[i + 1][j] + newMat[i - 1][j] + newMat[i][j + 1] + newMat[i][j - 1]) // 4
                    else:
                        newMat[i,j,:] = (newMat[i-1,j,:] + newMat[i+1,j-1,:] + newMat[i,j+1,:] + newMat[i,j,:])//4


    elif map_type.__eq__('m2r'):
        for i in range(_x):
            for j in range(_y):
                val = rmap[i][j]

                x0 = val // MODE_VALUE
                y0 = val %  MODE_VALUE

                if x0 >= _x or y0 >= _y or x0 < 0 or y0 < 0:
                    continue

                if dim == 1:
                    newMat[i][j] = matrix[x0][y0]
                else:
                    newMat[i,j,:] = matrix[x0,y0,:]

    return newMat



def showMatrix(matrix , dim = 3, fname = FG+'DEFAULT.png', show = True):
    a = np.array(matrix)

    if a.min() < 0:
        a[np.where(a == a.min())] = 0

    if a.max() > 1:
        a[np.where(a == a.max())] = 1

    a = a * 255
    a = np.uint8(a)
    if dim == 3:
        img = Image.fromarray(a[:,:,0] , 'L')
    elif dim == 2:
        img = Image.fromarray(a[:, :], 'L')

    img.save(fname)
    if show==True:
        img.show()

    return img


def markPredictions(matrix, pmap, WIDTH = 3 , FILL = 128, fname = FG+'Default.png'):
    im = Image.fromarray(matrix)
    idx = np.where(pmap == 1)
    draw = ImageDraw.Draw(im)

    for k in range(len(idx[0])):
        [i,j] = [idx[0][k] , idx[1][k]]
        draw.line((j , i  , j , i  ), fill = FILL, width=WIDTH)

    im.save(fname)
    return im



def drawLines(matrix, idx , Y, WIDTH = 3 , FILL = 128, ws = 50, fname = FG+'lines.png', threshold = 0.51):
    # ws = window size, how many pixels go left or right in x-axis

    im = Image.fromarray(matrix)
    draw = ImageDraw.Draw(im)

    for k in range(len(idx[0])):
        [i,j] = [idx[0][k] , idx[1][k]]
        if Y[k] >= threshold:
            draw.line((j , i - ws , j , i + ws ), fill = FILL, width=WIDTH)


    im.save(fname)
    return np.asanyarray(im)


def probMap(shape,idx, Y):
    pmap = np.zeros(shape)
    for k in range(len(idx[0])):
        pmap[idx[0][k] , idx[1][k]] = Y[k]
    return pmap

def pmapCutoff(pmap, threshold = 0.5):
    p = np.zeros(pmap.shape)
    p[ np.where(pmap >= threshold) ] = 1
    return p


def modeIndex(M):
    '''Gets N 2D probability maps and
    returns maximum index of those values'''
    matrix = np.array(M)
    nonIndex = np.where(matrix[:,:,0] == 0)

    result = -np.ones((matrix.shape[0], matrix.shape[1]))
    mx = np.array(result)

    for d in range(matrix.shape[2]):
        mx = np.maximum(mx, matrix[:,:,d])

    for d in range(matrix.shape[2]):
        idx = np.where(mx == matrix[:,:,d])
        result[idx] = d

    result[nonIndex] = -1
    return result


def drawLinesSlope(matrix, idx , sloopes, WIDTH = 3 , FILL = 128, ws = 50, fname = FG+'Slopes.png', prelative = False, parray=None):
    '''ws = window size, how many pixels go left or right in x-axis'''

    slopes = np.tan(sloopes)
    im = Image.fromarray(matrix)
    draw = ImageDraw.Draw(im)

    if prelative==False:

        for k in range(len(idx[0])):
            [i,j] = [idx[0][k] , idx[1][k]]
            S = slopes[k]

            if np.abs(S) <= 1.0:
                [x1, y1] = [-ws, np.floor(-S * ws)]
                [x2, y2] = [ ws, np.floor( S * ws)]
            elif np.abs(S > 4.5):
                [x1, y1] = [np.floor(-ws / S), -ws]
                [x2, y2] = [np.floor( ws / S),  ws]
            else:
                continue
                #[x1,x2] = [0,0]
                #[y1,y2] = [0-ws, ws]

            draw.line((j + x1, i - y1, j + x2, i - y2), fill = FILL, width=WIDTH)

    else:

        parray = np.ndarray.flatten(parray)

        for k in range(len(idx[0])):
            [i, j] = [idx[0][k], idx[1][k]]
            S = slopes[k]
            _ws = int(np.ceil(ws * parray[k])) + 1

            if np.abs(S) <= 1.0:
                [x1, y1] = [-_ws, np.floor(-S * _ws)]
                [x2, y2] = [_ws, np.floor(S * _ws)]
            elif np.abs(S > 4.5):
                [x1, y1] = [np.floor(-_ws / S), -_ws]
                [x2, y2] = [np.floor(_ws / S), _ws]
            else:
                [x1, x2] = [0, 0]
                [y1, y2] = [0 - _ws, _ws]

            draw.line((j + x1, i - y1, j + x2, i - y2), fill=FILL, width=WIDTH)


    im.save(fname)
    return np.asanyarray(im)


def drawLinesWithEndingPoints(bg, lines, fname=FG+'lines.png', _width=5):
    # Format of lines: array of pairs [P1,P2]
    # P1 = [x1,y1] , P2=[x2,y2]
    bg = np.uint8(bg)
    im = Image.fromarray(bg)
    draw = ImageDraw.Draw(im)

    for l in lines:
        draw.line((l[0][1], l[0][0], l[1][1], l[1][0]), fill=128, width=_width)


    im.save(fname)
    return im


def drawCurves(bg, curves, fname=FG+'curves.png', _width=5):
    # Each curve contains two lists [Xset, Yset]
    # Xset = [x1,x2,....] , Yset = [y1, y2, ...]

    bg = np.uint8(bg)
    im = Image.fromarray(bg)
    draw = ImageDraw.Draw(im)

    for c in curves:
        x = c[0]
        y = c[1]

        for i in range(len(x)-1):
            draw.line( (y[i], x[i], y[i+1], x[i+1]) , fill=128, width=_width )

    im.save(fname)
    return im





def colour2vec(colour = 'red'):
    if colour.__eq__('red'):
        return np.array([1,0,0])
    elif colour.__eq__('green'):
        return np.array([0, 1, 0])
    elif colour.__eq__('blue'):
        return np.array([0, 0, 1])
    elif colour.__eq__('yellow'):
        return np.array([1, 1, 0])
    elif colour.__eq__('white'):
        return np.array([1, 1, 1])
    elif colour.__eq__('blue'):
        return np.array([0, 0, 1])
    else:
        return np.array([0, 0, 0])



def getRandomColour(channel=3):
    return np.random.choice(range(10,255),channel)



def circular_mask(width = 5 , R = None):
    radius = (width - 1) / 2

    if R is None:
        R = radius

    Y, X = np.ogrid[:width, :width]
    distance = np.sqrt((Y - radius) ** 2 + (X - radius) ** 2)

    return distance <= R
