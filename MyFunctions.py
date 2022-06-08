import os
import glob
import csv
import math
import pandas as pd
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy import signal
from scipy.signal import argrelextrema
from scipy.fft import fft, ifft, fftfreq
import cufflinks as cf
from IPython.display import display,HTML

#import pywt

#Define units
cm = 1/2.54  # centimeters in inches

#Define colors and styles
#ojo quitar el rojo al inicio, es solo para SW
#'blue', 'red','blue',
colorLegend = ['black', 'blue', 'orangered', 'green', 'red', 'blueviolet', 'brown', 'coral',
                   'cornflowerblue', 'crimson', 'darkblue', 'darkcyan', 'darkmagenta', 'darkorange', 'darkred',
                   'darkseagreen', 'darkslategray', 'darkviolet', 'deeppink', 'deepskyblue', 'dodgerblue',
                   'firebrick', 'forestgreen', 'fuchsia', 'gold', 'goldenrod', 'green', 'hotpink', 'indianred',
                   'indigo', 'purple', 'rebeccapurple', 'saddlebrown', 'salmon',
                   'seagreen', 'sienna', 'slateblue', 'steelblue', 'violet', 'yellowgreen', 'aqua', 'aquamarine',
                   'darkgoldenrod', 'darkorchid', 'darkslateblue', 'darkturquoise', 'greenyellow', 'navy',
                   'palevioletred', 'royalblue', 'sandybrown']
lineStyle = ["solid", "dotted", "dashed", "dashdot"]
Ls = len(lineStyle)

def DownSample(x,m):
    xDown = []
    i = 0
    while i <= len(x):
        if (i % m )==0:
             xDown.append(x[i])
        i = i+1
    xDown = np.array(xDown)
    return(xDown)

def LoadSignal(file,jump, xRange, yRange):
    #jump especifica cuantas filas se salta
    with open(file, newline='') as file:
        reader = csv.reader(file, delimiter =',')
        for k in range(jump):
            next(reader)
        xi = []; yi = []
        for row in reader:
            auxX = float(row[0])
            auxY = float(row[1])
            if (auxX >= xRange[0] and auxX <= xRange[1]):
                xi.append(auxX)
                if auxY < yRange[0]:
                    auxY = yRange[0]
                if auxY > yRange[1]:
                    auxY = yRange[1]
                yi.append(auxY)
        xi = np.array(xi)
        yi = np.array(yi)
    return [xi,yi]

def LoadParam(file, jump):
    with open(file, newline='') as file:
        reader = csv.reader(file, delimiter =',')
        for k in range(jump):
            next(reader)
        files, params = [], []
        for row in reader:
            files.append(row[0])
            params.append(row[1])
    return [files, params]

def ReadFolderTx(files, yASE, xRange, yRange):
    #yASE is np array
    x,Tx,L = [], [], []
    filesCSV = glob.glob('*.CSV')
    NOF = len(files)
    for i in range(NOF):
        sufix ="0" + str(files[i]) + ".CSV"
        fileName =  [this for this in filesCSV if this.startswith("W") and this.endswith(sufix)]
        #np arrays
        [xi, yi] = LoadSignal(fileName[0], 29, xRange, yRange)
        Txi = yi-yASE
        x.append(xi)
        Tx.append(Txi)
        L.append(len(xi))
    return [x, Tx, L]

def ReadFolderStability(fileInit, xRange, yRange, param):
    #Read files (only xRange interval)
    x = []; y = []; L = [];
    NOF =len(param) # número de columnas
    for i in range(0, NOF, 4):
        if fileInit + i  < 10:
             file = 'W00' + str(fileInit + i) + '.CSV'
        else:
             if fileInit + i  < 100:
                file = 'W00' + str(fileInit + i) + '.CSV'
             else:
                file = 'W0' + str(fileInit + i) + '.CSV'
        [xi, yi] = LoadSignal(file, 29, xRange, yRange)
        x.append(xi)
        y.append(yi)
        L.append(len(xi))
    return [x,y,L]

def CreateDataFrame(filepath, dfParam):
    files = dfParam["fileName"].tolist()
    param = dfParam["param"].tolist()
    os.chdir(filepath)
    filesCSV = glob.glob('*.CSV')
    for i in range(len(param)):
        sufix = "0" + str(files[i]) + ".CSV"
        fileName = [this for this in filesCSV if this.startswith("W") and this.endswith(sufix)]
        var=fileName[0]
        if i == 0:
            df = pd.read_csv(fileName[0], header=22, names=["Wavelength", str(param[i])])
        else:
            dfaux = pd.read_csv( fileName[0], header=22, names=["xaux", str(param[i])])
            df[str(param[i])] = dfaux[str(param[i])]
    return df

def List2df(x,y,L,param):
#unifico la longitud de las listas para volverlas dataframe
    NOF = len(param)
    Lmax = max(L)
    for i in range(NOF):
        Li = L[i]
        if Li < Lmax:
            xMissed = (Lmax - Li)
            noisyPAd = np.random.normal(-0.1, 0.2, xMissed)
            nP= noisyPAd.tolist()
            yP = [y[i][Li-1]] * xMissed
            yPad = [sum(n) for n in zip(nP,yP)]
            auxList = y[i] + yPad
            y[i] = auxList
            if i == 0:
                xStep = round(x[i][1] - x[i][0], 4)
                x0 = x[i][Li-1]
                xPad = [x0 + x * xStep for x in range(0, xMissed)]
                x[i] = x[i] + xPad
                df = pd.DataFrame(list(zip(x[i], y[i])), columns=['Wavelength', str(param[i])])
            else:
                df[str(param[i])] = y[i]
        else:
            if i == 0:
                df = pd.DataFrame(list(zip(x[i], y[i])), columns=['Wavelength', str(param[i])])
            else:
                df[str(param[i])] = y[i]
    return df

#kymax, ymax[kymax], FWHM[kymax] = SelectLaserSignal(x,y,L)
def SelectLaserSignal(x,y,L):
    LL = len(L)
    x1 = np.empty(LL)
    x2 = np.empty(LL)
    ymax = np.empty(LL)
    FWHM = np.empty(LL)
    #Hallar todos y elegir el mayoor pico de potencia
    for i in range(LL):
        xi = np.array(x[i])
        yi = np.array(y[i])
        x1[i], x2[i], ymax[i], FWHM[i] = Calculate_yMax_FWHM(xi, yi)
    kymax = np.argmax(ymax)
    return kymax, ymax[kymax], FWHM[kymax]

# lambdaPeak, peak = StabVar(x,y)
def StabVar(x,y):
    NOF = len(x)
    kPeak = np.empty(NOF, dtype=int)
    lambdaPeak = np.empty(NOF)
    peak = np.empty(NOF)
    #Hallar todos y elegir el mayoor pico de potencia
    for i in range(NOF):
        xi = np.array(x[i])
        yi = np.array(y[i])
        peak[i] = np.max(yi)
        kPeak[i] = np.argmax(yi)
        lambdaPeak[i] = xi[kPeak[i]]
    return lambdaPeak, peak

def List2dfXY(x,y,L,param):
#unifico la longitud de las listas para volverlas dataframe
    NOF = len(param)
    Lmax = max(L)
    for i in range(NOF):
        Li = L[i]
        if Li < Lmax:
            xMissed = (Lmax - Li)
            noisyPAd = np.random.normal(-0.1, 0.2, xMissed)
            nP= noisyPAd.tolist()
            yP = [y[i][Li-1]] * xMissed
            yPad = [sum(n) for n in zip(nP,yP)]
            auxList = y[i] + yPad
            y[i] = auxList
            if i == 0:
                xStep = round(x[i][1] - x[i][0], 4)
                x0 = x[i][Li-1]
                xPad = [x0 + x * xStep for x in range(0, xMissed)]
                x[i] = x[i] + xPad
                df = pd.DataFrame(list(zip(x[i], y[i])), columns=['Wavelength', str(param[i])])
            else:
                df[str(param[i])] = y[i]
        else:
            if i == 0:
                df = pd.DataFrame(list(zip(x[i], y[i])), columns=['Wavelength', str(param[i])])
            else:
                df[str(param[i])] = y[i]
    return df

def PlotParamInt(df):
    col_names = df.columns.values[1:]
    paramStr = col_names.tolist()
    NOF = len(paramStr)
    colorLegend =[ ' black', ' blue', ' blueviolet', ' brown', ' cadetblue', ' chocolate', ' coral',
                    ' cornflowerblue', ' crimson', ' darkblue', ' darkcyan', ' darkmagenta', ' darkorange', ' darkred',
                    ' darkseagreen', ' darkslategray', ' darkviolet', ' deeppink', ' deepskyblue', ' dodgerblue',
                    ' firebrick', ' forestgreen', ' fuchsia', ' gold', ' goldenrod', ' green', ' hotpink', ' indianred',
                    ' indigo', ' orangered', ' purple', ' rebeccapurple', ' red', ' saddlebrown', ' salmon',
                    ' seagreen', ' sienna', ' slateblue', ' steelblue', ' violet', ' yellowgreen', 'aqua', 'aquamarine',
                    'darkgoldenrod', 'darkorchid', 'darkslateblue', 'darkturquoise', 'greenyellow', 'navy',
                    'palevioletred', 'royalblue', 'sandybrown']

    A = df["Wavelength"].tolist()
    fig1 = make_subplots()
    for i in range(NOF):
        B = df[paramStr[i]]
        fig1.add_trace(go.Scatter(
            x=A,
            y=B,
            legendgroup = 'lgd'+str(i),
            name=paramStr[i],
            mode="lines",
            line_color=colorLegend[i],
            ))
    fig1.update_layout(height=800, width=1800)
    #fig1.show() descomentar lo uestra en una nueva pestaña
    return fig1

def PlotParamListsInt(x,y,param):
    NOF = len(param)
    colorLegend =[ ' black', ' blue', ' blueviolet', ' brown', ' cadetblue', ' chocolate', ' coral',
                    ' cornflowerblue', ' crimson', ' darkblue', ' darkcyan', ' darkmagenta', ' darkorange', ' darkred',
                    ' darkseagreen', ' darkslategray', ' darkviolet', ' deeppink', ' deepskyblue', ' dodgerblue',
                    ' firebrick', ' forestgreen', ' fuchsia', ' gold', ' goldenrod', ' green', ' hotpink', ' indianred',
                    ' indigo', ' orangered', ' purple', ' rebeccapurple', ' red', ' saddlebrown', ' salmon',
                    ' seagreen', ' sienna', ' slateblue', ' steelblue', ' violet', ' yellowgreen', 'aqua', 'aquamarine',
                    'darkgoldenrod', 'darkorchid', 'darkslateblue', 'darkturquoise', 'greenyellow', 'navy',
                    'palevioletred', 'royalblue', 'sandybrown']
    fig1 = make_subplots()
    for i in range(NOF):
        A = x[i]
        B = y[i]
        fig1.add_trace(go.Scatter(
            x=A,
            y=B,
            legendgroup = 'lgd'+str(i),
            name=str(param[i]),
            mode="lines",
            line_color=colorLegend[i],
            ))
    #fig1.show()
    return


def Dist2Curv(param):
    curv = np.empty(len(param),dtype=int)
    L = 0.15 #en metros
    param = np.array(param)
    p2 = np.power(param*1e-6, 2)    #en m
    curv = np.around(2 * param/(p2+L*L), 0)
    curv = curv.astype(np.int)
    #curv = 2 * param / (p2 + L * L)
    #curv en 1/m
    return curv

def PlotSignalInt(x,y):
    fig = make_subplots(1)
    fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line_color='black',
            ))
    fig.show()

#paramSel = SelectingParam(param, indexSel)
def SelectingParam(param, indexSel):
    paramSel = []
    for i in range(len(indexSel)):
        k = indexSel[i]
        paramSel.append(param[k])
    return paramSel

# listSel = SelectingList(list, indexSel)
def SelectingList(list, indexSel):
    listSel = []
    for i in range(len(indexSel)):
        k = indexSel[i]
        listSel.append(list[k])
    return listSel


def SelectDataFrame(df,xRange, paramSel):
    NOF = len(paramSel)
    paramStr = []
    x = df[(df['Wavelength'] >= xRange[0]) & (df['Wavelength'] <= xRange[1])]['Wavelength'].tolist()
    df1 = pd.DataFrame()
    df1['Wavelength'] = x
    for i in range(NOF):
        paramStr.append(str(paramSel[i]))
        yi = df[(df['Wavelength'] >= xRange[0]) & (df['Wavelength'] <= xRange[1])][paramStr[i]].tolist()
        df1[paramStr[i]] = yi
    return df1

def PlotTxRef(x1,y1,xRange):
    ymax = 0
    ymin = math.floor(min(y1))
    fig, ax = plt.subplots()
    ax.plot(x1, y1, '-b')
    # Extinction ratio
    xa = 1542.2 #ymax
    ya = -1.43
    xb = 1547.9 #ymin
    yb = -12.31
    xy = (xa, ya)
    xytext = (xa, yb)
    ax.annotate('', xy=xy, xycoords='data', xytext = xytext,
                    textcoords = 'data', arrowprops = dict(arrowstyle="<->", ec = "k", shrinkA = 0, shrinkB = 0))
    ER = (ya - yb)
    plt.text(xa - 3, yb - 2, 'ER\n' + str(round(abs(ER), 1)) + 'dB', size=8)
    # Dashed lines Extinction ratio
    xy = (xa, yb)
    xytext = (xb, yb)
    ax.annotate('', xy=xy, xycoords='data', xytext=xytext,
                textcoords='data', arrowprops=dict(arrowstyle="-", linestyle="--", ec="k", shrinkA=0, shrinkB=0))
    # Annotate FSR
    xa = 1553
    ya = -3.369
    xb = 1563.8
    yb = -4.221
    xy = (xa, ya + 0.2)
    xytext = (xb, ya + 0.2)
    ax.annotate('', xy=xy, xycoords='data', xytext = xytext,
                    textcoords = 'data', arrowprops = dict(arrowstyle="<->", ec = "k", shrinkA = 0, shrinkB = 0))

    # Dashed lines FSR
    xy = (xb, ya + 0.2)
    xytext = (xb, yb + 0.2)
    ax.annotate('', xy=xy, xycoords='data', xytext=xytext,
                textcoords='data', arrowprops=dict(arrowstyle="-", linestyle="--", ec="k", shrinkA=0, shrinkB=0))
    #FSR
    FSR = (xb - xa)
    plt.text((xa + xb) / 2 - 4, ya + 0.5, 'FSR=' + str(round(abs(FSR), 1)) + 'nm', size=8)
    #Setting Axis
    dx =10
    fig, ax = SettingAxis(fig, ax, xRange, [ymin, ymax], dx, 'Tx')
    #Save figure
    plt.savefig('TxRef', dpi=300,transparent=True, bbox_inches='tight')
    return

def FastFourier(x ,y):
    N = len(x)
    dx = round(x[1] - x[0],4)
    Fs = 1/dx
    Y = fft(y)
    sF = fftfreq(N, dx)[:N // 2]
    mY = 2.0 / N * np.abs(Y[0:N // 2])
    k1 = math.floor(N/Fs)
    return [sF[:k1], mY[:k1]]

def PlotFFT(x,y, xRange, yRange):
    sF, mY = FastFourier(x, y)
    fig, ax = plt.subplots()
    ax.plot(sF, mY, '-m')
    # Annotate SFi
    peaksIndex, properties = signal.find_peaks(mY, height=2)
    xa = sF[peaksIndex[0]]
    ya = mY[peaksIndex[0]]
    xy = (xa, ya)
    xytext = (xa, ya+2)
    ax.annotate('', xy=xy, xycoords='data',
                xytext=xytext, textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                ec="k",
                                shrinkA=0, shrinkB=0))
    plt.text(xa, ya+2, 'SF='+ str(round(xa, 4))+'1/nm', size=8)
    fig, ax = SettingAxis(fig, ax, xRange, yRange, dx, 'FFT')
    plt.savefig("FFT.png", dpi=350, bbox_inches="tight", transparent=True)
    return

def LineStyleChange(i,m, Ls):
    if i>=m:
        return lineStyle[(i-m) % Ls]
    else:
        return lineStyle[i % Ls]

def ColorLegendChange(i,m):
    if i>=m:
        return colorLegend[i-m]
    else:
        return colorLegend[i]


def PlotTxParam(df1, varControl, direction):
    col_names = df1.columns.values[1:]
    paramStr = col_names.tolist()
    NOF = len(paramStr)
    xmin = int(df1["Wavelength"].min())
    xmax = int(df1["Wavelength"].max())
    minYi = df1[paramStr].min()
    kmin = df1[paramStr].idxmin()
    fig, ax = plt.subplots()
    for i in range(NOF):
        #ax.plot(df1["Wavelength"], df1[paramStr[i]], color=colorLegend[i], linestyle=lineStyle[i % Ls], linewidth=0.8)
        m=4
        ax.plot(df1["Wavelength"], df1[paramStr[i]], color=colorLegend[i], linestyle=LineStyleChange(i, m,Ls), linewidth=0.8)
    lgd = plt.legend(paramStr, fontsize=6,
                            title=SelecTextVarControl(varControl),
                            title_fontsize=6,
                            bbox_to_anchor=(0, 1),
                            #loc='upper right',
                            loc='upper left',
                            fancybox=False)
    #SEt xlim,ylim
    ymin = min(minYi)
    ymax = 0

    #Arrow indicating the tunning direction
    xOrigin = (xmin+xmax)/2
    #xOrigin = 1554
    if kmin[2] > kmin[3]:
        xEnd = xOrigin + 1.5
    else:
        xEnd = xOrigin - 1.5

    #xEnd = xOrigin + 1.5 #ojo sta fija
    #yOrigin = -1
    #yOrigin = -16
    ax.annotate('', xy=(xOrigin, yOrigin), xycoords='data',
                xytext=(xEnd, yOrigin), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                ec="k",
                                shrinkA=0, shrinkB=0))
    dx = 5
    ax.text(1554.5, -15, "P", color=colorLegend[m])
    fig, ax = SettingAxis(fig, ax, [xmin, xmax], [ymin, ymax],dx, 'Tx')
    # Save figure
    plt.savefig('Tx'+varControl+direction+'.png', dpi=300, transparent=True, bbox_inches='tight',
                bbox_extra_artists=(lgd,))
    return

def PlotParamLists(x,y, param,varControl, direction):
    paramStr =[]
    NOF = len(param)
    min_x = np.empty(NOF)
    max_x = np.empty(NOF)
    min_y = np.empty(NOF)
    kmax = np.empty(NOF, dtype=int)
    xPeak = np.empty(NOF)
    fig, ax = plt.subplots()
    for i in range(NOF):
        xi = x[i]
        yi = y[i]
        paramStr.append(str(param[i]))
        min_x[i] = np.min(xi)
        max_x[i] = np.max(xi)
        min_y[i] = np.min(yi)
        kmax[i] = np.argmax(yi)
        xPeak[i] = xi[kmax[i]]
        ax.plot(x[i], y[i], color=colorLegend[i], linestyle=lineStyle[i % Ls], linewidth=1, label=str(param[i]))
    lgd = plt.legend(paramStr, fontsize=6,
                     title=SelecTextVarControl(varControl),
                     title_fontsize=6,
                     bbox_to_anchor=(0.65, 1),
                     loc='upper center',
                     fancybox=False)
    #ax.legend(prop={"size":6}, loc="upper right", bbox_to_anchor=(1.1, 1))
    xmin = min(min_x)
    xmax = max(max_x)
    #Arrow indicating the tunning direction
    xOrigin = xPeak[int(NOF/2)]
    xEnd = xPeak[int(NOF/2)+1]
    ymax = 0
    ymin = min(min_y)
    yOrigin = ymax-3
    ax.annotate('', xy=(xOrigin, yOrigin), xycoords='data',
                xytext=(xEnd, yOrigin), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                ec="k",
                                shrinkA=0, shrinkB=0))
    #fig, ax = SettingAxis(fig, ax, [xmin, xmax], [ymin, ymax], 'Pout')
    fig, ax = SettingAxis(fig, ax, [1550, 1570], [-80, -10], 'Pout')
    # Save figure
    """
    plt.savefig('LaserParam'+varControl+direction+'.png', dpi=300, transparent=True, bbox_inches='tight',
                bbox_extra_artists=(lgd,))
    """
    plt.savefig('LaserParam' + varControl + direction + '.png', dpi=300, transparent=True)
    return

def SelecTextVarControl(varControl):
    if varControl == 'Temp':
        title = r'$\mathrm{Temp.} (^{\circ}C)$'
    elif varControl == 'Curv':
        title = '$\mathrm{Curv} (m^{-1})$'
    elif varControl == 'Torsion':
        title = r'$\mathrm{Torsion} (^{\circ})$'
    else:
        title = ''
    return title

def SettingSWAxis(fig, ax, xRange, yRange, typeSignal):
    if typeSignal=='Tx':
        xLabel = 'Wavelength (nm)'
        yLabel = 'Transmission (dB)'
    elif typeSignal == 'Pout':
        xLabel = 'Wavelength (nm)'
        yLabel = ''
    elif typeSignal=='FFT':
        xLabel = 'Spatial frequency (1/nm)'
        yLabel = 'Magnitude (p.u)'
    elif typeSignal=='Lin':
        xLabel = ''
        yLabel = 'Wavelength (nm)'
    elif typeSignal=='PoutStab':
        xLabel = 'Time(s)'
        yLabel = 'Output power (dBm)'
    elif typeSignal=='lambdaStab':
        xLabel = 'Time(s)'
        yLabel = 'Wavelength (nm)'
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    else:
        xLabel= 'x'
        yLabel = 'y'
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    #plt.xticks(fontsize=8)
    #plt.yticks(fontsize=8)
    ax.set_xlabel(xLabel, fontsize=10)
    ax.set_ylabel(yLabel, fontsize=10)
    ax.set_xlim(xRange)
    ax.set_ylim(yRange)
    return fig, ax

def SettingAxis(fig, ax, xRange, yRange, dx, typeSignal):
    if typeSignal=='Tx':
        xLabel = 'Wavelength (nm)'
        yLabel = 'Transmission (dB)'
    elif typeSignal == 'Pout':
        xLabel = 'Wavelength (nm)'
        yLabel = 'Output power (dBm)'
    elif typeSignal=='FFT':
        xLabel = 'Spatial frequency (1/nm)'
        yLabel = 'Magnitude (p.u)'
    elif typeSignal=='Lin':
        xLabel = ''
        yLabel = 'Wavelength (nm)'
    elif typeSignal=='PoutStab':
        xLabel = 'Time(s)'
        yLabel = 'Output power (dBm)'
    elif typeSignal=='lambdaStab':
        xLabel = 'Time(s)'
        yLabel = 'Wavelength (nm)'
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    else:
        xLabel= 'x'
        yLabel = 'y'
    #ax.set_xticks(list(range(xRange[0], xRange[1]+1, 2))) #para el TEDFL parametrico
    #ax.set_xticks(list(range(xRange[0], xRange[1] + 1, 50))) #para el MZI vs  C parametrico
    #ax.set_xticks(list(range(xRange[0], xRange[1] + 1, 2))) #para linealidad por temepratura
    #ax.set_xticks(list(range(xRange[0], xRange[1] + 1, 100)))  # para linealidad por temepratura
    #ax.set_xticks(list(range(xRange[0], xRange[1] + 1, 5)))  # para SW Inc
    ax.set_xticks(list(range(xRange[0], xRange[1] + 1, dx)))  # para SW Inc
    #ax.set_xticks(list(range(xRange[0], xRange[1] + 1, 10)))  # para SW Dec
    #ax.set_xticks(list(range(xRange[0], xRange[1]+1, 4))) #para el TEDFL Temp+SW
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    ax.set_xlabel(xLabel, fontsize=10)
    ax.set_ylabel(yLabel, fontsize=10)
    ax.set_xlim(xRange)
    ax.set_ylim(yRange)
    auxWidth = 8.8 * cm
    auxHeight = 7.5 * cm
    figure = plt.gcf()
    figure.set_size_inches(auxWidth, auxHeight)
    plt.tight_layout()
    return fig, ax

def PlotLinTxMax(x,y, varControl, direction, xRange, yRange):
    NOF = len(x)
    fig, ax = plt.subplots()
    for i in range(NOF):
        m=4
        ax.scatter(x[i], y[i], color=colorLegend[i+m], marker='d')
    b = EstimateCoef(x, y)
    [Sr, St, r2] = Error(x, y, b)
    [xx, yy] = RegressionLin(xRange, b)
    ax.plot(xx, yy, color='k', linestyle='dashed')
    yProm = sum(yRange) / 2
    if b[1]>=0:#si la pendiente es positiva
        xText = sum(xRange) / 4
    else:#si la pendiente es negativa
        xText = sum(xRange) / 2.5
    yText = yRange[0] + 3 * (yRange[1] - yRange[0]) / 4
    if varControl=='Curv':
        unitControl='C'
    if varControl=='Temp':
        unitControl='T'
    plt.text(xText, yText,
             '$\lambda$' + '=' + str(round(b[1], 4)) + unitControl+'+' + str(round(b[0], 2)) + 'nm \n' + '$R^2$' + '=' + str(
                 round(r2, 4)), size=8)
    dx = 200
    fig, ax = SettingAxis(fig, ax, xRange, yRange, dx, 'Lin')
    ax.set_xlabel(SelecTextVarControl(varControl), fontsize=10)
    plt.savefig('TxLinMax' + varControl + direction + '.png', dpi=300, transparent=True, bbox_inches='tight')
    return

def PlotLinLaserMax(x,y, varControl, direction, xRange, yRange):
    NOF = len(x)
    fig, ax = plt.subplots()
    for i in range(NOF):
        #ax.scatter(x[i], y[i], color=colorLegend[i], marker='d')
        #ax.scatter(x[i], y[i], color=colorLegend[i+1], marker='d')
        ax.scatter(x[i], y[i], color=colorLegend[i+2], marker='d')
    b = EstimateCoef(x, y)
    [Sr, St, r2] = Error(x, y, b)
    [xx, yy] = RegressionLin(xRange, b)
    ax.plot(xx, yy, color='k', linestyle='dashed')
    yProm = sum(yRange) / 2
    if b[1]>=0:#si la pendiente es positiva
        xText = sum(xRange) / 4
    else:#si la pendiente es negativa
        xText = sum(xRange) / 3
    #xText = (max(x)+min(x))/3
    yText = max(y)
    if varControl=='Curv':
        unitControl='C'
    if varControl=='Temp':
        unitControl='T'
    plt.text(xText, yText,
             '$\lambda$' + '=' + str(round(b[1], 4)) + unitControl+'+' + str(round(b[0], 2)) + 'nm \n' + '$R^2$' + '=' + str(
                 round(r2, 4)), size=8)
    """
    # Arrow indicating the tunning direction
    xPoint = x[int(NOF/2)]
    yPoint = yText +2
    xOrigin = x[int(NOF/2)+1]
    ax.annotate('', xy=(xOrigin, yPoint), xycoords='data',
                    xytext=(xPoint, yPoint), textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    ec="k",
                                    shrinkA=0, shrinkB=0))
    """
    fig, ax = SettingAxis(fig, ax, xRange, yRange, dx, 'Lin')
    ax.set_xlabel(SelecTextVarControl(varControl), fontsize=10)
    plt.savefig('LaserLin' + varControl + direction + '.png', dpi=300, transparent=True, bbox_inches='tight')
    return

#Linear Regression
#[xArray, yArray] = LinearityMaxLists(x,y)
def LinearityMaxLists(x,y):
    #x, y, lists o f lists
    NOF = len(x)
    xArray = np.empty(NOF)
    yArray = np.empty(NOF)
    for i in range(NOF):
        xi = np.array(x[i])
        yi = np.array(y[i])
        yArray[i] = np.max(yi)
        ki = np.argmax(yi)
        xArray[i] = xi[ki]
    return xArray, yArray

def LinearityMinLists(x,y):
    #x, y, lists o f lists
    NOF = len(x)
    xArray = np.empty(NOF)
    yArray = np.empty(NOF)
    for i in range(NOF):
        xi = np.array(x[i])
        yi = np.array(y[i])
        yArray[i] = np.min(yi)
        ki = np.argmin(yi)
        xArray[i] = xi[ki]
    return xArray, yArray

#xArray, yArray = LinearityMax(df1):
def LinearityMax(df1):
    col_names = df1.columns.values[1:]
    paramStr = col_names.tolist()
    NOF = len(paramStr)
    xArray = np.empty(NOF)
    yArray = np.empty(NOF)
    for i in range(NOF):
        xi = df1['Wavelength'].tolist()
        yi = df1[paramStr[i]].tolist()
        yi = np.array(yi)
        yArray[i] = np.max(yi)
        ki = np.argmax(yi)
        xArray[i] = xi[ki]
    return xArray, yArray

def LinearityMin(df1):
    col_names = df1.columns.values[1:]
    paramStr = col_names.tolist()
    NOF = len(paramStr)
    xArray = np.empty(NOF)
    yArray = np.empty(NOF)
    for i in range(NOF):
        xi = df1['Wavelength'].tolist()
        yi = df1[paramStr[i]].tolist()
        yi = np.array(yi)
        yArray[i] = np.min(yi)
        ki = np.argmin(yi)
        xArray[i] = xi[ki]
    return xArray, yArray


def EstimateCoef(x, y):
    x = np.array(x)
    y = np.array(y)
    # number of observations/points
    n = np.size(x)
    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x
    return (b_0, b_1)

def Error(X,Y,a):
    Fun = np.ones(X.shape) * a[len(a) - 1]
    for i in range(len(a) - 2, -1, -1):
        Fun=Fun* X + a[i]
    Sr= sum( (Y-Fun)**2)
    K=sum(Y)/len(Y)
    St=sum( (Y-K)**2 )
    r2=(St-Sr)/St
    return ([Sr,St,r2])

def RegressionLin(xRange, a):
    No = 100
    # rango de x para graficarNo puntos
    xx = np.linspace(xRange[0], xRange[1], No);
    # halla el y por el interpolacion
    yy = np.ones(xx.shape) * a[len(a) - 1]
    for i in range(0, -1, -1):
        yy = yy * xx + a[i]
        # yy=sol[0]+sol[1]*xx+sol[2]*xx**2
    return ([xx,yy])

# Laser
# [x, y, L] = fu.ReadFolderPout(files, xRange, yRange)
def ReadFolderPout(files, xRange, yRange):
    #yASE is np array
    x,y,L = [], [], []
    filesCSV = glob.glob('*.CSV')
    NOF = len(files)
    for i in range(NOF):
        sufix ="0" + str(files[i]) + ".CSV"
        fileName =  [this for this in filesCSV if this.startswith("W") and this.endswith(sufix)]
        #np arrays
        [xi, yi] = LoadSignal(fileName[0], 29, xRange, yRange)
        x.append(xi)
        y.append(yi)
        L.append(len(xi))
    return [x, y, L]

# fu.PlotLaserParam(df1, varControl,direction)
def PlotLaserParam(df1, varControl,direction):
    colorLegend = ['black','blue','orangered','green', 'red','blueviolet', 'brown', 'coral',
                   'cornflowerblue', 'crimson', 'darkblue', 'darkcyan', 'darkmagenta', 'darkorange', 'darkred',
                   'darkseagreen', 'darkslategray', 'darkviolet', 'deeppink', 'deepskyblue', 'dodgerblue',
                   'firebrick', 'forestgreen', 'fuchsia', 'gold', 'goldenrod', 'green', 'hotpink', 'indianred',
                   'indigo', 'purple', 'rebeccapurple',  'saddlebrown', 'salmon',
                   'seagreen', 'sienna', 'slateblue', 'steelblue', 'violet', 'yellowgreen', 'aqua', 'aquamarine',
                   'darkgoldenrod', 'darkorchid', 'darkslateblue', 'darkturquoise', 'greenyellow', 'navy',
                   'palevioletred', 'royalblue', 'sandybrown']
    lineStyle = ["solid", "dotted", "dashed", "dashdot"]
    Ls = len(lineStyle)
    # legend title
    col_names = df1.columns.values[1:]
    paramStr = col_names.tolist()
    NOF = len(paramStr)
    xmin = int(df1["Wavelength"].min())
    xmax = int(df1["Wavelength"].max())
    yimin= df1[paramStr].min()
    yimax = df1[paramStr].max()
    kmax = df1[paramStr].idxmax()
    ymax = -10
    ymin = min(yimin)
    fig, ax = plt.subplots()
    for i in range(NOF):
        plt.plot(df1["Wavelength"], df1[paramStr[i]], color=colorLegend[i],linestyle=lineStyle[i%Ls], linewidth=0.8)
    lgd = plt.legend(paramStr, fontsize=6,
                     title=SelecTextVarControl(varControl),
                     title_fontsize=6,
                     bbox_to_anchor=(1.01, 1),
                     #bbox_to_anchor=(0, 1),
                     loc='upper right',
                     #loc='lower right',
                     #loc='upper left',
                     fancybox=False)
    # Arrow indicating the tunning direction
    xOrigin = (xmin + xmax) / 2
    if kmax[0] > kmax[2]:
        xEnd = xOrigin + 1.5
    else:
        xEnd = xOrigin - 1.5
    yOrigin = (ymax+ max(yimax))/2
    """
    ax.annotate('', xy=(xOrigin, yOrigin), xycoords='data',
                xytext=(xEnd, yOrigin), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                ec="k",
                                shrinkA=0, shrinkB=0))
    """
    fig, ax = SettingAxis(fig, ax, [1520, 1570], [-80, -10], dx,  'Pout')
    #fig, ax = SettingAxis(fig, ax, [xmin, xmax], [-80, -10], 'Pout')
    plt.savefig('LaserParam&SW' + varControl + direction+'.png', dpi=300, transparent=True, bbox_inches='tight',
                bbox_extra_artists=(lgd,))
    return

def SelectLaserSignal(x,y,L):
    LL = len(L)
    xmax = np.empty(LL)
    ymax = np.empty(LL)
    x1 = np.empty(LL)
    x2 = np.empty(LL)
    FWHM = np.empty(LL)
    #Hallar todos y elegir el mayoor pico de potencia
    for i in range(LL):
        xi = np.array(x[i])
        yi = np.array(y[i])
        xmax[i], ymax[i], x1[i], x2[i], FWHM[i] = Cal_xyMax_x3dB_FWHM(xi, yi)
    kymax = np.argmax(ymax)
    return kymax, ymax[kymax], FWHM[kymax]

#xmax, ymax, x[k1], x[k2],FWHM = fu.Cal_xyMax_x3dB_FWHM(x, y)
def Cal_xyMax_x3dB_FWHM(x, y):
    x = np.array(x)
    y = np.array(y)
    kmax = np.argmax(y)
    xmax = x[kmax]
    ymax = y[kmax]
    y3dB = ymax - 3
    d = np.asarray(np.where((y - y3dB) > 0))
    k1 = d[0, 0]-1
    k2 = d[0, -1]+1
    FWHM = x[k2] - x[k1]
    return xmax, ymax, x[k1], x[k2], FWHM

# k1, k2,FWHM = fu.Cal_k3dB_FWHM(x, y)
def Cal_k3dB_FWHM(x, y):
    x = np.array(x)
    y = np.array(y)
    kmax = np.argmax(y)
    xmax = x[kmax]
    ymax = y[kmax]
    y3dB = ymax - 3
    d = np.asarray(np.where((y - y3dB) > 0))
    k1 = d[0, 0]-1
    k2 = d[0, -1]+1
    FWHM = x[k2] - x[k1]
    return k1, k2, FWHM

#SMSR, kPeaks, kRef = CalculateSMSRall(x, y, prom, dist)
def CalculateSMSRall(x, y, prom, dist):
    x = np.array(x)
    y = np.array(y)
    SMSR = []
    kPeaks = []
    kRef = []
    #Find all prominences > prom(l general)
    #kAll, properties = signal.find_peaks(y, height=-68, prominence=1)
    kAll, properties = signal.find_peaks(y, prominence=prom, distance=dist)
    NP = len(kAll)
    peaksAll = y[kAll]
    minRef = min(peaksAll)
    xPeaksAll = x[kAll]
    prominences = properties.get('prominences')
    for i in range(NP):
        if peaksAll[i]-minRef > 22:
            kPeaks.append(kAll[i])
            if i == 0 : # si el pico está al inicio
                if peaksAll[i + 1]-minRef < 22: # si el pico está al incio y el siguiente NO es un maximo
                # la referencia es el siguiente
                    kRef.append(kAll[i + 1])
                    #SMSR resto el siguiente
                    SMSR.append(int(abs(peaksAll[i] - peaksAll[i+1])))
                else: # si el pico está al incio y el siguiente es un maximo
                    kRef.append(int((kAll[i + 1]+kAll[i])/2))
                    SMSR.append(int(abs(peaksAll[i] - y[int((kAll[i + 1]+kAll[i])/2)])))
            elif i == NP - 1:  # si el pico está al final
                if peaksAll[i - 1]-minRef < 22:  #si el pico está al final y el anterior NO es un maximo
                # la referencia es el anterior
                    kRef.append(kAll[i - 1])
                    # SMSR resto el anterior
                    SMSR.append(int(abs(peaksAll[i] - peaksAll[i-1])))
                else:  #si el pico está al final y el anterior  es un maximo
                    #kRef.append(-1)
                    #SMSR.append(-1)
                    kRef.append(int((kAll[i - 1] + kAll[i]) / 2))
                    SMSR.append(int(abs(peaksAll[i] - y[int((kAll[i-1] + kAll[i]) / 2)])))
            else:  # si el pico esta entre dos picos, comparar izq y derecha
                refRight = peaksAll[i+1]
                refLeft = peaksAll[i-1]
                if refRight >= refLeft:
                    if peaksAll[i + 1] - minRef <22:  # si el siguente no es un maximo
                        kRef.append(kAll[i + 1])
                        SMSR.append(int(abs(peaksAll[i] - peaksAll[i+1])))
                    elif peaksAll[i - 1] - minRef < 22:  # si el anterior no es un maximo
                        kRef.append(kAll[i - 1])
                        SMSR.append(int(abs(peaksAll[i] - peaksAll[i-1])))
                    else: # sie stá entre dos máximos
                        #kRef.append(-1)
                        #SMSR.append(-1)
                        kRef.append(int((kAll[i] + kAll[i + 1]) / 2))
                        SMSR.append(int(abs(peaksAll[i] - y[int((kAll[i] + kAll[i + 1]) / 2)])))
                else: #if refLeft >= refRight
                    if peaksAll[i - 1] - minRef < 22: # si el anterior no es un maximo
                        kRef.append(kAll[i - 1])
                        SMSR.append(int(abs(peaksAll[i] - peaksAll[i - 1])))
                    elif peaksAll[i + 1] - minRef < 22:  # si el  siguiente no es un maximo
                        kRef.append(kAll[i + 1])
                        SMSR.append(int(abs(peaksAll[i] - peaksAll[i + 1])))
                    else: # si stá entre dos máximos
                        #kRef.append(-1)
                        #SMSR.append(-1)
                        kRef.append(int((kAll[i] + kAll[i+1]) / 2))
                        SMSR.append(int(abs(peaksAll[i] - y[int((kAll[i] + kAll[i+1]) / 2)])))
    return SMSR, kPeaks, kRef

def Cal_FWHM_x3dB(x, y, kPeaks):
    Lpeaks = len(kPeaks)
    x1 = np.empty(Lpeaks)
    x2 = np.empty(Lpeaks)
    FWHM = np.empty(Lpeaks)
    for i in range(Lpeaks):
        kMax =kPeaks[i]
        ymax = y[kMax]
        # Left
        k1 = kMax
        y3dB = y[kMax] - 3
        while y[k1] > y3dB:
            k1 = k1 - 1
        yPoints = [y[k1], y[k1 + 1]]
        xPoints = [x[k1], x[k1 + 1]]
        m = (x[k1]- x[k1 + 1])/(y[k1]- y[k1 + 1])
        x3dB = x[k1] + m*(y3dB-y[k1])
        x1[i] = round(x3dB, 4)
        #Right
        k2 = kMax
        while y[k2] >= y3dB:
            k2 = k2 + 1
        yint = [y[k2], y[k2-1]]
        xint = [x[k2], x[k2-1]]
        m = (x[k2] - x[k2-1]) / (y[k2] - y[k2-1])
        x3dB = x[k2] + m * (y3dB - y[k2])
        x2[i] = round(x3dB, 4)
        FWHM[i] = x2[i] - x1[i]
    return np.round(FWHM, decimals=2), x1, x2


def CalculateLaserParameters(x, y, prom, dist):
    x = np.array(x)
    y = np.array(y)
    SMSR = []
    kPeaks = []
    kRef = []
    #Find all prominences >5
    #kAll, properties = signal.find_peaks(y, prominence=5, distance=dist)
    kAll, properties = signal.find_peaks(y, prominence=prom)
    NP = len(kAll)
    peaksAll = y[kAll]
    xPeaksAll = x[kAll]
    prominences = properties.get('prominences')
    for i in range(NP):
        if prominences[i] > 20:
            kPeaks.append(kAll[i])
            if i == 0:  # si el pico está al incio
                # la referencia es el siguiente
                kRef.append(kAll[i + 1])
                #SMSR resto el siguiente
                SMSR.append(round(abs(peaksAll[i] - peaksAll[i+1]),1))
            elif i == NP - 1:  # si el pico está al final
                # la referencia es el anterior
                kRef.append(kAll[i - 1])
                # SMSR resto el anterior
                SMSR.append(round(abs(peaksAll[i] - peaksAll[i-1]),1))
            else:  # si el pico esta entre dos picos, comparar izq y derecha
                refRight = peaksAll[i+1]
                refLeft = peaksAll[i-1]
                if refRight >= refLeft:
                    if prominences[i + 1] < 20:
                        kRef.append(kAll[i + 1])
                        SMSR.append(round(abs(peaksAll[i] - peaksAll[i+1],1)))
                    else:
                        kRef.append(kAll[i - 1])
                        SMSR.append(round(abs(peaksAll[i] - peaksAll[i-1]),1))
                else:# refLeft >refRight
                    if prominences[i - 1] < 20:
                        kRef.append(kAll[i - 1])
                        SMSR.append(round(abs(peaksAll[i] - peaksAll[i - 1]),1))
                    else:
                        kRef.append(kAll[i + 1])
                        SMSR.append(round(abs(peaksAll[i] - peaksAll[i + 1]),1))
    return SMSR, kPeaks, kRef

def PlotLaserFeaturesNew(x, y, xRange, yRange, prom, dist):
    fig, ax = plt.subplots()
    ax.plot(x, y, color='b', linewidth=0.8)
    dx = 5
    fig, ax = SettingAxis(fig, ax, xRange, yRange,  dx, 'Pout')
    SMSR, kPeaks, kRef, FWHM, xa, xb = LaserFeatures(x, y, [1]) # es un sola señal
    Lp = len(kPeaks)
    for i in range(Lp):
        # lambda max
        xmax = x[kPeaks[i]][0]
        ymax = y[kPeaks[i]][0]
        plt.text(xmax - 1, ymax + 6, str(round(xmax, 2)) + 'nm', size=7)
        xy1 = (xmax, ymax + 0.5)
        xytext1 = (xmax, ymax + 5)
        ax.annotate('', xy=xy1, xycoords='data',
                    xytext=xytext1, textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    ec="k",
                                    shrinkA=0, shrinkB=0))
        # FWHM
        x1 = xa[i][0]
        x2 = xb[i][0]
        # left arrow
        xy1 = (x1, ymax - 3)
        xytext1 = (x1 - 1, ymax - 3)
        ax.annotate('', xy=xy1, xycoords='data',
                    xytext=xytext1, textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    ec="k",
                                    shrinkA=0, shrinkB=0))
        # right arrow
        xy2 = (x2, ymax - 3)
        xytext2 = (x2 + 1, ymax - 3)
        ax.annotate('', xy=xy2, xycoords='data',
                    xytext=xytext2, textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    ec="k",
                                    shrinkA=0, shrinkB=0))
        # FWHM text
        xFWHM = x1 + 1
        yFWHM = ymax - 7
        #str(round(b[0], 2)) + 'nm \n'
        plt.text(xFWHM, yFWHM, str(round(FWHM[i][0]*1000)) + 'pm', size=7)
        # SMSR
        xref = x[kRef[i]]
        yref = y[kRef[i]]
        xy = ((xmax + xref) / 2, ymax)
        xytext = ((xmax + xref) / 2, yref)
        ax.annotate('', xy=xy, xycoords='data',
                    xytext=xytext, textcoords='data',
                    arrowprops=dict(arrowstyle="<->",
                                    ec="k",
                                    shrinkA=0, shrinkB=0))

        # Horizontal Lines lower
        xy = (xref, yref + 0.3)
        xytext = ((xref + xmax) / 2, yref + 0.3)
        ax.annotate('', xy=xy, xycoords='data',
                    xytext=xytext, textcoords='data',
                    arrowprops=dict(arrowstyle="-",
                                    linestyle="--",
                                    ec="k",
                                    shrinkA=1, shrinkB=1))

        # Horizontal Lines upper
        xy = (xmax, ymax + 0.1)
        xytext = ((xref + xmax) / 2, ymax + 0.1)
        ax.annotate('', xy=xy, xycoords='data',
                    xytext=xytext, textcoords='data',
                    arrowprops=dict(arrowstyle="-",
                                    linestyle="--",
                                    ec="k",
                                    shrinkA=1, shrinkB=1))

        # SMSR text
        if xref < xmax:
            #xtext = xref - (xmax - xref) / 2
            xtext = xref - 1
        if xref > xmax:
            xtext = (xref + xmax) / 2 + 1
        ytext = (ymax + yref) / 2
        plt.text(xtext, ytext, str(round(SMSR[i][0], 1)) + 'dB', size=7)
        # print figure
        whichDir = os.getcwd()  # current directory
        plt.savefig('Laser' + str(Lp) + 'B.png', dpi=300, transparent=True, bbox_inches='tight')

def PlotSWLaserFeatures(x, y, L, param, xRange, yRange):
    SMSRall, kPeaksall, kRefall, FWHMall, xaAll, xbAll = LaserFeatures(x, y, L)
    fig,ax = plt.subplots(5, 2, sharex=True, sharey=True)
    labelSW = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)']
    xmaxAll = []
    for k in range(len(L)):
        xk = x[k]
        yk = y[k]
        paramk = param[k]
        SMSR = SMSRall[k]
        kPeaks = kPeaksall[k]
        kRef = kRefall[k]
        FWHM = FWHMall[k]
        xa = xaAll[k]
        xb = xbAll[k]
        xmaxAll.append(SelectingList(xk, kPeaks))
        f = k % 5
        c = int(k / 5)
        ax[f, c].plot(xk, yk, color='b', linewidth=0.8, label=paramk)
        fig, ax[f,c] = SettingSWAxis(fig, ax[f,c], xRange, yRange, 'Pout')
        ax[f,c].text(1522,-8, labelSW[k], size=8)
        ley = 'C='+str(paramk) + '$m^{-1}$'
        ax[f, c].text(1540, -76, ley, size=6, color='blue')
        Lp = len(kPeaks)
        for i in range(Lp):
            # lambda max
            xmax = xk[kPeaks[i]]
            #xmaxAll.append(xmax)
            ymax = yk[kPeaks[i]]
            if ((Lp==1) or (i !=Lp-1  and (xk[kPeaks[i+1]] - xmax > 1) ) or (i==Lp-1 and (xmax - xk[kPeaks[i-1]] > 1)  )):
                ax[f, c].text(xmax - 2, ymax + 6, str(round(xmax, 2)) + 'nm', size=6)
                xy1 = (xmax, ymax + 0.5)
                xytext1 = (xmax, ymax + 5)
                ax[f, c].annotate('', xy=xy1, xycoords='data',
                    xytext=xytext1, textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                    ec="k",lw=0.5,
                    shrinkA=0, shrinkB=0))
            # SMSR
            if SMSR[i]>20:
                xref = xk[kRef[i]]
                yref = yk[kRef[i]]
                # SMSR text
                #xy = ((xmax + xref) / 2, ymax)
                #xytext = ((xmax + xref) / 2, yref)
                xy = (xref, ymax)
                xytext = (xref, yref)
                ax[f, c].annotate('', xy=xy, xycoords='data',
                    xytext=xytext, textcoords='data',
                    arrowprops=dict(arrowstyle="<->",
                    ec="k", lw=0.5,
                    shrinkA=0, shrinkB=0))
                # Horizontal Lines lower
                xy = (xref, yref + 0.3)
                #xytext = ((xref + xmax) / 2, yref + 0.3)
                xytext = (xref, yref + 0.3)
                ax[f, c].annotate('', xy=xy, xycoords='data',
                    xytext=xytext, textcoords='data',
                    arrowprops=dict(arrowstyle="-",
                    linestyle="--",lw=0.5,
                    ec="k",
                    shrinkA=1, shrinkB=1))
                # Horizontal Lines upper
                xy = (xmax, ymax + 0.1)
                #xytext = ((xref + xmax) / 2, ymax + 0.1)
                xytext = (xref, ymax + 0.1)
                ax[f, c].annotate('', xy=xy, xycoords='data',
                    xytext=xytext, textcoords='data',
                    arrowprops=dict(arrowstyle="-",
                    linestyle="--",lw=0.5,
                    ec="k",
                    shrinkA=1, shrinkB=1))
                # SMSR text
                if xref < xmax:
                    #xtext = xref - (xmax - xref) / 2
                    xtext = xref+0.5
                    #xtext = xref - 1
                if xref > xmax:
                    xtext = (xref + xmax) / 2-0.5
                    #xtext = xmax+1

                """    
                if xref < xmax:
                    xtext = xref - (xmax-xref) / 2
                if xref > xmax:
                    xtext = (xref + xmax) / 2 +1
                """
                ytext = (ymax + yref) / 2
                ax[f, c].text(xtext, ytext, str(round(SMSR[i], 1)) + '\ndB', size=6)
    ax[2, 0].set_ylabel('Output Power (dBm)')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for a in ax.flat:
        a.label_outer()
    auxWidth = 16 * cm
    auxHeight = 25 * cm
    figure = plt.gcf()
    figure.set_size_inches(auxWidth, auxHeight)
    plt.tight_layout()
    fig.savefig('SW.png', dpi=300, transparent=True, )
    print(labelSW, xmaxAll, SMSRall,FWHMall)
    #Write CSV
    whichDir = os.getcwd()
    file = open("SW.csv", "w", newline='')
    spamreader = csv.writer(file)
    spamreader.writerow(["Signal", "Wavelength peaks (nm)", "SMSR (dB)", "FWHM (nm)"])
    for i in range(len(L)):
        FWHMi = FWHMall[i].tolist()
        listi=[labelSW[i], xmaxAll[i], SMSRall[i], FWHMi]
        spamreader.writerow(listi)
    file.close()
    return


def PlotTN_SW(x, y, param, L, xRange, yRange):
    SMSRall, kPeaksall, kRefall, FWHMall, xaAll, xbAll = LaserFeatures(x, y, L)
    fig,ax = plt.subplots()
    #labelSW = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)']
    for k in range(len(L)):
        xk = x[k]
        yk = y[k]
        SMSR = SMSRall[k]
        kPeaks = kPeaksall[k]
        kRef = kRefall[k]
        FWHM = FWHMall[k]
        xa = xaAll[k]
        xb = xbAll[k]
        plt.plot(xk, yk, color=colorLegend[k], linewidth=0.8)
        fig, ax = SettingSWAxis(fig, ax, xRange, yRange, 'Pout')
        Lp = len(kPeaks)
        for i in range(Lp):
            # lambda max
            xmax = xk[kPeaks[i]]
            ymax = yk[kPeaks[i]]
            if ((Lp==1) or (i !=Lp-1  and (xk[kPeaks[i+1]] - xmax > 1) ) or (i==Lp-1 and (xmax - xk[kPeaks[i-1]] > 1)  )):
                ax.text(xmax - 1, ymax + 6, str(round(xmax, 2)) + 'nm', size=6)
                xy1 = (xmax, ymax + 0.5)
                xytext1 = (xmax, ymax + 5)
                ax.annotate('', xy=xy1, xycoords='data',
                    xytext=xytext1, textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                    ec="k",
                    shrinkA=0, shrinkB=0))
            """
            # SMSR
            if SMSR[i]>20:
                xref = xk[kRef[i]]
                yref = yk[kRef[i]]
                xy = ((xmax + xref) / 2, ymax)
                xytext = ((xmax + xref) / 2, yref)
                ax.annotate('', xy=xy, xycoords='data',
                    xytext=xytext, textcoords='data',
                    arrowprops=dict(arrowstyle="<->",
                    ec="k",
                    shrinkA=0, shrinkB=0))
                # Horizontal Lines lower
                xy = (xref, yref + 0.3)
                xytext = ((xref + xmax) / 2, yref + 0.3)
                ax.annotate('', xy=xy, xycoords='data',
                    xytext=xytext, textcoords='data',
                    arrowprops=dict(arrowstyle="-",
                    linestyle="--",
                    ec="k",
                    shrinkA=1, shrinkB=1))
                # Horizontal Lines upper
                xy = (xmax, ymax + 0.1)
                xytext = ((xref + xmax) / 2, ymax + 0.1)
                ax.annotate('', xy=xy, xycoords='data',
                    xytext=xytext, textcoords='data',
                    arrowprops=dict(arrowstyle="-",
                    linestyle="--",
                    ec="k",
                    shrinkA=1, shrinkB=1))
                # SMSR text
                if xref < xmax:
                    #xtext = (xmax + xref) / 2
                    xtext = xref - (xmax-xref) / 2
                if xref > xmax:
                    xtext = (xref + xmax) / 2 +1
                ytext = (ymax + yref) / 2
                ax[f, c].text(xtext, ytext, str(round(SMSR[i], 1)) + 'dB', size=6)
    ax[2, 0].set_ylabel('Output Power (dBm)')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for a in ax.flat:
        a.label_outer()
    
    """
    fig, ax = SettingAxis(fig, ax, xRange, yRange, dx, 'Pout')
    plt.savefig('TN_SW.png', dpi=300, transparent=True, )
    return


def PlotSW(x, y, param, L, xRange, yRange):
    SMSRall, kPeaksall, kRefall, FWHMall, xaAll, xbAll = LaserFeatures(x, y, L)
    fig, ax = plt.subplots()
    NS = len(L)
    # labelSW = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)']
    paramStr = []
    for k in range(NS):
        xk = x[k]
        yk = y[k]
        SMSR = SMSRall[k]
        kPeaks = kPeaksall[k]
        kRef = kRefall[k]
        FWHM = FWHMall[k]
        xa = xaAll[k]
        xb = xbAll[k]
        paramStr.append(str(param[k]))
        #plt.plot(xk, yk, color=colorLegend[k], linewidth=0.8)
        ax.plot(xk, yk, color=colorLegend[k], linewidth=1, label=paramStr[k])
        #fig, ax = SettingSWAxis(fig, ax, xRange, yRange, 'Pout')
        Lp = len(kPeaks)
        for i in range(Lp):
            # lambda max
            xmax = xk[kPeaks[i]]
            ymax = yk[kPeaks[i]]
            if ((Lp == 1) or (i != Lp - 1 and (xk[kPeaks[i + 1]] - xmax > 1)) or (
                    i == Lp - 1 and (xmax - xk[kPeaks[i - 1]] > 1))):
                ax.text(xmax - 1, ymax + 6, str(round(xmax, 2)) + 'nm', size=6)
                xy1 = (xmax, ymax + 0.5)
                xytext1 = (xmax, ymax + 5)
                ax.annotate('', xy=xy1, xycoords='data',
                            xytext=xytext1, textcoords='data',
                            arrowprops=dict(arrowstyle="->",
                                            ec="k",
                                            shrinkA=0, shrinkB=0))
                """
                # FWHM
                x1 = xa[i]
                x2 = xb[i]
                # left arrow
                xy1 = (x1, ymax - 3)
                xytext1 = (x1 - 1, ymax - 3)
                ax.annotate('', xy=xy1, xycoords='data',
                            xytext=xytext1, textcoords='data',
                            arrowprops=dict(arrowstyle="->",
                                            ec="k",
                                            shrinkA=0, shrinkB=0))
                # right arrow
                xy2 = (x2, ymax - 3)
                xytext2 = (x2 + 1, ymax - 3)
                ax.annotate('', xy=xy2, xycoords='data',
                            xytext=xytext2, textcoords='data',
                            arrowprops=dict(arrowstyle="->",
                                            ec="k",
                                            shrinkA=0, shrinkB=0))
                # FWHM text
                #xFWHM = x1 + 1
                xFWHM = x1+0.4
                yFWHM = ymax - 7
                plt.text(xFWHM, yFWHM, str(round(FWHM[i] * 1000, 2)) + 'pm', size=6)

            # SMSR
            if SMSR[i]>20:
                xref = xk[kRef[i]]
                yref = yk[kRef[i]]
                #xy = ((xmax + xref) / 2, ymax)
                xy = ((xmax + xref) / 2 + (xmax-xref)/4, ymax)
                #xytext = ((xmax + xref) / 2, yref)
                xytext = ((xmax + xref) / 2 + (xmax-xref)/4, yref)
                ax.annotate('', xy=xy, xycoords='data',
                    xytext=xytext, textcoords='data',
                    arrowprops=dict(arrowstyle="<->",
                    ec="k",
                    shrinkA=0, shrinkB=0))
                # Horizontal Lines lower
                xy = (xref, yref + 0.3)
                #xytext = ((xref + xmax) / 2, yref + 0.3)
                xytext = ((xref + xmax) / 2+ (xmax - xref) / 4, yref + 0.3)
                ax.annotate('', xy=xy, xycoords='data',
                    xytext=xytext, textcoords='data',
                    arrowprops=dict(arrowstyle="-",
                    linestyle="--",
                    ec="k",
                    shrinkA=1, shrinkB=1))
                # Horizontal Lines upper
                xy = (xmax, ymax + 0.1)
                #xytext = ((xref + xmax) / 2, ymax + 0.1)
                xytext = ((xref + xmax) / 2 + (xmax - xref) / 4, ymax + 0.1)
                ax.annotate('', xy=xy, xycoords='data',
                    xytext=xytext, textcoords='data',
                    arrowprops=dict(arrowstyle="-",
                    linestyle="--",
                    ec="k",
                    shrinkA=1, shrinkB=1))
                # SMSR text
                if xref < xmax:
                    #xtext = xref - (xmax-xref) / 2
                    xtext =  (xref + xmax) / 2 + (xmax - xref) / 4 + 0.2
                if xref > xmax:
                    #xtext = (xref + xmax) / 2 + 1
                    xtext = (xref + xmax) / 2 + (xmax - xref) / 4 + 0.2
                ytext = (ymax + yref) / 2
                ax.text(xtext, ytext, str(round(SMSR[i], 1)) + 'dB', size=6, color=colorLegend[k])
            """
    #ax.set_ylabel('Output Power (dBm)')
    lgd = plt.legend(paramStr, fontsize=6,
                    title=SelecTextVarControl('Curv'),
                    title_fontsize=6,
                    bbox_to_anchor=(1, 1),
                    loc='upper right',
                    fancybox=False)
    # Hide x labels and tick labels for top plots and y ticks for right plots
    """
    for a in ax.flat:
        a.label_outer()
    """
    fig, ax = SettingAxis(fig, ax, xRange, yRange, dx, 'Pout')
    """
    auxWidth = 9 * cm
    auxHeight = 8 * cm
    figure = plt.gcf()
    figure.set_size_inches(auxWidth, auxHeight)
    plt.tight_layout()
    """
    fig.savefig('TN_SW_Inc.png', dpi=300, transparent=True, )
    return


def PlotStabPoints(pStab, timeSel,timeRange, yRange, typeSignal):
    if typeSignal=='PoutStab':
        yerr=0.02
    if typeSignal=='lambdaStab':
        yerr=0.004
    fig, ax = plt.subplots()
    ax.errorbar(timeSel, pStab, yerr=yerr, fmt = 's', ms=1, ls='-', color='blue', ecolor='black', capsize=3, elinewidth=0.8)
    #ax.plot(timeSel, pStab, color='b', marker='d', markersize=2)
    dx=20
    fig, ax = SettingAxis(fig, ax, timeRange, yRange, dx, typeSignal)
    fig.savefig(typeSignal+'.png', dpi=300, transparent=True, bbox_inches='tight')
    varPoint = max(pStab) - min(pStab)
    return varPoint

def PlotLaser2BFeatures(x, y, fig, ax,  SMSR, kPeaks, kRef, FWHM, xa, xb ):
    Lp = len(kPeaks)
    for i in range(Lp):
        # lambda max
        xmax = x[kPeaks[i]]
        ymax = y[kPeaks[i]]
        plt.text(xmax - 1, ymax  + 6, str(round(xmax, 2)) + 'nm', size=6)
        #plt.text(xmax - 1, ymax + 6, ' $\lambda$' + str(i) + '=' + str(round(xmax, 2)) + 'nm', size=6)
        xy1 = (xmax, ymax + 0.5)
        xytext1 = (xmax, ymax + 5)
        ax.annotate('', xy=xy1, xycoords='data',
                xytext=xytext1, textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                ec="k",
                                shrinkA=0, shrinkB=0))
        # FWHM
        x1 = xa[i]
        x2 = xb[i]
        # left arrow
        xy1 = (x1, ymax - 3)
        xytext1 = (x1 - 1, ymax - 3)
        ax.annotate('', xy=xy1, xycoords='data',
                    xytext=xytext1, textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    ec="k",
                                    shrinkA=0, shrinkB=0))
        # right arrow
        xy2 = (x2, ymax - 3)
        xytext2 = (x2 + 1, ymax - 3)
        ax.annotate('', xy=xy2, xycoords='data',
                    xytext=xytext2, textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    ec="k",
                                    shrinkA=0, shrinkB=0))
        #FWHM text
        xFWHM = x1 + 1
        yFWHM = ymax - 5
        plt.text(xFWHM, yFWHM,  str(round(FWHM[i]*1000,2)) + 'pm', size=6)
        # SMSR
        xref = x[kRef[i]]
        yref = y[kRef[i]]
        xy = ((xmax + xref) / 2, ymax)
        xytext = ((xmax + xref) / 2, yref)
        ax.annotate('', xy=xy, xycoords='data',
                    xytext=xytext, textcoords='data',
                    arrowprops=dict(arrowstyle="<->",
                                    ec="k",
                                    shrinkA=0, shrinkB=0))

        # Horizontal Lines lower
        xy = (xref, yref + 0.3)
        xytext = ((xref + xmax)/2, yref + 0.3)
        ax.annotate('', xy=xy, xycoords='data',
                    xytext=xytext, textcoords='data',
                    arrowprops=dict(arrowstyle="-",
                                    linestyle="--",
                                    ec="k",
                                    shrinkA=1, shrinkB=1))

        # Horizontal Lines upper
        xy = (xmax, ymax + 0.1)
        xytext = ((xref + xmax)/2, ymax + 0.1)
        ax.annotate('', xy=xy, xycoords='data',
                    xytext=xytext, textcoords='data',
                    arrowprops=dict(arrowstyle="-",
                                    linestyle="--",
                                    ec="k",
                                    shrinkA=1, shrinkB=1))

        # SMSR text
        xtext = (xmax+ xref)/2
        ytext = (ymax + yref) / 2
        plt.text(xtext, ytext, str(round(SMSR[i], 1)) + 'dB', size=6)
    #print figure
    plt.savefig('Laser'+str(Lp)+'B.png', dpi=300, transparent=True, bbox_inches='tight')

def PlotLaser3DInt(x,y,param):
    NOF = len(param)
    figS = go.Figure()
    for i in range(NOF):
        xi = x[i]
        yi = param[i] * np.ones(len(xi))
        zi = y[i]
        figS.add_trace(go.Scatter3d(x=xi,
                                    y=yi,
                                    z=zi,
                                    mode='lines',
                                    showlegend=False,
                                    marker=dict(
                                        size=12,
                                        opacity=0.8
                                        )))
    #figS.update_layout(title="Stability")
    figS.show()
    return

# SMSR, kPeaks, kRef, FWHM, xa, xb = LaserFeatures(x,y,L)
def LaserFeatures(x,y,L):
    NOF = len(L)
    SMSR, kPeaks, kRef = [], [], []
    FWHM, xa, xb =  [], [], []
    for i in range (NOF):
        if NOF==1:
            xi=x
            yi=y
        else:
            xi= x[i]
            yi=y[i]
        SMSRi, kPeaksi, kRefi = CalculateSMSRall(xi, yi, 0.5, 100)
        FWHMi, xai, xbi = Cal_FWHM_x3dB(xi, yi, kPeaksi)
        SMSR.append(SMSRi)
        kPeaks.append(kPeaksi)
        kRef.append(kRefi)
        FWHM.append(FWHMi)
        xa.append(xai)
        xb.append(xbi)
    return SMSR, kPeaks, kRef, FWHM, xa, xb


def PlotLaserSw3D(x, z, param,xRange):
    color = ['k', 'b', 'r', 'g', 'c', 'm', 'y']
    yAxisString = []
    Lc = len(color)
    #x, wavelength list of lists
    #z, output power list of lists
    fig = plt.figure()
    ax = plt.subplot(projection='3d')
    cValue = []
    verts = []
    NS = len(param)
    for i in range(NS):
        #yAxisString.append(str(param[i]))
        yAxisString.append('C'+str(i))
        yi = [i] * len(x[i])
        zi = z[i]
        Lz = len(zi)
        xp = np.array([x[i]])
        yp = np.array([yi])
        zp = np.array([zi])
        ax.plot_wireframe(xp, yp, zp, color=color[i % Lc], linewidth=0.8)
    ax.grid(False)
    ax.azim = -69
    ax.elev = 7
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('k')
    ax.yaxis.pane.set_edgecolor('k')
    ax.zaxis.pane.set_edgecolor('k')
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.set_xlabel('Wavelength (nm)',fontsize=8)
    #ax.set_ylabel('Time(s)',fontsize=8)
    #for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(6)
    ax.xaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_tick_params(labelsize=6)
    ax.zaxis.set_tick_params(labelsize=6)
    ax.set_zticks(list(range(-80,-9,10)))
    #plt.xticks(list(range(1545,1561,5)), ['1545', '1550', '1555', '1560'] ,fontsize=6)
    ax.set_xticks(list(range(xRange[0],xRange[1], 20)), fontsize=6)
    #plt.yticks(list(range(NS)), ['0','','','','','','','','','','80'], fontsize=6)
    ax.set_yticks(list(range(NS)),yAxisString, fontsize=6)
    ax.set_zlabel('Output power (dBm)',fontsize=8)
    ax.set_xlim(xRange[0], xRange[1])
    ax.set_zlim(-80, -10)
    auxWidth = 9 * cm
    auxHeight = 15 * cm
    figure = plt.gcf()
    figure.set_size_inches(auxWidth, auxHeight)
    plt.tight_layout()
    plt.show()
    #plt.savefig('Stability3D.png', dpi=300, transparent=True, bbox_inches='tight')
    plt.savefig('LaserSw3D.png', dpi=300, transparent=True,bbox_inches='tight')
    return

def PlotLaserStability3D(x, z, time,xRange):
    color = ['k', 'b', 'r', 'g', 'c', 'm', 'y']
    Lc = len(color)
    #x, wavelength list of lists
    #z, output power list of lists
    fig = plt.figure()
    ax = plt.subplot(projection='3d')
    cValue = []
    verts = []
    NS = len(time)
    for i in range(NS):
        yi = [i] * len(x[i])
        zi = z[i]
        Lz = len(zi)
        xp = np.array([x[i]])
        yp = np.array([yi])
        zp = np.array([zi])
        ax.plot_wireframe(xp, yp, zp, color=color[i % Lc], linewidth=0.6)
        ax.grid(False)
        ax.set_facecolor('white')
        ax.azim = -69
        ax.elev = 7
    plt.xlabel('Wavelength (nm)',fontsize=8, labelpad=-5)
    plt.ylabel('Time(s)', fontsize=8, labelpad=-5)
    ax.set_zlabel('Output power (dBm)', fontsize=9, labelpad=-5)
    ax.xaxis.set_tick_params(labelsize=6, pad=-3)
    ax.yaxis.set_tick_params(labelsize=6, pad=-3)
    ax.zaxis.set_tick_params(labelsize=6, pad=-2)
    ax.set_zticks(list(range(-80,-19,20)))
    #ax.set_xticks(list(range(1540, 1561, 5)), ['''1545', '1550', '1555', '1560'],fontsize=6)
    ax.set_xticks(list(range(1540, 1561, 5)),['1540','1545', '1550','1555', '1560'], fontsize=7)
    ax.set_yticks(list(range(NS)), ['0','','','','','','','','','','80'], fontsize=7)
    ax.set_xlim(xRange[0], xRange[1])
    ax.set_zlim(-80, -20)
    plt.tight_layout()
    auxWidth = 12 * cm
    auxHeight =10  * cm
    figure = plt.gcf()
    figure.set_size_inches(auxWidth, auxHeight)
    plt.show()
    plt.savefig('Stability3D.png', dpi=300, transparent=True, bbox_inches='tight')
    return

def PointsLinearity(df1, val):
    col_names = df1.columns.values[1:]
    paramStr = col_names.tolist()
    NOF = len(paramStr)
    if val == 'max':
        for i in range(NOF):
            df1['max' + str(i)] = df1.iloc[argrelextrema(df1[paramStr[i]].values, np.greater_equal, order=15)[0]][paramStr[i]]
    elif val == 'min':
        for i in range(NOF):
            df1['min' + str(i)] = df1.iloc[argrelextrema(df1[paramStr[i]].values, np.less_equal, order=15)[0]][paramStr[i]]
    else:
        #falta verificar
        valY1 = df1[(df1[paramStr] >= val)][paramStr]
        kval = df1[(df1[paramStr] >= val)][paramStr].idxmin()
        valX1 = df1["Wavelength"].loc[kval].tolist()
    return df1