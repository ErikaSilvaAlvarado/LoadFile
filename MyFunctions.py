import os
import glob
import csv
import math
import pandas as pd
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
from scipy.signal import argrelextrema
from scipy.fft import fft, ifft, fftfreq
import json
import plotly
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

def CreateTxDataFrame(filepath,dfEDFA, dfParam):
    xEDFA = dfEDFA["xEDFA"].tolist()
    yEDFA = dfEDFA["yEDFA"].tolist()
    xASE = DownSample(xEDFA, 5)
    xRange = [min(xASE), max(xASE)]
    yASE = DownSample(yEDFA, 5)
    files = dfParam["fileName"].tolist()
    param = dfParam["param"].tolist()
    paramStr = []
    os.chdir(filepath)
    filesCSV = glob.glob('*.CSV')
    for i in range(len(param)): # NOF files cycle
        sufix = "0" + str(files[i]) + ".CSV"
        fileName = [this for this in filesCSV if this.startswith("W") and this.endswith(sufix)]
        var=fileName[0]
        paramStr.append(str(param[i]))
        df = pd.read_csv(fileName[0], header=22, names=["Wavelength", paramStr[i]])  # create dataframe
        yi = df[(df['Wavelength'] >= xRange[0]) & (df['Wavelength'] <= xRange[1])][paramStr[i]].tolist()
        if i == 0:
            x0 = df[(df['Wavelength'] >= xRange[0]) & (df['Wavelength'] <= xRange[1])]['Wavelength'].tolist()
            df1 = pd.DataFrame(list(zip(x0, yi-yASE)), columns=['Wavelength', paramStr[i]])
        else:
            df1[paramStr[i]] = yi-yASE
    return df1

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

def CreateDataPlot(df):
    dataList = []
    col_names = df.columns.values[1:]
    paramStr = col_names.tolist()
    NOF = len(paramStr)
    for i in range(NOF):
        dataList.append({"x": df["Wavelength"].tolist(), "y": df[paramStr[i]].tolist(), "name": paramStr[i]})
    return dataList

def PlotParamIntLgd(df,showLgd):
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
            legendgroup ='lgd'+str(i),
            name=paramStr[i],
            mode="lines",
            line_color=colorLegend[i],
            showlegend=showLgd,
            ))
    fig1.update_layout(hovermode='closest')
    fig1.update_xaxes(showgrid=False, title_font=dict(size=16, family='Helvetica'))
    fig1.update_yaxes(showgrid=False, title_font=dict(size=16, family='Helvetica'))
    fig1.update_layout(xaxis=dict(title="Wavelength (nm)", linecolor="black", zeroline=False),
                      yaxis=dict(title="Transmission (dBm)", linecolor="black",zeroline=False))
    #fig1.update_layout(plot_bgcolor="white", xaxis=dict(title="Wavelength (nm)", linecolor="black"),
    #                   yaxis=dict(title="Transmission (dBm)", linecolor="black"))
    #fig1.show()
    return fig1

def PlotTxParam(df1, varControl, dx, direction):
    col_names = df1.columns.values[1:]
    paramStr = col_names.tolist()
    NOF = len(paramStr)
    xmin = int(df1["Wavelength"].min())
    xmax = int(df1["Wavelength"].max())
    minYi = df1[paramStr].min()
    kmin = df1[paramStr].idxmin()
    fig, ax = plt.subplots()
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    for i in range(NOF):
        # ax.plot(df1["Wavelength"], df1[paramStr[i]], color=colorLegend[i], linestyle=lineStyle[i % Ls], linewidth=0.8)
        m = 4
        ax.plot(df1["Wavelength"], df1[paramStr[i]], color=colorLegend[i], linestyle=LineStyleChange(i, m, Ls),
                linewidth=0.8)
        if paramStr[i]=='0.0':
            paramStr[i] == '0'
    lgd = plt.legend(paramStr, fontsize=6,
                     title=SelecTextVarControl(varControl),
                     title_fontsize=6,
                     bbox_to_anchor=(0, 1),
                     # loc='upper right',
                     loc='upper left',
                     fancybox=False)
    # SEt xlim,ylim
    ymin = min(minYi)
    ymax = 0

    # Arrow indicating the tunning direction
    xOrigin = (xmin + xmax) / 2
    # xOrigin = 1554
    """
    if kmin[2] > kmin[3]:
        xEnd = xOrigin + 1.5
    else:
        xEnd = xOrigin - 1.5
    
    xEnd = xOrigin + 1.5 #ojo sta fija
   
    yOrigin = -16
    
    yOrigin = -16
    ax.annotate('', xy=(xOrigin, yOrigin), xycoords='data',
                xytext=(xEnd, yOrigin), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                ec="k",
                                shrinkA=0, shrinkB=0))
    #dx = int(np.round((xmax-xmin)/5,1))
    ax.text(1554.5, -15, "P", color=colorLegend[m])
    """
    fig, ax = SettingAxis(fig, ax, [xmin, xmax], [ymin, ymax], dx, 'Tx')
    # Save figure
    nameFig= 'Tx' + varControl + direction + '.png'
    plt.savefig(nameFig, dpi=300, transparent=True, bbox_inches='tight',
                bbox_extra_artists=(lgd,))
    return nameFig

def SettingAxis(fig, ax, xRange, yRange, dx, typeSignal):
        if typeSignal == 'Tx':
            xLabel = 'Wavelength (nm)'
            yLabel = 'Transmission (dB)'
        elif typeSignal == 'Pout':
            xLabel = 'Wavelength (nm)'
            yLabel = 'Output power (dBm)'
        elif typeSignal == 'FFT':
            xLabel = 'Spatial frequency (1/nm)'
            yLabel = 'Magnitude (p.u)'
        elif typeSignal == 'Lin':
            xLabel = ''
            yLabel = 'Wavelength (nm)'
        elif typeSignal == 'PoutStab':
            xLabel = 'Time(s)'
            yLabel = 'Output power (dBm)'
        elif typeSignal == 'lambdaStab':
            xLabel = 'Time(s)'
            yLabel = 'Wavelength (nm)'
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        else:
            xLabel = 'x'
            yLabel = 'y'
        if dx!='':
            ax.set_xticks(list(range(xRange[0], xRange[1] + 1, dx)))
        # ax.set_xticks(list(range(xRange[0], xRange[1]+1, 2))) #para el TEDFL parametrico
        # ax.set_xticks(list(range(xRange[0], xRange[1] + 1, 50))) #para el MZI vs  C parametrico
        # ax.set_xticks(list(range(xRange[0], xRange[1] + 1, 2))) #para linealidad por temepratura
        # ax.set_xticks(list(range(xRange[0], xRange[1] + 1, 100)))  # para linealidad por temepratura
        # ax.set_xticks(list(range(xRange[0], xRange[1] + 1, 5)))  # para SW Inc
        # ax.set_xticks(list(range(xRange[0], xRange[1] + 1, 10)))  # para SW Dec
        # ax.set_xticks(list(range(xRange[0], xRange[1]+1, 4))) #para el TEDFL Temp+SW
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
            legendgroup ='lgd'+str(i),
            name=paramStr[i],
            mode="lines",
            line_color=colorLegend[i],
            showlegend=True,
            ))
    fig1.update_layout(height=800, width=1800)
    #fig1.show()
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
    curv = np.empty(len(param),dtype=float)
    L = 0.15 #en metros
    param = np.array(param)
    p2 = np.power(param*1e-3, 2)    #llevar d de um a m
    #curv = (2 * param * 1e-6) / (p2 + L * L)
    curv = np.around(2 * param*1e-3/(p2+L*L), 3)
    #
    return curv #curv en 1/m

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

def RefreshDataFrame(df,xRange, paramStr):
    NOF = len(paramStr)
    x = df[(df['Wavelength'] >= xRange[0]) & (df['Wavelength'] <= xRange[1])]['Wavelength'].tolist()
    df1 = pd.DataFrame()
    df1['Wavelength'] = x
    for i in range(NOF):
        yi = df[(df['Wavelength'] >= xRange[0]) & (df['Wavelength'] <= xRange[1])][paramStr[i]].tolist()
        df1[paramStr[i]] = yi
    return df1


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

def FastFourier(x ,y):
    N = len(x)
    dx = round(x[1] - x[0],4)
    Fs = 1/dx
    Y = fft(y)
    sF = fftfreq(N, dx)[:N // 2]
    mY = 2.0 / N * np.abs(Y[0:N // 2])
    k1 = math.floor(N/Fs)
    return [sF[:k1], mY[:k1]]


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