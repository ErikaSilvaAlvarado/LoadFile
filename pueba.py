import os
import numpy as np
import pandas as pd
import glob
import csv
import MyFunctions as fu
import plotly.express as px

if __name__ == '__main__':
    basedir = "/home/estudiante/LoadFile/Uploads"
    filepath = basedir
    dfParam = pd.read_csv(filepath + '/Dec.csv', skiprows=1, header=None, names=["fileName", "param"])
    dfEDFA = pd.read_csv(filepath + '/EDFA.CSV', header=22, names=["xEDFA", "yEDFA"])
    yEDFA = dfEDFA["yEDFA"].tolist()
    yASE = fu.DownSample(yEDFA, 5)
    df = fu.CreateDataFrame(filepath, dfParam)
    fig = fu.PlotParamInt(df)
