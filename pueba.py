import os
import numpy as np
import pandas as pd
import glob
import csv
import MyFunctions as fu
import plotly.express as px
import json

if __name__ == '__main__':
    df1 = pd.read_csv("dataAll.csv")
    xmin = 1530
    xmax = 1531
    df2 = fu.RefreshDataFrame(df1, [xmin, xmax], '')