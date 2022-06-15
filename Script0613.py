#
import os
from flask import Flask,g,abort, render_template, request, redirect, url_for
# from app import app
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from os.path import join, dirname, realpath
import numpy as np
import pandas as pd
import glob
import csv
import MyFunctions as fu
import json
import plotly
pd.options.plotting.backend = "plotly"
import plotly.express as px
from sqlalchemy import create_engine
# instancia del objeto Flask
app = Flask(__name__)
# Carpeta de subida
app.config['UPLOAD_FOLDER'] = 'Uploads'
"""
@app.route('/callback', methods=['POST', 'GET'])
def cb():
    return gm(request.form.getlist('selectValues'))
"""

@app.route("/")
def upload_file():
 # renderizamos la plantilla "index.html"
 return render_template('index.html')


@app.route("/upload", methods=['POST', 'GET'])
def uploader():
    #if request.method == 'POST':
        uploaded_files = request.files.getlist('archivo')
        basedir = os.path.abspath(os.path.dirname(__file__))
        filepath = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
        for file in uploaded_files:
            # obtenemos el archivo del input "archivo"
            filename = secure_filename(file.filename)
            # Guardamos el archivo en el directorio "ArchivosPDF"
            file.save(os.path.join(basedir, app.config['UPLOAD_FOLDER'], filename))
        dfEDFA = pd.read_csv(filepath + '/EDFA.CSV', header=22, names=["xEDFA", "yEDFA"])
        dfParam = pd.read_csv(filepath + '/Dec.csv', skiprows=1, header=None, names=["fileName", "param"])
        df = fu.CreateTxDataFrame(filepath, dfEDFA, dfParam)  # require EDFA and fileName
        df.to_csv("dataAll.csv", index=False)
        param = dfParam["param"].values
        paramStr = ["%.1f" % x for x in param]
        return render_template('generalPlot.html', paramStr=paramStr, graphJSON=gm(paramStr))

def gm(paramStr):
    df1 = pd.read_csv("dataAll.csv")
    xmin = df1["Wavelength"].min()
    xmax = df1["Wavelength"].max()
    df2 = fu.RefreshDataFrame(df1,[xmin,xmax], paramStr)
    fig = fu.PlotParamInt(df2)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def cb():
    return gm(request.form.getlist('selectValues'))

if __name__ == '__main__':
    # Iniciamos la aplicación
    app.run(debug=True)