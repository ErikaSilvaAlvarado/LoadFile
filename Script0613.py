#
import os
from flask import Flask, render_template, request, redirect, url_for
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

# instancia del objeto Flask
app = Flask(__name__)
# Carpeta de subida
app.config['UPLOAD_FOLDER'] = 'Uploads'

@app.route("/")
def upload_file():
 # renderizamos la plantilla "index.html"
 return render_template('index.html')

@app.route("/upload", methods=['GET','POST'])
def uploader():
    #data = []
    #if request.method == 'POST':
        uploaded_files = request.files.getlist('archivo')
        basedir = os.path.abspath(os.path.dirname(__file__))
        for file in uploaded_files:
            # obtenemos el archivo del input "archivo"
            filename = secure_filename(file.filename)
            # Guardamos el archivo en el directorio "ArchivosPDF"
            filepath = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
            file.save(os.path.join(basedir, app.config['UPLOAD_FOLDER'], filename))
        dfEDFA = pd.read_csv(filepath + '/EDFA.CSV', header=22, names=["xEDFA", "yEDFA"])
        dfParam = pd.read_csv(filepath + '/Dec.csv', skiprows=1, header=None, names=["fileName", "param"])
        df = fu.TxDataFrame(filepath, dfEDFA, dfParam)
        dataJSON = df.to_json()
        return render_template('generalPlot.html', graphJSON=CreateJSONgraph(dataJSON), cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/callback', methods=['POST', 'GET'])
def cb():
    return CustomJSONgraph(request.args.get('data'),graphJSON)

def CustomJSONgraph(paramList, graphJSON):
    df = pd.read_json(graphJSON.values)
    df2 = pd.DataFrame()
    for i in range (len(paramList)):
        df2[paramList[i]] =df[df[str(paramList[i])]]
        if i == 0:
            df2["Wavelength"] = df["Wavelength"]
    dataJSON2 = df2.to_json()
    return render_template('generalPlot.html', graphJSON=CreateJSONgraph(dataJSON2), cls=plotly.utils.PlotlyJSONEncoder)


def CreateJSONgraph(dataJSON):
    df = pd.read_json(dataJSON)
    fig = fu.PlotParamInt(df)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

if __name__ == '__main__':
    # Iniciamos la aplicación
    app.run(debug=True)

