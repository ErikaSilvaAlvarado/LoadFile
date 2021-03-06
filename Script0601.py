# Importamos todo lo necesario
import os
import jinja2
from flask import Flask, render_template, request, redirect, url_for
#from app import app
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
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
    # renderiamos la plantilla "index.html"
    return render_template('index.html')
    #return render_template('index.html')

@app.route("/upload", methods=['GET','POST'])
def uploader():
    data = []
    if request.method == 'POST':
        uploaded_files = request.files.getlist('archivo')
        basedir = os.path.abspath(os.path.dirname(__file__))
        for file in uploaded_files:
            # obtenemos el archivo del input "archivo"
            filename = secure_filename(file.filename)
            # Guardamos el archivo en el directorio "ArchivosPDF"
            filepath = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
            file.save(os.path.join(basedir, app.config['UPLOAD_FOLDER'], filename))
        dfParam = pd.read_csv(filepath + '/Dec.csv', skiprows=1, header=None, names=["fileName", "param"])
        dfEDFA = pd.read_csv(filepath + '/EDFA.CSV', header=22, names=["xEDFA", "yEDFA"])
        yEDFA = dfEDFA["yEDFA"].tolist()
        yASE = fu.DownSample(yEDFA, 5)
        df = fu.CreateDataFrame(filepath, dfParam)
        dataJSON = json.dumps(df, cls=plotly.utils.PlotlyJSONEncoder)
        fig = fu.PlotParamInt(df)
        #graphJSONaux = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('generalPlot.html', graphJSON=graphJSON, dataJSON=dataJSON)


@app.route('/callback', methods=['POST', 'GET'])
def cb():
    return CustomPlot(request.args.get('data'))

if __name__ == '__main__':
 # Iniciamos la aplicaci??n
 app.run(debug=True)

