#
import os
from flask import Flask, jsonify, g,abort, render_template, request, redirect, url_for
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

@app.route('/callback', methods=['POST', 'GET'])
def cb():
    paramStr = request.form.getlist('selectValues')
    xmin = float(request.form['xmin'])
    xmax = float(request.form['xmax'])
    dx = int(request.form['dx'])
    varControl = request.form['varControl']
    xRange = [xmin, xmax]
    if request.form.get('Leyenda'):
        flagLgd = True
    else:
        flagLgd = False
    #graphJSON = gm(paramStr, xRange, dx, flagLgd, varControl)
    dataJSON, layoutJSON, nameFig = gm(paramStr, xRange, dx, flagLgd, varControl)
    basedir = os.path.abspath(os.path.dirname(__file__))
    filepath = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    #return render_template('customPlot.html', graphJSON=graphJSON)
    #return render_template('customPlot.html', graphJSON=graphJSON, dataJSON = dataJSON)
    return render_template('customPlot.html', dataJSON=dataJSON, layoutJSON=layoutJSON, nameFig =filepath+'/'+nameFig)
    #return dataJSON
    #return render_template('customPlot.html', dataJSON=gm(paramStr, xRange, flagLgd, varControl))
    #return render_template('customPlot.html',  graphJSON=gm(paramStr,xRange,dx, flagLgd,varControl))

@app.route("/")
def upload_file():
    # renderizamos la plantilla "index.html"
    return render_template('index.html')


@app.route("/upload", methods=['POST', 'GET'])
def uploader():
    #if request.method == 'POST':
        uploaded_files = request.files.getlist('archivo')
        isCurv= request.form.get('isCurv')
        basedir = os.path.abspath(os.path.dirname(__file__))
        filepath = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
        for file in uploaded_files:
            # obtenemos el archivo del input "archivo"
            filename = secure_filename(file.filename)
            # Guardamos el archivo en el directorio "ArchivosPDF"
            file.save(os.path.join(basedir, app.config['UPLOAD_FOLDER'], filename))
        dfEDFA = pd.read_csv(filepath + '/EDFA.CSV', header=22, names=["xEDFA", "yEDFA"])
        xmin = dfEDFA["xEDFA"].min()
        xmax = dfEDFA["xEDFA"].max()
        xRange = [xmin, xmax]
        dx = ''
        dfParam = pd.read_csv(filepath + '/Car.csv', skiprows=1, header=None, names=["fileName", "param"])
        param = dfParam["param"].tolist()
        if isCurv:
            curv = fu.Dist2Curv(param)
            dfParam["param"]=curv
        df = fu.CreateTxDataFrame(filepath, dfEDFA, dfParam)  # require EDFA and fileName
        df.to_csv("dataAll.csv", index=False)
        param = dfParam["param"].values
        paramStr = [str(x) for x in param]
        """
        if isCurv:
            paramStr = ["%.6f" % x for x in param]
        else:
            paramStr = ["%.1f" % x for x in param]
        """
        flagLgd = True
        varControl=''
        #graphJSON = gm(paramStr, xRange, dx, flagLgd, varControl)
        dataJSON, layoutJSON, nameFig = gm(paramStr,xRange,dx, flagLgd,varControl)
        return render_template('generalPlot.html', paramStr=paramStr, dataJSON=dataJSON, layoutJSON=layoutJSON)
#        return render_template('generalPlot.html', paramStr=paramStr, graphJSON=graphJSON)

def gm(paramStr,xRange,dx, flagLgd,varControl):
    df1 = pd.read_csv("dataAll.csv")
    x = df1["Wavelength"]
    df2 = fu.RefreshDataFrame(df1,xRange, paramStr)
    fig = fu.PlotParamIntLgd(df2,flagLgd)
    dataJSON = json.dumps(fig.data, cls=plotly.utils.PlotlyJSONEncoder)
    #graphJSON = json.dumps(fig.data, cls=plotly.utils.PlotlyJSONEncoder)
    layoutJSON = json.dumps(fig.layout, cls=plotly.utils.PlotlyJSONEncoder)
    nameFig = fu.PlotTxParam(df2, varControl, dx, 'Inc')
    return dataJSON, layoutJSON, nameFig
    #return graphJSON


if __name__ == '__main__':
    # Iniciamos la aplicación
    app.run(debug=True)