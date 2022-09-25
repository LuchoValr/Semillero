import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import sklearn.metrics
from sklearn.metrics import r2_score
#Lectura
dat_personal='C:\\Users\\lucho\\OneDrive\\Documentos\\Proyectos\\Modelo\\Área - Características generales (Personas).xlsx'
dato_ocupa='C:\\Users\\lucho\\OneDrive\\Documentos\\Proyectos\\Modelo\\Área - Ocupados.xlsx'
dat_personal_r=pd.read_excel(dat_personal)
print(dat_personal)
dat_personal_r.info()
dato_ocupa_r=pd.read_excel(dato_ocupa)
print(dato_ocupa_r)
dato_ocupa_r.info()
#Fusion de datos
data=dat_personal_r.merge(dato_ocupa_r, how='left',on=['DIRECTORIO','ORDEN'])
print(data)
data1=data[['DIRECTORIO','ORDEN','P6020','P6210','ESC','P6040','P6500','P6440']]
print(data1)

data1['EDAD2']=(data1['P6040']**2)
print(data1)
#1 es hombre y 0 mujer
data1['P6020']=data1['P6020'].replace([2.0], 0)
print(data1)
#1 si tiene contrato y 0 que no
data1['P6440']=data1['P6440'].replace([2.0], 0)
print(data1)
#Con dropna se eliminan todos los datos nulos
datadropna=data1.dropna()
print(datadropna)
#Regresiones con eliminacion de todos los Nan
x_train=datadropna[['ESC','P6040','EDAD2']]
y_train=datadropna['P6500']
x_test=datadropna[['ESC','P6040','EDAD2']]
y_test=datadropna['P6500']
algoritmo=linear_model.LinearRegression()
algoritmo.fit(x_train, y_train)
algoritmo.predict(x_test)
print('Valor de las pendientes o coeficientes "B1":')
print(algoritmo.coef_)
print('Valor de la intersección o coeficiente "B0":')
print(algoritmo.intercept_)
print('Precisión del modelo:')
print(algoritmo.score(x_train, y_train))
#Imputacion de datos
data_imp=data1.pivot_table(index=['ESC','P6030S3','EDAD2'], values='P6500', aggfunc=['mean','median'])





