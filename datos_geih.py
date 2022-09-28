import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import sklearn.metrics
from sklearn.metrics import r2_score
import numpy as np
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
data2=data1
data2.info()
for col in data2.columns:
  print(col)
  print(data2[col].unique())
  print(data2[col].dtypes)
  print('---'*10)
#Salario
data_impw=data2.pivot_table(index=['ESC','P6040','EDAD2'], values='P6500', aggfunc=['mean','median'])
print(data_impw)
data_impw=data2['P6500'].fillna(data2.groupby(['ESC','P6040','EDAD2'])['P6500'].transform('median'))
print(data_impw)
data_impw2=data_impw.interpolate(method='polynomial', order=2)
print(data_impw2)
#Educacion
data_impe=data2.pivot_table(index=['P6500','P6040','EDAD2'], values='ESC', aggfunc=['mean','median'])
print(data_impe)
data_impe=data2['ESC'].fillna(data2.groupby(['P6500','P6040','EDAD2'])['ESC'].transform('median'))
print(data_impe)
data_impe2=data_impe.interpolate(method='polynomial', order=2)
#EDAD (Proxy de exper)
data_imped=data2.pivot_table(index=['P6500','ESC','EDAD2'], values='P6040', aggfunc=['mean','median'])
print(data_imped)
data_imped=data2['P6040'].fillna(data2.groupby(['P6500','ESC','EDAD2'])['P6040'].transform('median'))
print(data_imped)
#EDAD2
data_imped2=data2.pivot_table(index=['P6500','ESC','P6040'], values='EDAD2', aggfunc=['mean','median'])
print(data_imped2)
data_imped2=data2['EDAD2'].fillna(data2.groupby(['P6500','ESC','P6040'])['EDAD2'].transform('median'))
print(data_imped2)
#Contar nulos
print(data_impw2.isnull().sum().sum())
data_impw2.info()
print(data_impe2.isnull().sum().sum())
data_impe2.info()
print(data_imped.isnull().sum().sum())
data_imped.info()
print(data_imped2.isnull().sum().sum())
data_imped2.info()

#Regresiones
x_train1=np.array([data_impe2,data_imped,data_imped2])
x_train1=np.transpose(x_train1)
y_train1=(data_impw2)
x_test1=np.array([data_impe2,data_imped,data_imped2])
x_test1=np.transpose(x_test1)
y_test1=(data_impw2)
algoritmo1=linear_model.LinearRegression()
algoritmo1.fit(x_train1, y_train1)
algoritmo1.predict(x_test1)
print('Valor de las pendientes o coeficientes "B1":')
print(algoritmo1.coef_)
print('Valor de la intersección o coeficiente "B0":')
print(algoritmo1.intercept_)
print('Precisión del modelo:')
print(algoritmo1.score(x_train1, y_train1))













