### Esta libreria lo que trae es un pipeline llamado pipeline_trafo, el cual se encarga de hacer el preprocesamiento de los datos
### identifique que habia data vacia en algunas columnas y luego que habia outliers en 2 de ellas. Por lo que decidi que, primero el
### pipeline empezara con la clase data_clean y su funcion, ésta dropeará columnas que no me sirven, luego sacará valores duplicados y
### por ultimo los cuantificará.
### En segunda instancia el pipeline quitará outliers en las columnas sqfeet y price (si es que está), , luego se hará una imputación
### de datos con el método KNN en las columnas lat y long.
### Por ultimo el pipeline va a reescalar todas las variables para que el modelo de ML tenga una mejor lectura sobre ellas, las dejará
### en variables entre 1 y 0.


import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, _passive_aggressive



class data_clean(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        '''
        Primera funcion para limpiar un poco el df, saco columnas innecesarias para
        el analisis, cambio a enteros, valores que eran str.
        '''

        #########################################################
        # Dropeo columnas que no me sirven.
        #########################################################

        col_drop = ['id','url','region_url','image_url', 'description']
        X = X.drop(columns=col_drop)

        #########################################################
        # Saco duplicados y valores vacios.
        #########################################################

        X[X.duplicated()]

        #########################################################
        # Reemplazo los valores faltantes de laundry y parking 
        # como ukn.
        #########################################################

        for i in X['laundry_options']:
            if i == None:
                i = 'ukn'

        for i in X['parking_options']:
            if i == None:
                i = 'ukn'

        #########################################################
        # Codificacion valores de las columnas que tienen str.
        #########################################################
        def codificacion(df, column):
            '''
            Funcion para codificar valores, cambia str a enteros.
            '''
            count = 0
            lista = list(df[column].unique())

            for i in lista:
                df[column] = df[column].replace({i : count})
                count += 1

        columns = ['region','laundry_options','parking_options','type','state']
        for i in columns:
            codificacion(X, i)

        #########################################################
        #########################################################
        print(1)
        print(X.shape)
        return X

class outliers(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        '''
        Funcion para quitar los outliers de sqfeet y price
        '''
        lista_columnas = [x for x in X]

        # Indico cuartiles y rango inter cuartil para determinar outliers
        # tanto en price con en sqfeet, solo en el df de train.

        if 'price' in lista_columnas:
          
            Q1 = X['sqfeet'].quantile(0.25)
            Q3 = X['sqfeet'].quantile(0.75)
            IQR = Q3 - Q1
            BI = Q1 - 1.5*IQR
            BS = Q3 + 1.5*IQR

            out_sqfeet = (X['sqfeet']<BI) | (X['sqfeet']>BS) 

            X = X[~out_sqfeet]

            Q1 = X['price'].quantile(0.25)
            Q3 = X['price'].quantile(0.75)
            IQR = Q3 - Q1
            BI = Q1 - 1.5*IQR
            BS = Q3 + 1.5*IQR
            out_price = (X['price']<BI) | (X['price']>BS)

            X = X[~out_price]
       
        return X


class category_price(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):

        if 'price' in [x for x in X]:

          # Nueva columna category price

          X['category_price'] = np.where(X['price']<=999,1,0)
          X = X.drop(columns=['price'])

          # Dropeo valores nulos para el train.
          X = X.dropna()

        return X




