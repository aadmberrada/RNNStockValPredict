#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 15:30:16 2021

@author: Abdoul_Aziz_Berrada
"""
#python3 -m venv .pyav ===> env vi

import warnings
warnings.filterwarnings('ignore')
import re
from io import StringIO
from datetime import datetime, timedelta
import time
import requests
import math
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler

#import graphviz as graphviz
#import tensorflow
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, LSTM

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

st.set_option('deprecation.showPyplotGlobalUse', False)



#--------------------------------- Head
## ===> Header
st.title("Stock's Value Prediction using RNNs")
#st.title("Stock Value Prediction using Neural Networks")
col1, col2, col3 = st.columns(3)

col1.image('https://github.com/aadmberrada/test/blob/master/img/mosef3.png')

col2.image("https://raw.githubusercontent.com/aadmberrada/test/master/img/YahooFinance.jpeg")

col3.image("https://github.com/aadmberrada/test/blob/master/img/py.png")

st.markdown("""
            
Ce projet a pour but de recueillir par webscrapping des données historiques d'entreprises cotées et puis de faire une analyse predictive in-sample en utilisant les Reccurent Neural Networks RNNs.""")

st.markdown(""" ==> Pre requis :
*   **Python :** pandas, numpy, pyplot, tensorflow, keras, sklearn, streamlit.

*   **Source des données :** https://finance.yahoo.com/quote/yhoo/history/""")

#------------------------------------ Sidebar

st.sidebar.markdown("**Abdoul Aziz Berrada - Amira Slimene**")

st.sidebar.markdown("------------------------------------")

st.sidebar.title("User Input Features")


#--------------
## ===> Dataset

st.sidebar.subheader(" I -  Chargement des données")

action = st.sidebar.selectbox("Selectionnez un actif à étudier", ['Apple Inc - AAPL', 'Tesla Inc - TSLA', 'Microsoft Corporation - MSFT', 
                                                                    'Amazon.com Inc - AMZN', 'Alphabet Inc - GOOGL',
                                                                    'Meta Platforms Inc - FB', 'NVIDIA Corporation - NVDA'])
stk = action.split(" ")[-1]
c = re.findall(r'[A-Za-z/./ /]+', action)[0]
jours = st.sidebar.slider("Selectionnez un nombre de jours d'étude", min_value = 2*360, max_value = 15*360, step = 180)

#-------------
##  ===> Deep Learning model

st.sidebar.subheader(" II -  4-layer RNN model")

scale = st.sidebar.selectbox("Scaler", ["MinMaxScaler", "StandardScaler"])

st.sidebar.write("Layer 0 - LSTM")
layer0 = st.sidebar.slider("Nombre de neurones L0", min_value = 10, max_value = 100, step = 10)

af0 = st.sidebar.selectbox("Fonction d'activation L0", ["tanh", "relu", None])

st.sidebar.write("Layer 1 - LSTM")
layer1 = st.sidebar.slider("Nombre de neurones L1", min_value = 10, max_value = 100, step = 10)

af1 = st.sidebar.selectbox("Fonction d'activation L1", ["tanh", "relu", None])

st.sidebar.write("Layer 2 - Dense")
layer2 = st.sidebar.slider("Nombre de neurones L2", min_value = 10, max_value = 100, step = 10)
af2 = st.sidebar.selectbox("Fonction d'activation L2", ["tanh", "relu", None])

#st.sidebar.write("Layer 3 - Dense")
#af3 = st.sidebar.selectbox("Fonction d'activation L3", ["tanh", "relu", None])

#a = st.sidebar.radio('Select one:', [1, 2])





#----------------------------- Corps du texte
st.header("** 1 - Données**")
if st.checkbox('Afficher les données'):
    @st.cache()
    class WebScrapStock:
        
        timeout = 60
        crumb_link = 'https://finance.yahoo.com/quote/{0}/history?p={0}'
        crumble_regex = r'CrumbStore":{"crumb":"(.*?)"}'
        stock_link = 'https://query1.finance.yahoo.com/v7/finance/download/{quote}?period1={dfrom}&period2={dto}&interval=1d&events=history&crumb={crumb}'
        headers_={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
    
    
        def __init__(self, stock_name, days_back=7):
            self.stock_name = stock_name
            self.session = requests.Session()
            self.dt = timedelta(days=days_back)
    
        def get_crumb(self):
            response = self.session.get(self.crumb_link.format(self.stock_name), timeout=self.timeout, verify = False, headers=self.headers_)
            response.raise_for_status()
            match = re.search(self.crumble_regex, response.text)
            
            if not match:
                raise ValueError("Pas de réponse de Yahoo Finance")
            else:
                self.crumb = match.group(1)
    
        def get_stock(self):
            if not hasattr(self, 'crumb') or len(self.session.cookies) == 0:
                self.get_crumb()
            now = datetime.utcnow()
            dateto = int(now.timestamp())
            datefrom = int((now - self.dt).timestamp())
            url = self.stock_link.format(quote=self.stock_name, dfrom=datefrom, dto=dateto, crumb=self.crumb)
            response = self.session.get(url, verify = False, headers=self.headers_)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text), parse_dates=['Date'])
            df["Stock"] = self.stock_name
            df = df[["Date", "Stock", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
            st.write("On va récupérer les données de", c, "depuis", df.iloc[0, 0], "soit sur", df.shape[0], "jours de marché.")
            return df, url
    
    df, url = WebScrapStock(stock_name  = stk, days_back=jours).get_stock()
    
    dff = df.copy()

    stock = re.findall(r'[A-Z]+', url)[0]
    period1 = re.findall(r'\d{5,}', url)[0]
    period2 = re.findall(r'\d{5,}', url)[1]
    def lien(stock, period1, period2):
        lien = "https://finance.yahoo.com/quote/"+stock+"/history?period1="+period1+"&period2="+period2+"&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true"
        return lien
    lien = lien(stock, period1, period2)

    st.markdown("Les données sont directement téléchargeables en format CSV à l'adresse [Yahoo Finance](lien)")
    #st.write([Yahoo Finance](lien))
    st.write(lien)  
    
    col1, col2 = st.columns([4, 1])
    
    x = df.shape[0] - 1
    y = df.shape[1] - 3
    x1 = df.shape[0] - 2
    y1 = y + 2
    d = 100*(df.iloc[x , y ]  / df.iloc[ x1, y ]  - 1)
    e = 100*(df.iloc[x , y1 ] /  df.iloc[ x1, y1 ] - 1)
    #d = (df.iloc[df.shape[0] - 1 , df.shape[5]] - df.iloc[df.shape[0], df.shape[5]])/df.iloc[df.shape[0], df.shape[5]]
        
    
    col2.metric(label= str(c) + "'s Price", value=  "{:.2f}".format(df.iloc[x , y ]), delta="{:.2f}".format(d)+"%")
    col2.metric(label= str(c) + "'s Volume", value=  "{:.0f}".format(df.iloc[x , y1 ]), delta="{:.2f}".format(e)+"%")
    
    col1.dataframe(df.sort_values("Date", ascending=False))
    
    st.write('Statistiques descriptives des actions', c)
    st.table(df.describe())
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    trace = go.Scatter(
    x = df["Date"],
    y = df["Close"],
    mode = "lines",
    name = "Cours de fermeture",
    marker = dict(color = '#CB7272'), 
    fill='tozeroy', fillcolor = '#4F7081')

    fig.add_trace(trace, secondary_y=False)
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='Red', rangeslider_visible=True)
    fig.update_yaxes(range = [0, 6], showline=True, linewidth=2, linecolor='black', gridcolor='black');
    data = [trace]
    layout = dict(autosize=False,
                      width=1000,
                      height=550,
                      title = "Évolution du cours de l'action "+ str(c)+" depuis " + str(df.iloc[0, 0]),
    xaxis= dict(title= 'Date', showgrid=False,showline=True),
    yaxis= dict(title= "Cours de l'action",ticklen= 5, zeroline= True, showline=True, showgrid=False), plot_bgcolor='#404754')
    fig = dict(data = data, layout = layout)
    #fig.update_xaxes()
    st.plotly_chart(fig) 
   
    df.set_index('Date', inplace=True)

st.header("** 2 - Méthodologie**")
if st.checkbox('Afficher la méthodologie'):
    #st.markdown("Parler des **RNN**, **LSTM** et de la méthodologie globale")
    st.markdown(" - **Pourquoi les RNNs** ?")
    
    st.markdown("""Les RNNs sont des types de réseaux de neurones de plus en plus utilisés. Ils peuvent prendre en inputs des 
                **audios**, des **chaines de caractères** et même des **documents** d'où leur intérêt grandissant en NLP avec les applications 
                en traduction automatique ou aussi le speech-to-text.""")
                
    st.markdown("""Ils offrent une vraie flexibilité quand à leur utilisation, ils ont aussi l'avantage de pouvoir 'prédire le futur'.
                """)
                
    st.markdown(" - **Comment fonctionnent les RNNs** ?")
    
    st.markdown("""Un RNN composé d'un neurone,  reçoit des entrées, produisant une sortie, et renvoit cette sortie à lui-même.
                Ainsi à chaque période t aussi appelé *frame*, ce RNN reçoit un input **Xt** et la sortie précédente **Yt-1** auxquels 
                on associe des poids **Wx** et **Wy**.""")
    
    st.markdown("""Dans notre cas, cette *frame* sera égale à 60 jours""")
    st.markdown("""En voici une illustration""")
    col1, col2 = st.columns([2, 1])
    col1.image("https://raw.githubusercontent.com/aadmberrada/test/master/img/Rnn.png")
    col2.image("https://raw.githubusercontent.com/aadmberrada/test/master/img/rnn2.png")
    
    st.markdown("""En raison des transformations que subissent les données lorsqu'elles traversent un
    RNN, certaines informations sont perdues à chaque pas de temps. Au bout d'un moment, les RNN
    ne contiennent pratiquement aucune trace des premières entrées, ce qui est problématique.""")
    
    st.markdown("""Pour y remedier, on va utiliser des couches **LSTM** *Long Short-Term Memory* dans le réseau de RNN.
    Elles sont semblables à une cellule de base, sauf qu'elles fonctionneront mieux ; l'algo
    converge plus rapidement, et il détectera les dépendances à long terme dans les données.""")
    
    #st.latex(r'''\begin{equation}\mathbf{y}{(t)}=\phi\left(\mathbf{W}{x}^{\top} \mathbf{x}{(t)}+\mathbf{W}{y}^{\top} \mathbf{y}_{(t-1)}+\mathbf{b}\right)\end{equation}''')
    




st.header("** 3 - Modélisation**")
if st.checkbox('Afficher la modélisation '):
        
    st.markdown(" - **Quelle sera la forme de notre réseau de neurones** ?")
    
    st.markdown("""On va contruire un réseau de neurones composé de 4 couches.""")
    
    st.markdown("""La couche L0 sera une couche LSTM avec l'option return_sequences=True afin d'avoir la recurrrence, 
                une fonction d'activation **tanh** et 10 neurones par défaut""")
    
    st.markdown("""La couche L1 sera une couche LSTM avec l'option return_sequences=False (car on mettre une couche Dense après), une fonction d'activation **tanh** et 10 neurones par défaut""")
    
    st.markdown("""La couche L2 sera une couche Dense avec une fonction d'activation **tanh** et et 10 neurones par défaut""")
    
    st.markdown("""La couche L3 est la couche Output finale, avec un seul neurone et une fonction d'activation **tanh**.""")
    
    st.graphviz_chart('''
        digraph {
            Actif -> Periode
            Periode -> Scaler
            Scaler -> Layer 0 - LSTM
            Layer 0 -> Layer 1 - LSTM
            Layer 1 - LSTM -> Layer 2 - DENSE
            Layer 2 - DENSE -> RUN
        }''')
    
    st.markdown(" - **Résultats et performances** ?")
    

    st.write("Quel sera le cours de l'action " +str(c)+ " au **lendemain** du ", dff.iloc[-1, 0], " ?")
    #----------------------------- 
    
    #----------------------------- Modélisations
    #Preprocessing
    st.write("Dimensions du train et du test")
    def preprocessing(scale):
        
        data = df.filter(["Close"])
        data_arr = data.values
        train_len = math.ceil(len(data_arr)*0.8)
        
        
        if scale == "MinMaxScaler":
            sc = MinMaxScaler
        else:
            sc = StandardScaler
        
        scaler = sc(feature_range=(0,1))
        data_scaled = scaler.fit_transform(data_arr)
        
        # ==>  Train set 
        train = data_scaled[0:train_len, :]
        X_train = []
        y_train = []
        
        for i in range(60, len(train)):
          X_train.append(train[ i - 60 : i, 0])
          y_train.append(train[i, 0])
        
        #Conversion en array
        X_train, y_train = np.array(X_train), np.array(y_train)
        
        #reshape
        #On veut des 3D or on a des 2D
        col4, col5 = st.columns(2)
        
        with col4:
            st.markdown("**Train set**")
            #st.write("Le train set contient", train_len, "observations")
            st.write(X_train.shape, "avant reshape")
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            st.write(X_train.shape, "après reshape")
        
        
        # ==> Test set
        test = data_scaled[train_len - 60 : , : ]
        
        X_test = []
        y_test = data_arr[train_len:, :]
        
        for i in range(60, len(test)):
          X_test.append(test[i - 60 : i, 0])
        
        #Conversion en array puis en 3D
        with col5:
            st.markdown("**Test set**")
            X_test = np.array(X_test)
            st.write(X_test.shape, "avant reshape")
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            st.write(X_test.shape, "après reshape")
        
        return data, X_train, X_test, y_train, y_test, scaler, train_len
    
    
    # ==> Model
    
    #LSTM ARN
    
    #st.warning('This is a warning')
    
    def model():
            
        model = Sequential()
        model.add(LSTM(layer0, activation=af0, return_sequences=True, input_shape = (X_train.shape[1], 1)))
        model.add(LSTM(layer1, activation=af1, return_sequences=False))
        model.add(Dense(layer2, activation=af2))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        model.fit(X_train, y_train, batch_size=1, epochs = 1)
        
        #Predictions
        y_pred = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred)
        
        #metrics
        rmse = np.sqrt(np.mean((y_test - y_pred)**2))
        st.write("RMSE %.2f" %rmse)
        
        return y_pred, model
    

st.info("Pour faire tourner le modèle, veuillez cocher les cases **Données, Méthodologie et Modélisations** puis choisir des paramètres dans le menu à gauche !")

if st.sidebar.button('Run the model'):
    
        st.spinner(text="L'opération peut prendre quelques minutes...")
        #st.write("-----L'opération peut prendre quelques minutes-----")
        
        t1 = time.time()
        data, X_train, X_test, y_train, y_test, scaler, train_len = preprocessing(scale)
        y_pred,  model = model()
        
        t2 = time.time() - t1
        
        st.write("L'opération a pris %.2f" %t2, "secondes.")
        tr = data[:train_len]
        val = data[train_len:]
        val["Pred"] = y_pred
        val = val.reset_index()
        tr = tr.reset_index()

        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        
        trace1 = go.Scatter(
        x = tr["Date"],
        y = tr["Close"],
        mode = "lines",
        name = "Données de d'entrainement",
        marker = dict(color = 'black'))
        
        trace2 = go.Scatter(
        x = val["Date"],
        y = val["Close"],
        mode = "lines",
        name = "Données de test",
        marker = dict(color = '#CB7272'))
        
        trace3 = go.Scatter(
        x = val["Date"],
        y = val["Pred"],
        mode = "lines",
        name = "données prédites",
        marker = dict(color = '#4F7081'))
        fig2.add_trace(trace2, secondary_y=False)
        fig2.add_trace(trace3, secondary_y=False)
        
        fig2.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='Red', rangeslider_visible=True)
        fig2.update_yaxes(range = [0, 6], showline=True, linewidth=2, linecolor='black', gridcolor='black');
        
        data = [trace1, trace2, trace3]
        data1 = [trace2, trace3]
        layout = dict(autosize=False,
                          width=1000,
                          height=550,
                          title = "Prédiction sur les données de test à partir du modèle",
        xaxis= dict(title= 'Date', showgrid=False,showline=True),
        yaxis= dict(title= "Cours de l'action",ticklen= 5, zeroline= True, showline=True, showgrid=False), plot_bgcolor='#404754')

        layout1 = dict(autosize=False,
                  width=1200,
                  height=550,
                  title = "Zoom sur la période de test",
        xaxis= dict(title= 'Date', showgrid=False,showline=True),
        yaxis= dict(title= "Cours de l'action",ticklen= 5, zeroline= True, showline=True, showgrid=False), plot_bgcolor='#404754')
        fig2 = dict(data = data, layout = layout)
        fig3 = dict(data = data1, layout = layout1)
        st.plotly_chart(fig2)     
        st.plotly_chart(fig3) 

        
        
        df1_close = df.filter(["Close"])
        last_60 = df1_close[-60:].values
        last_60_scl = scaler.transform(last_60)
        
        x_test = []
        x_test.append(last_60_scl)
        x_test = np.array(x_test)
        
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        pred = model.predict(x_test)
        pred = scaler.inverse_transform(pred)
        st.write("Sur la base de ce modèle l'action", c, "coutera demain", pred[0][0], "$ en sachant qu'il coûte actuellement", dff.iloc[-1, 5], "$")
#----------------------------- 












