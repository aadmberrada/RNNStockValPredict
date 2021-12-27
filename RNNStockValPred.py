#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 15:30:16 2021

@author: Abdoul_Aziz_Berrada
"""

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
from bs4 import BeautifulSoup
import json
from lxml import html
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#import graphviz as graphviz
#import tensorflow
#import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

st.set_option('deprecation.showPyplotGlobalUse', False)



#--------------------------------- Head
## ===> Header
#st.title("")
st.markdown("<h1 style='text-align: center;'> Stock's Value Prediction using RNNs</h1>", unsafe_allow_html=True)
#st.title("Stock Value Prediction using Neural Networks")
col1, col2, col3 = st.columns([0.75, 0.75, 0.75])

col1.image('https://raw.githubusercontent.com/aadmberrada/test/master/img/mosef3.png')

col2.image("https://raw.githubusercontent.com/aadmberrada/test/master/img/YahooFinance.jpeg")

col3.image("https://raw.githubusercontent.com/aadmberrada/test/master/img/py.png")

st.markdown("""
            
Ce projet a pour but de recueillir par webscrapping des données historiques d'entreprises cotées et puis de faire une analyse predictive in-sample en utilisant les Reccurent Neural Networks RNNs.""")

st.markdown(""" ==> Pre requis :
*   **Python :** pandas, numpy, pyplot, tensorflow, keras, sklearn, streamlit, BeautifulSoup.

*   **Source des données :** https://finance.yahoo.com/quote/yhoo/history/""")

st.info("Pour commencer, veuiller cocher les différentes cases !")
#------------------------------------ Sidebar

st.sidebar.subheader("Abdoul Aziz Berrada - Amira Slimene")

st.sidebar.markdown("------------------------------------")

st.sidebar.title("User Input Features")


#--------------
## ===> Dataset

st.sidebar.subheader(" Étape I -  Chargement des données")

action = st.sidebar.selectbox("Selectionnez un actif à étudier", ['Apple Inc - AAPL', 'Tesla Inc - TSLA', 'Microsoft Corporation - MSFT', 
                                                                    'Amazon.com Inc - AMZN', 'Alphabet Inc - GOOGL',
                                                                    'Meta Platforms Inc - FB', 'NVIDIA Corporation - NVDA', 'Netflix - NFLX'])
stk = action.split(" ")[-1]
c = re.findall(r'[A-Za-z/./ /]+', action)[0]
jours = st.sidebar.slider("Selectionnez un nombre de jours d'étude", min_value = 2*360, max_value = 15*360, step = 180)

#-------------
##  ===> Deep Learning model

st.sidebar.subheader(" Étape II -  Modélisation")

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
        #st.write("On va récupérer les données de", c, "depuis", df.iloc[0, 0], "soit sur", df.shape[0], "jours de marché.")
        return df, url
    
st.header("** 1 - Statistiques**")
#st.subheader("** 1.1 - Statistiques **")
if st.checkbox("Voir les statistiques"):

    st.warning("Si les données n'apparaissent pas et qu'une erreur survient, veuillez décocher la case 'Voir les statistiques' et passer.")

    
    jours = jours
    df1, url1 = WebScrapStock(stock_name  = 'AAPL', days_back=jours).get_stock()
    df2, url2 = WebScrapStock(stock_name  = 'TSLA', days_back=jours).get_stock()
    df3, url3 = WebScrapStock(stock_name  = 'MSFT', days_back=jours).get_stock()
    df4, url4 = WebScrapStock(stock_name  = 'AMZN', days_back=jours).get_stock()
    df5, url5 = WebScrapStock(stock_name  = 'GOOGL', days_back=jours).get_stock()
    df6, url6 = WebScrapStock(stock_name  = 'FB',  days_back=jours).get_stock()
    df7, url7 = WebScrapStock(stock_name  = 'NVDA', days_back=jours).get_stock()
    df8, url8 = WebScrapStock(stock_name  = 'NFLX', days_back=jours).get_stock()
    
    st.markdown("Quelques statistiques sur les cours des différents titres")
    col10, col11, col12, col13 = st.columns(4)

    col14, col15, col16, col17 = st.columns(4)

    def var(df__):
        x = df__.shape[0] - 1
        y = df__.shape[1] - 3
        x1 = df__.shape[0] - 2
        y1 = y + 2
        d = 100*(df__.iloc[x , y ]  / df__.iloc[ x1, y ]  - 1)
        e = 100*(df__.iloc[x , y1 ] /  df__.iloc[ x1, y1 ] - 1)
        return d, e, x, y, y1
    
    d_1, e_1, x1, y1, y11 = var(df1)
    d_2, e_2, x2, y2, y12 = var(df2)
    d_3, e_3, x3, y3, y13 = var(df3)
    d_4, e_4, x4, y4, y14 = var(df4)
    d_5, e_5, x5, y5, y15 = var(df5)
    d_6, e_6, x6, y6, y16 = var(df6)
    d_7, e_7, x7, y7, y17 = var(df7)
    d_8, e_8, x8, y8, y18 = var(df8)
    
    col10.metric(label= "AAPL" + "'s Price", value=  "{:.2f}".format(df1.iloc[x1 , y1 ]), delta="{:.2f}".format(d_1)+"%")
    col10.metric(label= "AAPL" + "'s Volume", value=  "{:.0f}".format(df1.iloc[x1 , y11 ]), delta="{:.2f}".format(e_1)+"%")

    col11.metric(label= 'TSLA' + "'s Price", value=  "{:.2f}".format(df2.iloc[x2 , y1 ]), delta="{:.2f}".format(d_2)+"%")
    col11.metric(label= 'TSLA' + "'s Volume", value=  "{:.0f}".format(df2.iloc[x2 , y12 ]), delta="{:.2f}".format(e_2)+"%")

    col12.metric(label= 'MSFT' + "'s Price", value=  "{:.2f}".format(df3.iloc[x3 , y3 ]), delta="{:.2f}".format(d_3)+"%")
    col12.metric(label= 'MSFT' + "'s Volume", value=  "{:.0f}".format(df3.iloc[x3 , y13 ]), delta="{:.2f}".format(e_3)+"%")

    col13.metric(label= 'AMZN' + "'s Price", value=  "{:.2f}".format(df4.iloc[x4 , y4 ]), delta="{:.2f}".format(d_4)+"%")
    col13.metric(label= 'AMZN' + "'s Volume", value=  "{:.0f}".format(df4.iloc[x4 , y14 ]), delta="{:.2f}".format(e_4)+"%")

    col14.metric(label= 'GOOGL' + "'s Price", value=  "{:.2f}".format(df5.iloc[x5 , y5 ]), delta="{:.2f}".format(d_5)+"%")
    col14.metric(label= 'GOOGL' + "'s Volume", value=  "{:.0f}".format(df5.iloc[x5 , y15 ]), delta="{:.2f}".format(e_5)+"%")

    col15.metric(label= 'FB' + "'s Price", value=  "{:.2f}".format(df6.iloc[x6 , y6 ]), delta="{:.2f}".format(d_6)+"%")
    col15.metric(label= 'FB' + "'s Volume", value=  "{:.0f}".format(df6.iloc[x6 , y16 ]), delta="{:.2f}".format(e_6)+"%")

    col16.metric(label= 'NVDA' + "'s Price", value=  "{:.2f}".format(df7.iloc[x7 , y7 ]), delta="{:.2f}".format(d_7)+"%")
    col16.metric(label= 'NVDA' + "'s Volume", value=  "{:.0f}".format(df7.iloc[x7 , y17 ]), delta="{:.2f}".format(e_7)+"%")

    col17.metric(label= 'NFLX' + "'s Price", value=  "{:.2f}".format(df8.iloc[x8 , y8 ]), delta="{:.2f}".format(d_8)+"%")
    col17.metric(label= 'NFLX' + "'s Volume", value=  "{:.0f}".format(df8.iloc[x8 , y18 ]), delta="{:.2f}".format(e_8)+"%")



st.header("** 2 - Données**")
if st.checkbox('Afficher les données'):
    st.info("Pour commencer, veuiller choisir un titre et une période d'étude dans 'Étape - I' dans le menu de gauche ")

    st.write("Vous aveez choisi",  action, "et une période de", jours, "jours soit", df.shape[0], "cotations.")

    df, url = WebScrapStock(stock_name  = stk, days_back=jours).get_stock()
    
    dff = df.copy()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    trace1 = go.Scatter(
    x = df["Date"],
    y = df["Close"],
    mode = "lines",
    name = "Cours de fermeture " + str(stk),
    marker = dict(color = '#CB7272'), 
    fill='tozeroy', fillcolor = '#4F7081')
  
    fig.add_trace(trace1, secondary_y=False)


    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='Red')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='black')
    data = [trace1]

    layout = dict(autosize=False,
                    width=1000,
                    height=550,
                    title = "Évolution du cours de l'action "+str(stk)+" depuis " + str(dff.iloc[0, 0])[:10],
    xaxis= dict(title= 'Date', showgrid=False,showline=True),
    yaxis= dict(title= "Cours de l'action",ticklen= 5, zeroline= True, showline=True, showgrid=False), plot_bgcolor='#404754')
    fig = dict(data = data, layout = layout)
    #fig.update_xaxes()
    st.plotly_chart(fig)
   


    stock = re.findall(r'[A-Z]+', url)[0]
    period1 = re.findall(r'\d{5,}', url)[0]
    period2 = re.findall(r'\d{5,}', url)[1]

    def lien_url(stock, period1, period2):
        lien = "https://finance.yahoo.com/quote/"+stock+"/history?period1="+period1+"&period2="+period2+"&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true"
        return lien
    lien = lien_url(stock, period1, period2)
    
    #st.markdown("""Les données sont directement téléchargeables en format CSV à l'adresse [Yahoo Finance](lien)""")
    #st.write(lien)
    #st.write("Les données sont directement téléchargeables en format CSV à l'adresse : ", [lien](lien))  
    
    col1, col2 = st.columns([4, 1])
    
    x = df.shape[0] - 1
    y = df.shape[1] - 3
    x1 = df.shape[0] - 2
    y1 = y + 2
    d = 100*(df.iloc[x , y ]  / df.iloc[ x1, y ]  - 1)
    e = 100*(df.iloc[x , y1 ] /  df.iloc[ x1, y1 ] - 1)
    #d = (df.iloc[df.shape[0] - 1 , df.shape[5]] - df.iloc[df.shape[0], df.shape[5]])/df.iloc[df.shape[0], df.shape[5]]
        
    
    #col2.metric(label= str(c) + "'s Price", value=  "{:.2f}".format(df.iloc[x , y ]), delta="{:.2f}".format(d)+"%")
    #col2.metric(label= str(c) + "'s Volume", value=  "{:.0f}".format(df.iloc[x , y1 ]), delta="{:.2f}".format(e)+"%")
    
    #col1.dataframe(
    st.dataframe(df.sort_values("Date", ascending=False))
    
    #st.write('Statistiques descriptives des actions', c)
    #st.table(df.describe())

    def statistiques(action, print_list = False):
        
        """
        Cette fonction permet de générer des statistiques d'une entreprise
        
        Parameters
        ----------
        action : str
            Nom du titre
        print_list : bool, , default=False
            Affiche ou non la liste des statistiques de base
        returns
        -----
            Des statistiques au format json
        """
        ## Stats
        url_stats = "https://finance.yahoo.com/quote/{}/key-statistics?p={}"
        #st.write("--Statistiques de l'entreprise " + action + "--")
        stock = action
        headers_={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
        response = requests.get(url_stats.format(stock, stock), headers = headers_)
        soup = BeautifulSoup(response.text, 'html.parser')
        pattern = re.compile(r'\s--\sData\s--\s')
        script_data = soup.find('script', text=pattern).contents[0]
        start = script_data.find("context") - 2
        json_data = json.loads(script_data[start:-12])
        #json_data['context'].keys()
        #json_data['context']['dispatcher']['stores']['QuoteSummaryStore'].keys()
        #json_data['context']['dispatcher']['stores']['QuoteSummaryStore']["defaultKeyStatistics"].keys()
        mylist = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']["defaultKeyStatistics"]

        if print_list == True:
            st.write(mylist)
            
        d = pd.DataFrame(json_data['context']['dispatcher']['stores']['QuoteSummaryStore']["defaultKeyStatistics"])
        d = d[d.columns[d.isna().sum()<3]].replace(to_replace = np.NaN, value = "-")

        j = d.to_json(orient="columns")
        parsed = json.loads(j)
        j_1 = json.dumps(parsed, indent=2)

        #j_1 = parsed
        return j_1

    j_1 = statistiques(str(stk), print_list = False)

    st.markdown("<h4 style='text-align: center;'>STATISTIQUES DE BASE</h4>", unsafe_allow_html=True)
    st.json(j_1)
   

    def financial_inf(action):

        """
        Cette fonction permet de générer des données financières d'une entreprise
        

        Parameters
        ----------
        action : str
            Nom du titre
        returns
        -----
            Les données de revenu, de la balance comptable et du cash flow au format annuel et trimestriel au format json
        """
        url_fin = "https://finance.yahoo.com/quote/{}/financials?p={}"
        stock = action
        headers_={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
        response = requests.get(url_fin.format(stock, stock), headers = headers_)
        soup = BeautifulSoup(response.text, 'html.parser')
        pattern = re.compile(r'\s--\sData\s--\s')
        script_data = soup.find('script', text=pattern).contents[0]
        #script_data[:500]
        #script_data[-500:]
        start = script_data.find("context") - 2

        json_data = json.loads(script_data[start:-12])
        #json_data['context'].keys()
        #json_data['context']['dispatcher']['stores']['QuoteSummaryStore'].keys()

        annual_is = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']["incomeStatementHistory"]["incomeStatementHistory"]
        #print(annual_is[0])
        #annual_is[0]["minorityInterest"]
        annual_cf = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']["cashflowStatementHistory"]["cashflowStatements"]
        annual_bs = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']["balanceSheetHistory"]["balanceSheetStatements"]

        quaterly_is = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']["incomeStatementHistoryQuarterly"]["incomeStatementHistory"]
        quaterly_cf = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']["cashflowStatementHistoryQuarterly"]["cashflowStatements"]
        quaterly_bs = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']["balanceSheetHistoryQuarterly"]["balanceSheetStatements"]

        return annual_is, annual_cf, annual_bs, quaterly_is, quaterly_cf, quaterly_bs

    annual_is, annual_cf, annual_bs, quaterly_is, quaterly_cf, quaterly_bs = financial_inf(str(stk))

    def income_stats():
        annual_is_stmts = []
        #annual
        for s in annual_is:
            statement  = {}
            for key, val in s.items():
                try:
                    statement[key] = val['raw']
                except TypeError:
                    continue
                except KeyError:
                    continue
            annual_is_stmts.append(statement)

            #quaterly
        quaterly_is_stmts = []
        for s in quaterly_is:
            statement  = {}
            for key, val in s.items():
                try:
                    statement[key] = val['raw']
                except TypeError:
                    continue
                except KeyError:
                    continue
            quaterly_is_stmts.append(statement)


        return annual_is_stmts, quaterly_is_stmts

    annual_is_stmts, quaterly_is_stmts = income_stats()

    st.markdown("<h4 style='text-align: center;'>INCOME STATEMENTS</h4>", unsafe_allow_html=True)
    #st.write("")
    col20, col21 = st.columns(2)
    with col20:
        st.write("Annuel")
        st.json(annual_is_stmts[0])
    
    with col21:
        st.write("Trimestriel")
        st.json(quaterly_is_stmts[0])
    


    def clash_flow():
        annual_cf_stmts = []
        #annual
        for s in annual_cf:
            statement  = {}
            for key, val in s.items():
                try:
                    statement[key] = val['raw']
                except TypeError:
                    continue
                except KeyError:
                    continue
            annual_cf_stmts.append(statement)

        #quaterly
        quaterly_cf_stmts = []
        for s in quaterly_cf:
            statement  = {}
            for key, val in s.items():
                try:
                    statement[key] = val['raw']
                except TypeError:
                    continue
                except KeyError:
                    continue
            quaterly_cf_stmts.append(statement)


        return annual_cf_stmts, quaterly_cf_stmts
        
    annual_cf_stmts, quaterly_cf_stmts = clash_flow()

    #st.write("CASH LOWS")
    st.markdown("<h4 style='text-align: center;'>CASH LOWS</h4>", unsafe_allow_html=True)
    col22, col23 = st.columns(2)
    with col22:
        st.write("Annuel")
        st.json(annual_cf_stmts[0])
    
    with col23:
        st.write("Trimestriel")
        st.json(quaterly_cf_stmts[0])
    
    
    
    def balance_sheet():
        annual_bs_stmts = []
        #annual
        for s in annual_bs:
            statement  = {}
            for key, val in s.items():
                try:
                    statement[key] = val['raw']
                except TypeError:
                    continue
                except KeyError:
                    continue
                annual_bs_stmts.append(statement)

        #quaterly
        quaterly_bs_stmts = []
        for s in quaterly_bs:
            statement  = {}
            for key, val in s.items():
                try:
                    statement[key] = val['raw']
                except TypeError:
                    continue
                except KeyError:
                    continue
            quaterly_bs_stmts.append(statement)


        return annual_bs_stmts, quaterly_bs_stmts
    
    annual_bs_stmts, quaterly_bs_stmts = balance_sheet()

    #st.write("")
    st.markdown("<h4 style='text-align: center;'>BALANCE SHEET</h4>", unsafe_allow_html=True)


  
    col24, col25 = st.columns(2)
    with col24:
        st.write("Annuel")
        st.json(annual_bs_stmts[0])
    
    with col25:
        st.write("Trimestriel")
        st.json(quaterly_bs_stmts[0])

    df.set_index('Date', inplace=True)

st.header("** 3 - Méthodologie**")
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
    
    st.latex(r'''\mathbf{y}{(t)}=\phi\left(\mathbf{W}{x}^{\top} \mathbf{x}{(t)}+\mathbf{W}{y}^{\top} \mathbf{y}_{(t-1)}+\mathbf{b}\right)''')
    




st.header("** 4 - Modélisation**")
if st.checkbox('Afficher la modélisation '):
    st.info("Pour faire tourner le modèle, veuillez d'abord cocher les cases **Données et Méthodologie** puis choisir des paramètres dans le menu à gauche !")

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
        
        return y_pred, rmse, model
    


if st.sidebar.button('Run the model'):
    
        st.spinner(text="L'opération peut prendre quelques minutes...")
        #st.write("-----L'opération peut prendre quelques minutes-----")
        
        t1 = time.time()
        data, X_train, X_test, y_train, y_test, scaler, train_len = preprocessing(scale)
        y_pred, rmse,  model = model()
        
        t2 = time.time() - t1
        
        st.write("L'opération a pris %.2f" %t2, "secondes.")

        if rmse < 15:
            st.success('Au regard de la RMSE, le modèle selectionné est performant!')
        else:
            st.warning("Au regard de la RMSE, le modèle selectionné n'est pas performant!")

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
        st.write("Sur la base de ce modèle l'action", c, "coutera le prochain jour de cotation", pred[0][0], "$ en sachant qu'il est côté actuellement", dff.iloc[-1, 5], "$")
#----------------------------- 












