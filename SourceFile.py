# Databricks notebook source
# MAGIC %md
# MAGIC ## Import

# COMMAND ----------

#pip install tensorflow
#pip install plotly
#pip install bs4
#pip install keras
#pip install lxml

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import re
from io import StringIO
from datetime import datetime, timedelta
import requests
import math
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots  
from plotly.offline import iplot
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import json
import csv
from bs4 import BeautifulSoup
import lxml
from lxml import html
from io import StringIO
import pyspark
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import round

# COMMAND ----------

# MAGIC %md
# MAGIC ## Données historiques

# COMMAND ----------

# MAGIC %md
# MAGIC On va webscrapper les données historiques de plusieurs valeurs technologiques, il s'agit de 'Apple Inc - AAPL', 'Tesla Inc - TSLA', 'Microsoft Corporation - MSFT', 
# MAGIC                                                                                       'Amazon.com Inc - AMZN', 'Alphabet Inc - GOOGL',
# MAGIC                                                                                   'Meta Platforms Inc - FB' et 'NVIDIA Corporation - NVDA'

# COMMAND ----------

class WebScrapStock:
    
    timeout = 30
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
        df = df[["Date", "Stock", "Open", "High", "Low", "Close", "Adj Close", "Volume"	]]
        return df, url

# COMMAND ----------


#On récupère les données historiques des différentes actions pour 10 années
jours = 360*10
df1, url1 = WebScrapStock(stock_name  = 'AAPL', days_back=jours).get_stock()
df2, url2 = WebScrapStock(stock_name  = 'TSLA', days_back=jours).get_stock()
df3, url3 = WebScrapStock(stock_name  = 'MSFT', days_back=jours).get_stock()
df4, url4 = WebScrapStock(stock_name  = 'AMZN', days_back=jours).get_stock()
df5, url5 = WebScrapStock(stock_name  = 'GOOGL', days_back=jours).get_stock()
df6, url6 = WebScrapStock(stock_name  = 'FB', days_back=jours).get_stock()
df7, url7 = WebScrapStock(stock_name  = 'NVDA', days_back=jours).get_stock()

df8 = df1.append(df2)
df9 = df8.append(df3)
df10 = df9.append(df4)
df11 = df10.append(df5)
df12 = df11.append(df6)
df13 = df11.append(df7)


data_ = df13.copy()
data_

# COMMAND ----------


fig = make_subplots(specs=[[{"secondary_y": True}]])

trace1 = go.Scatter(
x = df2["Date"],
y = df2["Close"],
mode = "lines",
name = "AAPL ",
marker = dict(color = 'grey'))

trace2 = go.Scatter(
x = df2["Date"],
y = df2["Close"],
mode = "lines",
name = "TSLA",
marker = dict(color = 'black'))

trace3 = go.Scatter(
x = df3["Date"],
y = df3["Close"],
mode = "lines",
name = "MSFT ",
marker = dict(color = 'red'))


trace4 = go.Scatter(
x = df4["Date"],
y = df4["Close"],
mode = "lines",
name = "AMZN",
marker = dict(color = 'green'))

trace5 = go.Scatter(
x = df5["Date"],
y = df5["Close"],
mode = "lines",
name = "GOOGL",
marker = dict(color = 'yellow'))


trace6 = go.Scatter(
x = df6["Date"],
y = df6["Close"],
mode = "lines",
name = "FB",
marker = dict(color = 'blue'))


trace7 = go.Scatter(
x = df7["Date"],
y = df7["Close"],
mode = "lines",
name = "NVDA",
marker = dict(color = 'pink'))


fig.add_trace(trace1, secondary_y=True)
fig.add_trace(trace2, secondary_y=True)
fig.add_trace(trace3, secondary_y=True)
fig.add_trace(trace4, secondary_y=True)
fig.add_trace(trace5, secondary_y=True)
fig.add_trace(trace6, secondary_y=True)
fig.add_trace(trace7, secondary_y=True)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='Red', rangeslider_visible=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='black');
data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7]

layout = dict(autosize=False,
                  width=1000,
                  height=550,
                  title = "Évolution du cours de l'action depuis " + str(data_.iloc[0, 0])[:10],
xaxis= dict(title= 'Date', showgrid=False,showline=True),
yaxis= dict(title= "Cours de fermetures",ticklen= 5, zeroline= True, showline=True, showgrid=False))
fig = dict(data = data, layout = layout)
#fig.update_xaxes()
iplot(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC On voit que le titre AMZN a un coût plus élevé que les autres. On remarque aussi une baisse de tous les cours vers 2020 après une ralentissment des marchés financiers. Ensuite s'en suit un décollage, coincidant avec la hausse des valeurs technologiques.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data engineering avec Spark

# COMMAND ----------

df_spark=spark.createDataFrame(data_)
print("La base de données contient", df_spark.count(), "observations et", len(df_spark.columns), "variables.")

df_spark = df_spark.orderBy(['Stock', 'Date'],
                          ascending=False)
df_spark.show()

# COMMAND ----------

#On formatte la date pour enlever les heures et minutes
df_p = pyspark.sql.functions.split(df_spark['Date'], '\s')
df_spark= df_spark.withColumn('Date',df_p.getItem(0))
df_spark.show()

# COMMAND ----------

# On créée un index pour pouvoir déterminer un taux de variation des valeurs de fermeture
df_spark = df_spark.withColumn("id", monotonically_increasing_id())

# COMMAND ----------

my_window = Window.partitionBy().orderBy("id")

#on dréée d'abord 2 colonnes prev et diff qui serviront à calculer les taux
df_spark = df_spark.withColumn("prev_value", F.lag(df_spark.Close).over(my_window))

df_spark = df_spark.withColumn("diff", F.when(F.isnull(df_spark.Close - df_spark.prev_value), 0)
                              .otherwise(df_spark.Close - df_spark.prev_value))

df_spark = df_spark.withColumn('Variations %', F.col('diff') / F.col('Close'))
df_spark.show()

# COMMAND ----------

to_round = ['Open',
 'High',
 'Low',
 'Close',
 'Adj Close',
 'Volume',
 'id',
 'prev_value',
 'diff',
 'Variations %']

for c in to_round:
    df_spark = df_spark.withColumn(c, round(c, 3))


df_spark.show()

# COMMAND ----------

df_spark = df_spark.select(
    ["Date",'Stock', 'Open', 'High', 'Close', 'Adj Close', 'Variations %']
                )
df_spark.show()

# COMMAND ----------

# MAGIC %md
# MAGIC On note sur le haut des données une correction du cours de TSLA

# COMMAND ----------

# MAGIC %md
# MAGIC ## Statistiques des actions

# COMMAND ----------

# MAGIC %md
# MAGIC Pour la suite du travail, on va se concentrer sur l'action AMZN.
# MAGIC 
# MAGIC Toutes les fonctions sont paramétrées et il suffit de changer le paramètre "action" pour analyser d'autres actions.

# COMMAND ----------

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
    print("--Statistiques de l'entreprise "+action+ "--")
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
        print(mylist)
        
    d = pd.DataFrame(json_data['context']['dispatcher']['stores']['QuoteSummaryStore']["defaultKeyStatistics"])
    d = d[d.columns[d.isna().sum()<3]].replace(to_replace = np.NaN, value = "-")

    j = d.to_json(orient="columns")
    parsed = json.loads(j)
    print(json.dumps(parsed, indent=2))

    return 

statistiques("AMZN", print_list = False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Statistiques financières

# COMMAND ----------

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

annual_is, annual_cf, annual_bs, quaterly_is, quaterly_cf, quaterly_bs = financial_inf("AMZN")

# COMMAND ----------

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

# COMMAND ----------

annual_is_stmts

# COMMAND ----------

pd.DataFrame(quaterly_is_stmts)

# COMMAND ----------

pd.DataFrame(annual_is_stmts)

# COMMAND ----------

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
annual_cf_stmts

# COMMAND ----------

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
annual_bs_stmts

# COMMAND ----------

# MAGIC %md
# MAGIC ##Modélisations avec données historiques de AMAZON

# COMMAND ----------

df, url = WebScrapStock('AMZN', days_back=360*15).get_stock()
dff = df.copy()

stock = re.findall(r'[A-Z]+', url)[0]
period1 = re.findall(r'\d{5,}', url)[0]
period2 = re.findall(r'\d{5,}', url)[1]

def lien(stock, period1, period2):
    lien = "https://finance.yahoo.com/quote/"+stock+"/history?period1="+period1+"&period2="+period2+"&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true"
    return lien
lien = lien(stock, period1, period2)
print("Vous pouvez voir les données çà cette adresse : ", lien)

# COMMAND ----------

df.describe()

# COMMAND ----------

df.sort_values("Date", ascending=False)

# COMMAND ----------

df.shape

# COMMAND ----------

fig = make_subplots(specs=[[{"secondary_y": True}]])

trace1 = go.Scatter(
x = df4["Date"],
y = df4["Close"],
mode = "lines",
name = "Cours de fermeture AMZN ",
marker = dict(color = 'black'))


fig.add_trace(trace1, secondary_y=False)


fig.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='Red', rangeslider_visible=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='black');
data = [trace1]

layout = dict(autosize=False,
                  width=1000,
                  height=550,
                  title = "Évolution du cours de l'action " +str(df.iloc[0, 1])+ " depuis " + str(dff.iloc[0, 0])[:10],
xaxis= dict(title= 'Date', showgrid=False,showline=True),
yaxis= dict(title= "Cours de l'action",ticklen= 5, zeroline= True, showline=True, showgrid=False))
fig = dict(data = data, layout = layout)
#fig.update_xaxes()
iplot(fig)


# COMMAND ----------

df.set_index('Date', inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Acheter des actions AMZN semble être une bonne idée

# COMMAND ----------

def preprocessing():
    
      """
        Cette fonction permet de transformer les données au format voulu

        returns
        -----
        Les inputs du modèle 
      """

      #X = df.drop(["Close", ])
      data = df.filter(["Close"])
      data_arr = data.values
      train_len = math.ceil(len(data_arr)*0.8)
      #print("Nombre d'observations des données d'entrainement", train_len)
      scaler = MinMaxScaler(feature_range=(0,1))
      data_scaled = scaler.fit_transform(data_arr)
      train = data_scaled[0:train_len, :]

      X_train = []
      y_train = []

      for i in range(60, len(train)):
        X_train.append(train[ i - 60 : i, 0])
        y_train.append(train[i, 0])

      #Conversion en array
      X_train, y_train = np.array(X_train), np.array(y_train)

      #Train set
      #reshape
      #On veut des 3D or on a des 2D
      print("--Train--")
      print(X_train.shape, "avant reshape")
      X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
      print(X_train.shape, "après reshape")

      #Test set

      test = data_scaled[train_len - 60 : , : ]
      X_test = []
      y_test = data_arr[train_len:, :]

      for i in range(60, len(test)):
        X_test.append(test[i - 60 : i, 0])

      #Conversion en array puis en 3D
      X_test = np.array(X_test)
      print("--Test--")
      print(X_test.shape, "avant reshape")
      X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
      print(X_test.shape, "après reshape")

      return X_train, y_train, X_test, y_test, train_len, scaler, data

X_train, y_train, X_test, y_test, train_len, scaler, data = preprocessing()

# COMMAND ----------

def modelisation():
      #LSTM ARN
    """
        Cette fonction entraine un modèle RNN afin de faire de la simulation in-sample des données du titre choisi 

        returns
        -----
        Les prédictions et le modèle entrainé
    """


    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape = (X_train.shape[1], 1)))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(100))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs = 1)

    #Predictions
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)

    #metrics
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    print("RMSE du modèle", rmse)

    return y_pred, model

y_pred, model = modelisation()

# COMMAND ----------

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
marker = dict(color = 'blue'))

trace3 = go.Scatter(
x = val["Date"],
y = val["Pred"],
mode = "lines",
name = "données prédites",
marker = dict(color = '#CB7272'))
fig2.add_trace(trace2, secondary_y=False)
fig2.add_trace(trace3, secondary_y=False)

fig2.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='Red', rangeslider_visible=True)
fig2.update_yaxes( showline=True, linewidth=2, linecolor='black', gridcolor='black');

data = [trace1, trace2, trace3]
data1 = [trace2, trace3]
layout = dict(autosize=False,
                  width=1200,
                  height=550,
                  title = "Évolution du cours de l'action "+ str(dff.iloc[0, 1])+ " depuis " + str(dff.iloc[0, 0])[:10],
xaxis= dict(title= 'Date', showgrid=False,showline=True),
yaxis= dict(title= "Cours de l'action",ticklen= 5, zeroline= True, showline=True, showgrid=False))

layout1 = dict(autosize=False,
                  width=1200,
                  height=550,
                  title = "Zoom sur la période de test",
xaxis= dict(title= 'Date', showgrid=False,showline=True),
yaxis= dict(title= "Cours de l'action",ticklen= 5, zeroline= True, showline=True, showgrid=False))
fig2 = dict(data = data, layout = layout)
fig3 = dict(data = data1, layout = layout1)
iplot(fig2)
iplot(fig3)


# COMMAND ----------

def predict_next_val():
  
    """
        Cette fonction permet de prédire la valeu du titre le prochain jour de cotation

        returns
        -----
        None
    """
    df1_close = df.filter(["Close"])

    last_60 = df1_close[-60:].values
    last_60_scl = scaler.transform(last_60)

    x_test = []
    x_test.append(last_60_scl)
    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    pred = model.predict(x_test)
    pred = scaler.inverse_transform(pred)

    print("La valeur de l'action pour le prochain jour de cotation est",pred[0][0], "$ en sachant qu'il est côté actuellement %.3f" %dff.iloc[-1, 5], "$")

    return 
predict_next_val()
