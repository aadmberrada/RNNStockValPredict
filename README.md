# StockValPred

Projet fait par :
  - [Abdoul Aziz Berrada](https://github.com/aadmberrada)
  - [Amira Slimene](https://github.com/aslimene)

But du projet : 

Générer un outil clé en main qui puisse permettre de sortir des statistiques d'entreprises côtées ainsi que de modéliser par une technique de Deep Learning ici les RNN (Recurrent Neural Networks) avec des couches de LSTM (Long Short Term Memory) la valeur du titre de l'entreprise lors de la prochaine journée de cotation.

Toutes les données utilisées dans ce projet sont générées par Webscrapping.

La présentation est faite sous streamlit. Elle contient 4 parties :

  1 - Statistiques 

Le bouton "Voir les statistiques" permet d'afficher principalement 2 infos sur 8 titres. Il s'agit de la variation du cours de fermeture ainsi que de la variation du volume d'échange du cours avec à chaque fois un sens de variation (soit hausse au baisse).

*Une erreur HTTPS peut subvenir à cause des restrictions dûes au webscrapping. Si les données n'apparaissent pas et que l'erreur survient, veuillez décocher la case 'Voir les statistiques' et passer.

  2 - Données

À ce niveau, on montre l'évolution d'un titre. L'utilisateur devra choisir dans la barre latérale (Étape I) un titre et une période d'étude.

Une courbe d'évolution est générée de même que des données de l'entreprise. Ces données fournies en format trimestriel et annuel et concernent les données financières, le cash flow, la balance (document comptable).

  3 - Méthodologie

Cette partie est détaillée dans la présentation.

  4 - Modélisations

Dans cette partie on fournit à l'utilisateur des options dans la barre latérale (Étape II) pour pouvoir faire tourner un modèle de prédiction. 
Les options sont le choix du scaler, du nombre de neurones des couches du réseau ainsi que les différentes fonction d'activation. Une fois les choix faits, l'utilisateur devra appuyer sur le bouton "Run the model" pour faire tourner le modèle. 


La présentation est disponible [ici](https://share.streamlit.io/aadmberrada/rnnstockvalpredict/RNNStockValPred.py)


[![IMAGE ALT TEXT HERE](https://www.youtube.com/watch?v=kUG57sHId8g/0.jpg)](https://www.youtube.com/watch?v=kUG57sHId8g)