# -*- coding: utf-8 -*-
#sklearn possui diversos algoritmos de Machine Learning
#O objeto df é um DataFrame, nele está os dados carregados e diversas funções para manipular os dados
#n_estimators numero de estimadores(arvores de decisão) que serão utilizados pelo random forest
#random_state parâmetro comum na maioria dos algoritmos e é importante mantê-lo fixo, o valor não importa, desde que seja sempre o mesmo, dessa forma conseguiremos gerar sempre o mesmo modelo com os mesmos dados
#criterion é a métrica utilizada para construção da árvore de decisão. Pode ser gini(pode ser utilizado para medir a impureza de um nó) ou entropy
#loc acessa um grupo de linhas e colunas por rótulo (s) ou uma matriz booleana
#values retornar uma representação Numpy do DataFrame
#dtype retorna os dtypes no DataFrame


import numpy as np #facilitar trabalho com arrays e tem diversas funções de álgebra, estatística e computação científica no geral 
 #separar uma parte dos dados para o treinamento
from sklearn.model_selection import cross_val_score #serve para ajudar na validação do modelo



import matplotlib.pyplot as plt #Utilizada para plotar gráficos, é uma ótima ferramenta na análise dos dados
import pandas as pd #estruturar os dados, ajuda na analise e na manipulação de grande quantidade de dados

#X dado de treinamento
#y dado de treinamento

import pandas as pd
dataset = pd.read_csv('/home/vinicius/Documents/IA/Python/credit_data1.csv') #ler o arquivo em formato csv

#dataset = dataset.drop('cliente_id',axis=1) #excluir a coluna cliente_id
#dataset =  #excluir linhas que tenham valores ausentes
#rotulo = dataset.iloc[:,1:4].values #armazenar a coluna rótulo
dataset = dataset.drop("rotulo",axis=1) ##excluir a coluna rotulo dos atributos

import pandas as pd
base = pd.read_csv('/home/vinicius/Documents/IA/Python/credit_data1.csv')
base = base.dropna()
base.drop(base[base.idade<0].index ,inplace=True)



previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values








#idade = dataset.loc[(dataset["idade"]<0) , ["idade"]]
#dataset.idade.values
#dataset = pd.to_numeric(dataset.idade, errors='ignore')
#dtype = np.float32



from sklearn.cross_validation import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(
previsores, classe, test_size = 0.30, random_state=0)



from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(previsores_treinamento, classe_treinamento)
pred = classifier.predict(previsores_teste)




from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 40, criterion = 'entropy', random_state = 0)
classifier.fit(previsores_treinamento, classe_treinamento)
pred_random = classifier.predict([[66155.9250950813
,18,8.77e+03]])
print( 'Risco de Crédito =', pred_random)


from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, pred_random)
matriz = confusion_matrix(classe_teste, pred_random)


dtype = np.float32
dataset.loc[(dataset["idade"] < 0),["idade"]]
dataset.idade.values
pd.to_numeric(dataset.idade, errors='ignore')
dataset.dtypes


#pandas.to_numeric(arg, errors='raise', downcast=None)


