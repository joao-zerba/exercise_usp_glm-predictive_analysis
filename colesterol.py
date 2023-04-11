# UNIVERSIDADE DE SÃO PAULO
# INTRODUÇÃO AO PYTHON E MACHINE LEARNING
# Exercícios de Predictive Analytics (Supervised Machine Learning)
# Prof. Dr. Luiz Paulo Fávero

#!/usr/bin/env python
# coding: utf-8


# In[ ]: Importação dos pacotes necessários
    
import pandas as pd # manipulação de dado em formato de dataframe
import seaborn as sns # biblioteca de visualização de informações estatísticas
import matplotlib.pyplot as plt # biblioteca de visualização de dados
import statsmodels.api as sm # biblioteca de modelagem estatística
import numpy as np # biblioteca para operações matemáticas multidimensionais
from scipy import stats # estatística chi2
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
import plotly.graph_objs as go # gráfico 3D
from scipy.stats import pearsonr # correlações de Pearson
import statsmodels.formula.api as smf # estimação de modelos

# In[ ]:
##############################################################################
#                  EXEMPLO 03 - MODELO LOGÍSTICO MULTINOMIAL                 #
##############################################################################

df_colesterol = pd.read_csv('colesterol.csv',delimiter=',')
df_colesterol

#Características das variáveis do dataset
df_colesterol.info()

#Estatísticas univariadas
df_colesterol.describe()


# In[ ]: Note que a variável Y 'atrasado' está definida como objeto

#Tabela de frequências absolutas da variável 'cat_colesterol' com labels
df_colesterol['cat_colesterol'].value_counts(sort=False)

#Criando uma variável 'cat_colesterol2' a partir da variável 'cat_colesterol',
#com labels iguais a 0, 1, 2, 3 e 4 e com tipo 'int' (poderia também ser do tipo
#'float'), a fim de que seja possível estimar o modelo por meio
#da função 'MNLogit'
df_colesterol.loc[df_colesterol['cat_colesterol']=='Ótimo',
                  'cat_colesterol2'] = 0 #categoria de referência
df_colesterol.loc[df_colesterol['cat_colesterol']=='Subótimo',
                  'cat_colesterol2'] = 1
df_colesterol.loc[df_colesterol['cat_colesterol']=='Limítrofe',
                  'cat_colesterol2'] = 2
df_colesterol.loc[df_colesterol['cat_colesterol']=='Elevado',
                  'cat_colesterol2'] = 3
df_colesterol.loc[df_colesterol['cat_colesterol']=='Muito Elevado',
                  'cat_colesterol2'] = 4

df_colesterol['cat_colesterol2'] = df_colesterol['cat_colesterol2'].astype('int64')


# In[ ]: Estimação do modelo logístico multinomial

from statsmodels.discrete.discrete_model import MNLogit

x = df_colesterol.drop(columns=['cat_colesterol','cat_colesterol2'])
y = df_colesterol['cat_colesterol2']

#Esse pacote precisa que a constante seja definida pelo usuário
X = sm.add_constant(x)

#Estimação do modelo - função 'MNLogit' do pacote
#'statsmodels.discrete.discrete_model'
modelo_colesterol = MNLogit(endog=y, exog=X).fit()

#Parâmetros do modelo
modelo_colesterol.summary()


# In[ ]: Vamos definir uma função 'Qui2' para se extrair a estatística geral
# do modelo

def Qui2(modelo_multinomial):
    maximo = modelo_multinomial.llf
    minimo = modelo_multinomial.llnull
    qui2 = -2*(minimo - maximo)
    pvalue = stats.distributions.chi2.sf(qui2,1)
    df = pd.DataFrame({'Qui quadrado':[qui2],
                       'pvalue':[pvalue]})
    return df


# In[ ]: Estatística geral do 'modelo_colesterol'

Qui2(modelo_colesterol)


# In[ ]: Fazendo predições para o 'modelo_colesterol'

# Exemplo: quais as probabilidades para cada categoria de colesterol,
#se o indivíduo não fumar e praticar atividades esportivas 4 vezes por semana?

#No nosso exemplo, tempos que:
# 0: Ótimo
# 1: Subótimo
# 2: Limítrofe
# 3: Elevado
# 4: Muito Elevado

resultado = modelo_colesterol.predict(pd.DataFrame({'const':[1],
                                                    'cigarro':[0],
                                                    'esporte':[4]})).round(4)

resultado

#Uma maneira de identificar a classe do resultado de acordo com o 'predict'

resultado.idxmax(axis=1)


# In[ ]: Adicionando as probabilidades de ocorrência de cada uma das
#categorias de Y definidas pela modelagem, bem como a respectiva
#classificação, ao dataframe original

#Probabilidades de ocorrência das cinco categoriais
#Definição do array 'phats':
phats = modelo_colesterol.predict()
phats

#Transformação do array 'phats' para o dataframe 'phats':
phats = pd.DataFrame(phats)
phats

#Concatenando o dataframe original com o dataframe 'phats':
df_colesterol = pd.concat([df_colesterol, phats], axis=1)
df_colesterol

# Analisando o resultado de acordo com a categoria de resposta:
predicao = phats.idxmax(axis=1)
predicao

#Adicionando a categoria de resposta 'predicao' ao dataframe original,
#por meio da criação da variável 'predicao'
df_colesterol['predicao'] = predicao
df_colesterol

#Criando a variável 'predicao_label' a partir da variável 'predicao',
#respeitando os seguintes rótulos:
# 0: Ótimo
# 1: Subótimo
# 2: Limítrofe
# 3: Elevado
# 4: Muito Elevado

df_colesterol.loc[df_colesterol['predicao']==0, 'predicao_label'] ='Ótimo'
df_colesterol.loc[df_colesterol['predicao']==1, 'predicao_label'] ='Subótimo'
df_colesterol.loc[df_colesterol['predicao']==2, 'predicao_label'] ='Limítrofe'
df_colesterol.loc[df_colesterol['predicao']==3, 'predicao_label'] ='Elevado'
df_colesterol.loc[df_colesterol['predicao']==4, 'predicao_label'] ='Muito Elevado'

df_colesterol


# In[ ]: Eficiência global do modelo

#Criando uma tabela para comparar as ocorrências reais com as predições
table = pd.pivot_table(df_colesterol,
                       index=['cat_colesterol'],
                       columns=['predicao_label'],
                       aggfunc='size').fillna(0)
table

#Transformando o dataframe 'table' para 'array', para que seja possível
#estabelecer o atributo 'sum'
table = table.to_numpy()
table

#Eficiência global do modelo
acuracia = (74 + 542 + 214)/table.sum()
acuracia


# In[ ]: Plotagens das probabilidades

#Plotagem das smooth probability lines para a variável 'esporte'

# 0: Ótimo
# 1: Subótimo
# 2: Limítrofe
# 3: Elevado
# 4: Muito Elevado

plt.figure(figsize=(10,10))
sns.regplot(x = df_colesterol['esporte'], y = df_colesterol[0],
            data=df_colesterol, order=4, ci=None, color='dodgerblue',
            marker='o', label='Ótimo',
            scatter_kws={'color':'dodgerblue', 's':120})
sns.regplot(x = df_colesterol['esporte'], y = df_colesterol[1],
            data=df_colesterol, order=4, ci=None, color='darkgreen',
            marker='o', label='Subótimo',
            scatter_kws={'color':'darkgreen', 's':120})
sns.regplot(x = df_colesterol['esporte'], y = df_colesterol[2],
            data=df_colesterol, order=4, ci=None, color='darkorange',
            marker='o', label='Limítrofe',
            scatter_kws={'color':'darkorange', 's':120})
sns.regplot(x = df_colesterol['esporte'], y = df_colesterol[3],
            data=df_colesterol, order=4, ci=None, color='red',
            marker='o', label='Elevado',
            scatter_kws={'color':'red', 's':120})
sns.regplot(x = df_colesterol['esporte'], y = df_colesterol[4],
            data=df_colesterol, order=4, ci=None, color='maroon',
            marker='o', label='Muito Elevado',
            scatter_kws={'color':'maroon', 's':120})
plt.ylabel('Probabilidades', fontsize=15)
plt.xlabel('Esporte', fontsize=15)
plt.legend(loc='upper right', fontsize=12)
plt.show()


# In[ ]: Plotagens das probabilidades

#Plotagem das smooth probability lines para a variável 'cigarro'

# 0: Ótimo
# 1: Subótimo
# 2: Limítrofe
# 3: Elevado
# 4: Muito Elevado

plt.figure(figsize=(10,10))
sns.regplot(x = df_colesterol['cigarro'], y = df_colesterol[0],
            data=df_colesterol, order=4, ci=None, color='dodgerblue',
            scatter=False, label='Ótimo')
sns.regplot(x = df_colesterol['cigarro'], y = df_colesterol[1],
            data=df_colesterol, order=4, ci=None, color='darkgreen',
            scatter=False, label='Subótimo')
sns.regplot(x = df_colesterol['cigarro'], y = df_colesterol[2],
            data=df_colesterol, order=4, ci=None, color='darkorange',
            scatter=False, label='Limítrofe')
sns.regplot(x = df_colesterol['cigarro'], y = df_colesterol[3],
            data=df_colesterol, order=4, ci=None, color='red',
            scatter=False, label='Elevado')
sns.regplot(x = df_colesterol['cigarro'], y = df_colesterol[4],
            data=df_colesterol, order=4, ci=None, color='maroon',
            scatter=False, label='Muito Elevado')
plt.ylabel('Probabilidades', fontsize=15)
plt.xlabel('Cigarro', fontsize=15)
plt.legend(loc='upper left', fontsize=12)
plt.show()


# In[ ]: Plotagem tridimensional para cada probabilidade de ocorrência de cada
#categoria da variável dependente

#Função 'go' do pacote 'plotly'

#Categoria 'Ótimo' 

import plotly.io as pio
pio.renderers.default = 'browser'

trace = go.Mesh3d(
    x=df_colesterol['esporte'], 
    y=df_colesterol['cigarro'],
    z=df_colesterol[0],
    opacity=1, intensity=df_colesterol[0], colorscale="Viridis")

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    width=800,
    height=800
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)

plot_figure.update_layout(scene = dict(
                        xaxis_title='esporte',
                        yaxis_title='cigarro',
                        zaxis_title='Ótimo'))

plot_figure.show()


# In[ ]: Plotagem tridimensional para cada probabilidade de ocorrência de cada
#categoria da variável dependente

#Categoria 'Subótimo'

pio.renderers.default = 'browser'

trace = go.Mesh3d(
    x=df_colesterol['esporte'], 
    y=df_colesterol['cigarro'],
    z=df_colesterol[1],
    opacity=1, intensity=df_colesterol[1], colorscale="Viridis")

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    width=800,
    height=800
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)

plot_figure.update_layout(scene = dict(
                        xaxis_title='esporte',
                        yaxis_title='cigarro',
                        zaxis_title='Subótimo'))

plot_figure.show()


# In[ ]: Plotagem tridimensional para cada probabilidade de ocorrência de cada
#categoria da variável dependente

#Categoria 'Limítrofe'

pio.renderers.default = 'browser'

trace = go.Mesh3d(
    x=df_colesterol['esporte'], 
    y=df_colesterol['cigarro'],
    z=df_colesterol[2],
    opacity=1, intensity=df_colesterol[2], colorscale="Viridis")

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    width=800,
    height=800
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)

plot_figure.update_layout(scene = dict(
                        xaxis_title='esporte',
                        yaxis_title='cigarro',
                        zaxis_title='Limítrofe'))

plot_figure.show()


# In[ ]: Plotagem tridimensional para cada probabilidade de ocorrência de cada
#categoria da variável dependente

#Categoria 'Elevado'

pio.renderers.default = 'browser'

trace = go.Mesh3d(
    x=df_colesterol['esporte'], 
    y=df_colesterol['cigarro'],
    z=df_colesterol[3],
    opacity=1, intensity=df_colesterol[3], colorscale="Viridis")

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    width=800,
    height=800
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)

plot_figure.update_layout(scene = dict(
                        xaxis_title='esporte',
                        yaxis_title='cigarro',
                        zaxis_title='Elevado'))

plot_figure.show()


# In[ ]: Plotagem tridimensional para cada probabilidade de ocorrência de cada
#categoria da variável dependente

#Categoria 'Muito Elevado'

pio.renderers.default = 'browser'

trace = go.Mesh3d(
    x=df_colesterol['esporte'], 
    y=df_colesterol['cigarro'],
    z=df_colesterol[4],
    opacity=1, intensity=df_colesterol[4], colorscale="Viridis")

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    width=800,
    height=800
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)

plot_figure.update_layout(scene = dict(
                        xaxis_title='esporte',
                        yaxis_title='cigarro',
                        zaxis_title='Muito Elevado'))

plot_figure.show()


# In[ ]: Visualização das sigmóides tridimensionais em um único gráfico

pio.renderers.default = 'browser'

trace = go.Mesh3d(
    x=df_colesterol['esporte'], 
    y=df_colesterol['cigarro'],
    z=df_colesterol[0],
    opacity=1)

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    width=800,
    height=800
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)

trace_1 = go.Mesh3d(
            x=df_colesterol['esporte'], 
            y=df_colesterol['cigarro'],
            z=df_colesterol[1],
            opacity=1)

plot_figure.add_trace(trace_1)

trace_2 = go.Mesh3d(
            x=df_colesterol['esporte'], 
            y=df_colesterol['cigarro'],
            z=df_colesterol[2],
            opacity=1)

plot_figure.add_trace(trace_2)

trace_3 = go.Mesh3d(
            x=df_colesterol['esporte'], 
            y=df_colesterol['cigarro'],
            z=df_colesterol[3],
            opacity=1)

plot_figure.add_trace(trace_3)

trace_4 = go.Mesh3d(
            x=df_colesterol['esporte'], 
            y=df_colesterol['cigarro'],
            z=df_colesterol[4],
            opacity=1)

plot_figure.add_trace(trace_4)

plot_figure.update_layout(scene = dict(
                        xaxis_title='esporte',
                        yaxis_title='cigarro',
                        zaxis_title='probabilidades'))

plot_figure.show()
