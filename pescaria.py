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
#               EXEMPLO 04 - MODELOS PARA DADOS DE CONTAGEM                  #
##############################################################################

df_pescaria = pd.read_csv('pescaria.csv', delimiter=',')
df_pescaria

#Características das variáveis do dataset
df_pescaria.info()

#Estatísticas univariadas
df_pescaria.describe()


# In[ ]: Tabela de frequências da variável dependente 'peixes'
#Função 'values_counts' do pacote 'pandas' sem e com normalização
#para gerar as contagens e os percentuais, respectivamente
contagem = df_pescaria['peixes'].value_counts(dropna=False)
percent = df_pescaria['peixes'].value_counts(dropna=False, normalize=True)
pd.concat([contagem, percent], axis=1, keys=['contagem', '%'], sort=True)


# In[ ]: Histograma da variável dependente 'peixes'

plt.figure(figsize=(15,10))
sns.histplot(data=df_pescaria, x='peixes', bins=10, color='aquamarine')
plt.xlabel('Peixes', fontsize=20)
plt.ylabel('Contagem', fontsize=20)
plt.show()


# In[ ]: Diagnóstico preliminar para observação de eventual igualdade entre a
#média e a variância da variável dependente 'peixes'

pd.DataFrame({'Média':[df_pescaria.peixes.mean()],
              'Variância':[df_pescaria.peixes.var()]})


# In[ ]: Estimação do modelo Poisson

#O argumento 'family=sm.families.Poisson()' da função 'smf.glm' define a
#estimação de um modelo Poisson
modelo_poisson = smf.glm(formula='peixes ~ pessoas + criancas + guia',
                         data=df_pescaria,
                         family=sm.families.Poisson()).fit()

#Parâmetros do modelo
modelo_poisson.summary()


# In[ ]: Todas as variáveis preditoras se mostraram estatisticamente
#diferentes de zero, considerando-se um nível de significância de 5%,
#ceteris paribus. Porém, já se pode afirmar que a estimação Poisson é a mais
#adequada?

# Teste de Superdispersão de Cameron e Trivedi (1990)
# Função 'overdisp'
# Instalação e carregamento da função 'overdisp' do pacote 'statstests.tests'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/
# pip install statstests
from statstests.tests import overdisp

#Elaboração direta do teste de superdispersão
overdisp(modelo_poisson, df_pescaria)


# In[ ]: Estimação do modelo binomial negativo do tipo NB2

#Construção de função para a definição do 'alpha' ('fi') ótimo que gera a
#maximização do valor de Log-Likelihood
n_samples = 10000
alphas = np.linspace(0, 10, n_samples)
llf = np.full(n_samples, fill_value=np.nan)
for i, alpha in enumerate(alphas):
    try:
        model = smf.glm(formula = 'peixes ~ pessoas + criancas + guia',
                        data=df_pescaria,
                        family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    except:
        continue
    llf[i] = model.llf
alpha_ótimo = alphas[np.nanargmax(llf)]
alpha_ótimo

#Plotagem dos resultados
plt.plot(alphas, llf, label='Log-Likelihood', color='cyan')
plt.axvline(x=alpha_ótimo, color='navy',
            label=f'alpha: {alpha_ótimo:0.5f}')
plt.legend()


# In[ ]: Estimação do modelo binomial negativo com o parâmetro 'alpha_ótimo'

modelo_bneg = smf.glm(formula='peixes ~ pessoas + criancas + guia',
                      data=df_pescaria,
                      family=sm.families.NegativeBinomial(alpha=alpha_ótimo)).fit()

#Parâmetros do modelo
modelo_bneg.summary()


# In[ ]: Comparando os modelos Poisson e binomial negativo

summary_col([modelo_poisson, modelo_bneg], 
            model_names=["Poisson","BNeg"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-lik':lambda x: "{:.2f}".format(x.llf),
                'Pseudo-R2':lambda x: "{:.4f}".format(x.pseudo_rsquared()),
        })


# In[ ]: likelihood ratio test para comparação de LL's entre modelos

#Definição da função 'lrtest'
def lrtest(modelos):
    modelo_1 = modelos[0]
    llk_1 = modelo_1.llnull
    llk_2 = modelo_1.llf
    
    if len(modelos)>1:
        llk_1 = modelo_1.llf
        llk_2 = modelos[1].llf
    LR_statistic = -2*(llk_1-llk_2)
    p_val = stats.chi2.sf(LR_statistic, 1)
    return round(LR_statistic,2), round(p_val,2)

lrtest([modelo_poisson, modelo_bneg])


# In[ ]: Gráfico para a comparação dos LL dos modelos Poisson e
#binomial negativo

#Definição do dataframe com os modelos e respectivos LL
df_llf = pd.DataFrame({'modelo':['Poisson','BNeg'],
                      'loglik':[modelo_poisson.llf, modelo_bneg.llf]})
df_llf

#Plotagem propriamente dita
fig, ax = plt.subplots(figsize=(15,10))

c = ['navy', 'dodgerblue']

ax1 = ax.barh(df_llf.modelo, df_llf.loglik, color = c)
ax.bar_label(ax1, label_type='center', color='white', fontsize=24)
ax.set_ylabel("Estimação", fontsize=20)
ax.set_xlabel("Log-Likehood", fontsize=20)


# In[ ]: Adicionando os fitted values dos modelos estimados até o momento,
#para fins de comparação

df_pescaria['fitted_poisson'] = modelo_poisson.fittedvalues
df_pescaria['fitted_bneg'] = modelo_bneg.fittedvalues

df_pescaria


# In[ ]: Fitted values dos modelos Poisson e binomial negativo, considerando,
#para fins didáticos, apenas a variável preditora 'pessoas'

plt.figure(figsize=(12,10))
sns.regplot(x= df_pescaria['pessoas'], y = df_pescaria['fitted_poisson'],
            order=3, color='navy',
            marker='o', label='Poisson',
            scatter_kws={'color':'navy', 's':80})
sns.regplot(x = df_pescaria['pessoas'], y = df_pescaria['fitted_bneg'],
            order=3, color='dodgerblue',
            marker='o', label='Binomial Negativo',
            scatter_kws={'color':'dodgerblue', 's':80})
plt.ylabel('Quantidade de Peixes', fontsize=15)
plt.xlabel('Pessoas', fontsize=15)
plt.legend(loc='upper left', fontsize=14)
plt.show()


# In[ ]: Fitted values dos modelos Poisson e binomial negativo, considerando,
#para fins didáticos, apenas a variável preditora 'criancas'

plt.figure(figsize=(12,10))
sns.regplot(x = df_pescaria['criancas'], y = df_pescaria['fitted_poisson'],
            order=3, color='navy',
            marker='o', label='Poisson',
            scatter_kws={'color':'navy', 's':80})
sns.regplot(x = df_pescaria['criancas'], y = df_pescaria['fitted_bneg'],
            order=3, color='dodgerblue',
            marker='o', label='Binomial Negativo',
            scatter_kws={'color':'dodgerblue', 's':80})
plt.ylabel('Quantidade de Peixes', fontsize=15)
plt.xlabel('Crianças', fontsize=15)
plt.legend(loc='upper right', fontsize=14)
plt.show()


# In[ ]: Fitted values dos modelos Poisson e binomial negativo, considerando,
#para fins didáticos, apenas a variável preditora 'guia' (dummizada)

df_pescaria.loc[df_pescaria['guia']=='nao', 'guia_dummy'] = 0
df_pescaria.loc[df_pescaria['guia']=='sim', 'guia_dummy'] = 1

plt.figure(figsize=(12,10))
# sns.regplot(x = df_pescaria['guia_dummy'], y = df_pescaria['fitted_poisson'],
#             order=3, color='navy',
#             marker='o', label='Poisson',
#             scatter_kws={'color':'navy', 's':80})
sns.regplot(x = df_pescaria['guia_dummy'], y = df_pescaria['fitted_bneg'],
            order=3, color='dodgerblue',
            marker='o', label='Binomial Negativo',
            scatter_kws={'color':'dodgerblue', 's':80})
plt.ylabel('Quantidade de Peixes', fontsize=15)
plt.xlabel('Guia (dummy)', fontsize=15)
plt.legend(loc='upper left', fontsize=14)
plt.show()


# In[ ]: Estimação do modelo ZIP

#Definição da variável dependente
y = df_pescaria.peixes

#Definição das variáveis preditoras que entrarão no componente de contagem
x1 = df_pescaria[['pessoas','criancas','guia']]
X1 = sm.add_constant(x1)

#Definição das variáveis preditoras que entrarão no componente logit (inflate)
x2 = df_pescaria[['criancas']] #inserimos 'criancas' apenas para fins didáticos
X2 = sm.add_constant(x2)

#Se estimarmos o modelo sem dummizar as variáveis categórias, o modelo retorna
#um erro
X1 = pd.get_dummies(X1, columns=['guia'], drop_first=True)

#Estimação do modelo ZIP pela função 'ZeroInflatedPoisson' do pacote
#'Statsmodels'

#Estimação do modelo ZIP
#O argumento 'exog_infl' corresponde às variáveis que entram no componente
#logit (inflate)
modelo_zip = sm.ZeroInflatedPoisson(y, X1, exog_infl=X2,
                                    inflation='logit').fit()

#Parâmetros do modelo
modelo_zip.summary()


# In[ ]: Estimação do modelo ZINB

#Estimação do modelo ZINB pela função 'ZeroInflatedNegativeBinomialP' do
#pacote 'statsmodels.discrete.count_model'

from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

#Estimação do modelo ZINB
#O argumento 'exog_infl' corresponde às variáveis que entram no componente
#logit (inflate)
modelo_zinb = ZeroInflatedNegativeBinomialP(y, X1, exog_infl=X2,
                                            inflation='logit').fit()

#Parâmetros do modelo
modelo_zinb.summary()


# In[ ]: Comparando os modelos Poisson, binomial negativo, ZIP e ZINB

summary_col([modelo_poisson, modelo_bneg, modelo_zip, modelo_zinb], 
            model_names=["Poisson","BNeg","ZIP","ZINB"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-lik':lambda x: "{:.2f}".format(x.llf),
                'Pseudo-R2':lambda x: "{:.4f}".format(x.pseudo_rsquared()),
        })


# In[ ]: Gráfico para a comparação dos LL dos modelos Poisson, BNeg, ZIP e
#ZINB

#Definição do dataframe com os modelos e respectivos LL
df_llf = pd.DataFrame({'modelo':['Poisson','BNeg','ZIP','ZINB'],
                      'loglik':[modelo_poisson.llf,
                                modelo_bneg.llf,
                                modelo_zip.llf,
                                modelo_zinb.llf]})
df_llf

#Plotagem propriamente dita
fig, ax = plt.subplots(figsize=(15,10))

c = ["navy", "dodgerblue", "red", "crimson"]

ax1 = ax.barh(df_llf.modelo, df_llf.loglik, color = c)
ax.bar_label(ax1, label_type='center', color='white', fontsize=24)
ax.set_ylabel("Estimação", fontsize=20)
ax.set_xlabel("Log-Likehood", fontsize=20)