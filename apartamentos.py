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
#       EXEMPLO 01 - MODELOS MÚLTIPLOS DE REGRESSÃO PELO CRITÉRIO OLS        #
##############################################################################
    
df_apartamentos = pd.read_csv('apartamentos.csv', delimiter=',')
df_apartamentos

#Características das variáveis do dataset
df_apartamentos.info()

#Estatísticas univariadas
df_apartamentos.describe()

#A variável bairro é do tipo categórica policotômica!


# In[ ]: Observando as correlações de Pearson entre as variáveis MÉTRICAS:

#Maneira simples pela função 'corr'
corr = df_apartamentos.corr()
corr

#Maneira mais elaborada pela função 'rcorr' do pacote 'pingouin'
import pingouin as pg

corr2 = pg.rcorr(df_apartamentos, method='pearson',
                 upper='pval', decimals=4,
                 pval_stars={0.01: '***',
                             0.05: '**',
                             0.10: '*'})
corr2


# In[ ]: Mapa de calor da matriz de correlações

plt.figure(figsize=(15,10))
sns.heatmap(df_apartamentos.corr(), annot=True, cmap = plt.cm.viridis,
            annot_kws={'size':22})
plt.show()


# In[ ]: Distribuições das variáveis, scatters, valores das correlações e suas
#respectivas significâncias

def corrfunc(x, y, **kws):
    (r, p) = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f} ".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.4, .9), xycoords=ax.transAxes)

plt.figure(figsize=(15,10))
graph = sns.pairplot(df_apartamentos, diag_kind="kde")
graph.map(corrfunc)
plt.show()


# In[ ]: Tabela de frequências absolutas da variável 'bairro'

df_apartamentos['bairro'].value_counts()


# In[ ]: Procedimento n-1 dummies aplicado à variável 'bairro'
df_apartamentos_dummies = pd.get_dummies(df_apartamentos, columns=['bairro'],
                                         drop_first=True)

df_apartamentos_dummies


# In[ ]: Estimação do modelo de regressão múltipla com n-1 dummies

#Definição da fórmula utilizada no modelo
lista_colunas = list(df_apartamentos_dummies.drop(columns=['preco']).columns)
formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "preco ~ " + formula_dummies_modelo

modelo_apartamentos = sm.OLS.from_formula(formula_dummies_modelo,
                                          df_apartamentos_dummies).fit()

#Parâmetros do modelo
modelo_apartamentos.summary()

#Os parâmetros das variáveis 'area' e 'area_terreno' não se mostraram
#estatisticamente significantes, na presença das demais variáveis,
#ceteris paribus. A multicolinearidade pode ser a razão para esse fato!


# In[ ]: Diagnóstico de multicolinearidade (Variance Inflation Factor e
#Tolerance)

from statsmodels.stats.outliers_influence import variance_inflation_factor

X1 = df_apartamentos_dummies[['area','comodos','area_terreno','bairro_moema',
                              'bairro_vilanova']]
X1 = sm.add_constant(X1)

vif = pd.Series([variance_inflation_factor(X1.values, i)
                  for i in range(X1.shape[1])],index=X1.columns)
vif

tolerance = 1/vif
tolerance

pd.concat([vif, tolerance], axis=1, keys=['VIF', 'Tolerance'])


# In[ ]: Aplicando o procedimento Stepwise

# Instalação e carregamento da função 'stepwise' do pacote
#'statstests.process'
# Autores do pacote: Helder Prado Santos e Luiz Paulo Fávero
# https://stats-tests.github.io/statstests/
# pip install statstests
from statstests.process import stepwise

# Estimação do modelo por meio do procedimento Stepwise
modelo_step_apartamentos = stepwise(modelo_apartamentos, pvalue_limit=0.05)


# In[ ]: Teste de verificação da aderência dos resíduos à normalidade

# Teste de Shapiro-Francia (n >= 30)
# Instalação e carregamento da função 'shapiro_francia' do pacote
#'statstests.tests'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/
# pip install statstests
from statstests.tests import shapiro_francia
shapiro_francia(modelo_step_apartamentos.resid)

# Interpretação
teste_sf = shapiro_francia(modelo_step_apartamentos.resid) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')

#Segundo o teste Shapiro-Francia, não há a aderência à normalidade dos 
#resíduos do 'modelo_step_apartamentos'!


# In[ ]: Plotando os resíduos do 'modelo_step_apartamentos' e acrescentando
#uma curva normal teórica para comparação entre as distribuições
#Kernel density estimation (KDE) - forma não-paramétrica para estimar
#a função densidade de probabilidade de uma variável aleatória

from scipy.stats import norm

plt.figure(figsize=(15,10))
sns.distplot(modelo_step_apartamentos.resid, fit=norm, kde=True, bins=20,
             color='darkblue')
sns.kdeplot(data=modelo_step_apartamentos.resid, multiple="stack", alpha=0.4,
            color='dodgerblue')
plt.xlabel('Resíduos do Modelo Linear', fontsize=16)
plt.ylabel('Densidade', fontsize=16)
plt.show()


# In[ ]: Função para o teste de Breusch-Pagan para a elaboração
# de diagnóstico de heterocedasticidade

# Criação da função 'breusch_pagan_test'

def breusch_pagan_test(modelo):

    df = pd.DataFrame({'yhat':modelo.fittedvalues,
                       'resid':modelo.resid})
   
    df['up'] = (np.square(df.resid))/np.sum(((np.square(df.resid))/df.shape[0]))
   
    modelo_aux = sm.OLS.from_formula('up ~ yhat', df).fit()
   
    anova_table = sm.stats.anova_lm(modelo_aux, typ=2)
   
    anova_table['sum_sq'] = anova_table['sum_sq']/2
    
    chisq = anova_table['sum_sq'].iloc[0]
   
    p_value = stats.chi2.pdf(chisq, 1)*2
    
    print(f"chisq: {chisq}")
    
    print(f"p-value: {p_value}")
    
    return chisq, p_value


# In[ ]: Teste de Breusch-Pagan propriamente dito

breusch_pagan_test(modelo_step_apartamentos)
#Presença de heterocedasticidade -> omissão de variável(is) explicativa(s)
#relevante(s)

#H0 do teste: ausência de heterocedasticidade.
#H1 do teste: heterocedasticidade, ou seja, correlação entre resíduos e
#uma ou mais variáveis explicativas, o que indica omissão de
#variável relevante!

# Interpretação
teste_bp = breusch_pagan_test(modelo_step_apartamentos) #criação do objeto 'teste_bp'
chisq, p = teste_bp #definição dos elementos contidos no objeto 'teste_bp'
alpha = 0.05 #nível de significância
if p > alpha:
    print('Não se rejeita H0 - Ausência de Heterocedasticidade')
else:
	print('Rejeita-se H0 - Existência de Heterocedasticidade')

#Não se verifica a existência de homocedasticidade nos termos de erro.
#Outra forma de se visualizar o fenômeno da heterocedasticidade é pela
#plotagem de Fitted Values x Resíduos. A formação de um cone é comum quando da 
#existência de heterocedasticidade.


# In[ ]: Adicionando os fitted values e os resíduos do 'modelo_step_apartamentos'
#no dataset 'df_apartamentos_dummies'

df_apartamentos_dummies['fitted_values'] = modelo_step_apartamentos.fittedvalues
df_apartamentos_dummies['residuos'] = modelo_step_apartamentos.resid
df_apartamentos_dummies


# In[ ]: Gráfico que relaciona resíduos e fitted values do
#'modelo_step_apartamentos'

plt.figure(figsize=(15,10))
sns.regplot(x='fitted_values', y='residuos', data=df_apartamentos_dummies,
            marker='o', fit_reg=False,
            scatter_kws={'color':'dodgerblue', 'alpha':0.5, 's':150})
plt.title('Gráfico de Dispersão entre Resíduos e Fitted Values', fontsize=20)
plt.xlabel('Fitted Values do Modelo Stepwise', fontsize=17)
plt.ylabel('Resíduos do Modelo Stepwise', fontsize=17)
plt.legend(fontsize=17)
plt.show()


# In[ ]: Uma variável Y aderente à normalidade pode ajudar na aderência à
#normalidade dos erros obtidos a posteriori em determinado modelo.
#Assim, vamos elaborar o seguinte gráfico

plt.figure(figsize=(10,10))
sns.histplot(data=df_apartamentos_dummies['preco'], kde=True, bins=20,
             color = 'dodgerblue')
plt.xlabel('Preço por m²', fontsize=16)
plt.ylabel('Frequência', fontsize=16)
plt.show()


# In[ ]: Transformação de Box-Cox

#Para o cálculo do lambda de Box-Cox
from scipy.stats import boxcox

#x é uma variável que traz os valores transformados (Y*)
#'lmbda' é o lambda de Box-Cox
x, lmbda = boxcox(df_apartamentos_dummies['preco'])

#Inserindo a variável transformada ('bc_preco') no dataset
#para a estimação de um novo modelo
df_apartamentos_dummies['bc_preco'] = x

df_apartamentos_dummies

#Apenas para fins de comparação e comprovação do cálculo de x
df_apartamentos_dummies['bc_preco2'] = ((df_apartamentos_dummies['preco']\
                                         **lmbda)-1)/lmbda

df_apartamentos_dummies

del df_apartamentos_dummies['bc_preco2']


# In[ ]: Observando a distribuição da nova variável Y ('bc_preco')

plt.figure(figsize=(10,10))
sns.histplot(data=df_apartamentos_dummies['bc_preco'], kde=True, bins=20,
             color = 'lightsalmon')
plt.xlabel('Preço por m² após a Transformação de Box-Cox', fontsize=16)
plt.ylabel('Frequência', fontsize=16)
plt.show()


# In[ ]: Estimando um novo modelo OLS com variável dependente
#transformada por Box-Cox

#Definição da fórmula utilizada no modelo
lista_colunas = list(df_apartamentos_dummies.drop(columns=['preco',
                                                           'bc_preco',
                                                           'fitted_values',
                                                           'residuos']).columns)
formula_bc_dummies_modelo = ' + '.join(lista_colunas)
formula_bc_dummies_modelo = "bc_preco ~ " + formula_bc_dummies_modelo

modelo_bc_apartamentos = sm.OLS.from_formula(formula_bc_dummies_modelo,
                                             df_apartamentos_dummies).fit()

#Parâmetros do modelo
modelo_bc_apartamentos.summary()

#Mais uma vez, os parâmetros das variáveis 'area' e 'area_terreno' não se
#mostraram estatisticamente significantes, na presença das demais, ceteris
#paribus.


# In[ ]: Aplicando o procedimento Stepwise no 'modelo_bc_apartamentos'

modelo_step_bc_apartamentos = stepwise(modelo_bc_apartamentos,
                                       pvalue_limit=0.05)


# In[ ]: Comparando os parâmetros do 'modelo_step_apartamentos' com os do
#'modelo_step_bc_apartamentos'
#CUIDADO!!! OS PARÂMETROS NÃO SÃO DIRETAMENTE COMPARÁVEIS!

summary_col([modelo_step_apartamentos, modelo_step_bc_apartamentos])

#Outro modo mais completo também pela função 'summary_col'
summary_col([modelo_step_apartamentos, modelo_step_bc_apartamentos],
            model_names=["MODELO STEPWISE","MODELO STEPWISE COM BOX-COX"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs))
        })


# In[ ]: Verificando a normalidade dos resíduos do 'modelo_step_bc_apartamentos'

# Teste de Shapiro-Francia
shapiro_francia(modelo_step_bc_apartamentos.resid)

# Interpretação
teste_sf = shapiro_francia(modelo_step_bc_apartamentos.resid) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')


# In[ ]: Plotando os novos resíduos do 'modelo_step_bc_apartamentos'

plt.figure(figsize=(15,10))
sns.distplot(modelo_step_bc_apartamentos.resid, fit=norm, kde=True, bins=20,
             color='coral')
sns.kdeplot(data=modelo_step_bc_apartamentos.resid, multiple="stack", alpha=0.4,
            color='lightsalmon')
plt.xlabel('Resíduos do Modelo Linear', fontsize=16)
plt.ylabel('Densidade', fontsize=16)
plt.show()


# In[ ]: Teste de Breusch-Pagan para diagnóstico de heterocedasticidade
#no 'modelo_step_bc_apartamentos'

breusch_pagan_test(modelo_step_bc_apartamentos)

# Interpretação
teste_bp = breusch_pagan_test(modelo_step_bc_apartamentos) #criação do objeto 'teste_bp'
chisq, p = teste_bp #definição dos elementos contidos no objeto 'teste_bp'
alpha = 0.05 #nível de significância
if p > alpha:
    print('Não se rejeita H0 - Ausência de Heterocedasticidade')
else:
	print('Rejeita-se H0 - Existência de Heterocedasticidade')


# In[ ]: Adicionando fitted values e resíduos do 'modelo_step_bc_apartamentos'
#no dataset 'df_apartamentos_dummies'

df_apartamentos_dummies['fitted_values_bc'] = modelo_step_bc_apartamentos.fittedvalues
df_apartamentos_dummies['residuos_bc'] = modelo_step_bc_apartamentos.resid
df_apartamentos_dummies


# In[ ]: Gráfico que relaciona resíduos e fitted values do
#'modelo_step_bc_apartamentos'

plt.figure(figsize=(15,10))
sns.regplot(x='fitted_values_bc', y='residuos_bc', data=df_apartamentos_dummies,
            marker='o', fit_reg=False,
            scatter_kws={'color':'lightsalmon', 'alpha':0.5, 's':150})
plt.title('Gráfico de Dispersão entre Resíduos e Fitted Values', fontsize=20)
plt.xlabel('Fitted Values do Modelo Stepwise com Box-Cox', fontsize=17)
plt.ylabel('Resíduos do Modelo Stepwise com Box-Cox', fontsize=17)
plt.legend(fontsize=17)
plt.show()


# In[ ]: Fazendo predições com o 'modelo_step_bc_apartamentos'
# Qual é o preço esperado por m² de um imóvel localizado em Moema e com
#4 cômodos, mantidas as demais condições constantes?

modelo_step_bc_apartamentos.predict(pd.DataFrame({'const':[1],
                                                  'comodos':[4],
                                                  'bairro_moema':[1],
                                                  'bairro_vilanova':[0]}))


# In[ ]: Não podemos nos esquecer de fazer o cálculo para a obtenção do fitted
#value de Y (variável 'preco')

(4.743458 * lmbda + 1) ** (1 / lmbda)