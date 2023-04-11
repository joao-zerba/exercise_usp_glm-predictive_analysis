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
#                    EXEMPLO 02 - MODELO LOGÍSTICO BINÁRIO                   #
##############################################################################

df_triatlo = pd.read_csv('triatlon.csv', delimiter=',')
df_triatlo

#Características das variáveis do dataset
df_triatlo.info()

#Estatísticas univariadas
df_triatlo.describe()


# In[ ]: Tabela de frequências absolutas da variável 'prova_finalizada'

df_triatlo['prova_finalizada'].value_counts()


# In[ ]: Note que a variável Y 'prova_finalizada' está definida como objeto
#(PROBLEMA!!!)

#Transformando a variável Y para 0 e 1 e para o tipo 'int' (poderia também
#ser do tipo 'float'), a fim de que seja possível estimar o modelo por meio
#da função 'sm.Logit.from_formula' ou da função 'smf.glm'
df_triatlo.loc[df_triatlo['prova_finalizada']=='sim', 'prova_finalizada'] = 1
df_triatlo.loc[df_triatlo['prova_finalizada']=='não', 'prova_finalizada'] = 0

df_triatlo['prova_finalizada'] = df_triatlo['prova_finalizada'].astype('int64')


# In[ ]: Estimação de um modelo logístico binário

modelo_triatlo = smf.glm(formula='prova_finalizada ~ carboidratos',
                         data=df_triatlo,
                         family=sm.families.Binomial()).fit()

#Parâmetros do modelo
modelo_triatlo.summary()


# In[ ]: Outputs do 'modelo_triatlo' pela função 'summary_col'

summary_col([modelo_triatlo],
            model_names=["MODELO LOGIT"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-lik':lambda x: "{:.2f}".format(x.llf)
        })


# In[ ]: Fazendo predições para o 'modelo_atrasos'.
#Exemplo: qual a probabilidade de determinado atleta completar a prova de
#triatlo, tendo consumido 3 gramas de carboidratos por kilo de peso corporal
#no dia anterior à competição?

modelo_triatlo.predict(pd.DataFrame({'carboidratos':[3]}))


# In[ ]: Construção de uma matriz de confusão

# Adicionando os valores previstos de probabilidade na base de dados
df_triatlo['phat'] = modelo_triatlo.predict()

#Visualizando a base de dados com a variável 'phat'
df_triatlo


# In[ ]: Construção de função para a definição da matriz de confusão

from sklearn.metrics import confusion_matrix, accuracy_score,\
    ConfusionMatrixDisplay, recall_score

def matriz_confusao(predicts, observado, cutoff):
    
    values = predicts.values
    
    predicao_binaria = []
        
    for item in values:
        if item < cutoff:
            predicao_binaria.append(0)
        else:
            predicao_binaria.append(1)
           
    cm = confusion_matrix(predicao_binaria, observado)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.xlabel('True')
    plt.ylabel('Classified')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()
        
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)

    #Visualizando os principais indicadores desta matriz de confusão
    indicadores = pd.DataFrame({'Sensitividade':[sensitividade],
                                'Especificidade':[especificidade],
                                'Acurácia':[acuracia]})
    return indicadores


# In[ ]: Matrizes de confusão propriamente ditas

#Matriz de confusão para cutoff = 0.5
matriz_confusao(observado=df_triatlo['prova_finalizada'],
                predicts=df_triatlo['phat'], 
                cutoff=0.5)

#Matriz de confusão para cutoff = 0.3
matriz_confusao(observado=df_triatlo['prova_finalizada'],
                predicts=df_triatlo['phat'], 
                cutoff=0.3)

#Matriz de confusão para cutoff = 0.7
matriz_confusao(observado=df_triatlo['prova_finalizada'],
                predicts=df_triatlo['phat'], 
                cutoff=0.7)


# In[ ]: Igualando critérios de especificidade e de sensitividade

#Tentaremos estabelecer um critério que iguale a probabilidade de
#acerto daqueles que chegarão atrasados (sensitividade) e a probabilidade de
#acerto daqueles que não chegarão atrasados (especificidade).

#ATENÇÃO: o que será feito a seguir possui fins didáticos, apenas. DE NENHUMA
#FORMA o procedimento garante a maximização da acurácia do modelo!

#Criação da função 'espec_sens' para a construção de um dataset com diferentes
#valores de cutoff, sensitividade e especificidade:

def espec_sens(observado,predicts):
    
    # adicionar objeto com os valores dos predicts
    values = predicts.values
    
    # range dos cutoffs a serem analisados em steps de 0.01
    cutoffs = np.arange(0,1.01,0.01)
    
    # Listas que receberão os resultados de especificidade e sensitividade
    lista_sensitividade = []
    lista_especificidade = []
    
    for cutoff in cutoffs:
        
        predicao_binaria = []
        
        # Definindo resultado binário de acordo com o predict
        for item in values:
            if item >= cutoff:
                predicao_binaria.append(1)
            else:
                predicao_binaria.append(0)
                
        # Cálculo da sensitividade e especificidade no cutoff
        sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
        especificidadee = recall_score(observado, predicao_binaria, pos_label=0)
        
        # Adicionar valores nas listas
        lista_sensitividade.append(sensitividade)
        lista_especificidade.append(especificidadee)
        
    # Criar dataframe com os resultados nos seus respectivos cutoffs
    resultado = pd.DataFrame({'cutoffs':cutoffs,'sensitividade':lista_sensitividade,'especificidade':lista_especificidade})
    return resultado


# In[ ]: Até o momento, foram extraídos 3 vetores: 'sensitividade',
#'especificidade' e 'cutoffs'. Assim, criamos um dataframe que contém
#os vetores mencionados

dados_plotagem = espec_sens(observado = df_triatlo['prova_finalizada'],
                            predicts = df_triatlo['phat'])
dados_plotagem


# In[ ]: Visualizando o novo dataframe 'dados_plotagem' e plotando os dados
#em um gráfico que mostra a variação da especificidade e da sensitividade
#em função do cutoff

plt.figure(figsize=(10,10))
plt.plot(dados_plotagem.cutoffs,dados_plotagem.sensitividade, '-o',
         color='maroon')
plt.plot(dados_plotagem.cutoffs,dados_plotagem.especificidade, '-o',
         color='teal')
plt.legend(['Sensitividade', 'Especificidade'], fontsize=17)
plt.xlabel('Cuttoff', fontsize=14)
plt.ylabel('Sensitividade / Especificidade', fontsize=14)
plt.show()


# In[ ]: Construção da curva ROC

from sklearn.metrics import roc_curve, auc

#Função 'roc_curve' do pacote 'metrics' do sklearn

fpr, tpr, thresholds =roc_curve(df_triatlo['prova_finalizada'],
                                df_triatlo['phat'])
roc_auc = auc(fpr, tpr)

#Cálculo do coeficiente de GINI
gini = (roc_auc - 0.5)/(0.5)

#Plotando a curva ROC
plt.figure(figsize=(10,10))
plt.plot(fpr,tpr, '-o', color='navy', markersize=8)
plt.plot(fpr,fpr, color='gray')
plt.title('Área abaixo da curva: %g' % round(roc_auc,4) +
          ' | Coeficiente de GINI: %g' % round(gini,4), fontsize=17)
plt.xlabel('1 - Especificidade', fontsize=15)
plt.ylabel('Sensitividade', fontsize=15)
plt.show()


# In[ ]: Construção da sigmoide
#Probabilidade de evento em função da variável 'carboidratos'

plt.figure(figsize=(15,10))
sns.regplot(x = df_triatlo.carboidratos, y = df_triatlo.prova_finalizada,
            data=df_triatlo, logistic=True, ci=None, color='navy',
            marker='o', scatter_kws={'color':'navy', 'alpha':0.5, 's':170})
plt.axhline(y = 0.5, color = 'grey', linestyle = ':')
plt.xlabel('Gramas de Carboidratos por Quilo de Peso Corporal', fontsize=17)
plt.ylabel('Probabilidade de Terminar a Prova', fontsize=17)
plt.show


# In[ ]: Modelo de Regressão PROBIT

from statsmodels.discrete.discrete_model import Probit

Y = df_triatlo['prova_finalizada']
X = df_triatlo['carboidratos']
X = sm.add_constant(X)
modelo = Probit(Y, X)
modelo_probit = modelo.fit()
modelo_probit.summary()


# In[ ]: Outputs dos modelos LOGIT e PROBIT pela função 'summary_col'

summary_col([modelo_triatlo, modelo_probit],
            model_names=["MODELO LOGIT", "MODELO PROBIT"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-lik':lambda x: "{:.2f}".format(x.llf)
        })


# In[ ]: Construção de curvas ROC com os outputs dos modelos LOGIT e PROBIT

# Adicionando os valores previstos de probabilidade obtidos pelo 'modelo_probit'
#na base de dados
df_triatlo['phat2'] = modelo_probit.predict()

#Visualizando a base de dados com a variável 'phat2' referente ao 'modelo_probit'
df_triatlo

#Construção da curva ROC para o 'modelo_probit'
fpr2, tpr2, thresholds2 =roc_curve(df_triatlo['prova_finalizada'],
                                   df_triatlo['phat2'])
roc_auc2 = auc(fpr2, tpr2)

#AUROCs para os modelos LOGIT e PROBIT
pd.DataFrame({'AUROC LOGIT':[round(roc_auc,4)],
              'AUROC PROBIT':[round(roc_auc2,4)]})


# In[ ]: likelihood ratio test para comparação de LL's entre os modelos
#'modelo_triatlo' e 'modelo_probit'

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

lrtest([modelo_triatlo, modelo_probit])


# In[ ]:Plotando as curvas ROC para os modelos LOGIT e PROBIT

plt.figure(figsize=(10,10))
plt.plot(fpr,tpr, '-o', color='navy', markersize=12) #modelo LOGIT
plt.plot(fpr2,tpr2, '-o', color='violet', markersize=6) #modelo PROBIT
plt.plot(fpr,fpr, color='gray')
plt.title('Curvas ROC', fontsize=20)
plt.xlabel('1 - Especificidade', fontsize=15)
plt.ylabel('Sensitividade', fontsize=15)
plt.legend(['Modelo LOGIT: AUROC = %g' % round(roc_auc,4),
            'Modelo PROBIT: AUROC = %g' % round(roc_auc2,4)],
           fontsize=15, loc='lower right')
plt.show()


# In[ ]: Construção das sigmoides dos modelos LOGIT e PROBIT

plt.figure(figsize=(15,10))
sns.regplot(x = df_triatlo.carboidratos, y = df_triatlo.phat,
            data=df_triatlo, logistic=True, ci=None, color='navy',
            fit_reg=False, marker='o',
            scatter_kws={'color':'navy', 'alpha':0.5, 's':170})
sns.regplot(x = df_triatlo.carboidratos, y = df_triatlo.phat2,
            data=df_triatlo, logistic=True, ci=None, color='violet',
            fit_reg=False, marker='o',
            scatter_kws={'color':'violet', 'alpha':0.5, 's':150})
plt.axhline(y = 0.5, color = 'grey', linestyle = ':')
plt.xlabel('Gramas de Carboidratos por Quilo de Peso Corporal', fontsize=17)
plt.ylabel('Probabilidade de Terminar a Prova', fontsize=17)
plt.legend(['Modelo LOGIT', 'Modelo PROBIT'], fontsize=15)
plt.show


# In[ ]: Para fins didáticos, vamos acrescentar uma variável à modelagem LOGIT.
#Essa variável refere-se à quantidade de provas de triatlo já realizadas pelos
#atletas, e encontra-se no dataset 'qtde_provas.csv'

df_provas = pd.read_csv('qtde_provas.csv', delimiter=',')
df_provas

#Características das variáveis do dataset
df_provas.info()

#Estatísticas univariadas
df_provas.describe()


# In[ ]: Estimação de um novo modelo logístico binário, agora múltiplo

#Inserção da variável 'provas' presente no dataframe 'df_provas' no dataframe
#'df_triatlo'
df_triatlo['provas'] = df_provas['provas']
df_triatlo

#Estimação do novo modelo logístico binário
modelo_triatlo_novo = smf.glm(formula='prova_finalizada ~ carboidratos +\
                              provas',
                              data=df_triatlo,
                              family=sm.families.Binomial()).fit()
#Parâmetros do modelo
modelo_triatlo_novo.summary()


# In[ ]: Adicionando os valores previstos de probabilidade obtidos pelo
#'modelo_triatlo_novo' ao dataframe 'df_triatlo'

df_triatlo['phat3'] = modelo_triatlo_novo.predict()
df_triatlo

#Construção da curva ROC para o 'modelo_triatlo_novo'
fpr3, tpr3, thresholds3 =roc_curve(df_triatlo['prova_finalizada'],
                                   df_triatlo['phat3'])
roc_auc3 = auc(fpr3, tpr3)

#AUROCs para os modelos 'modelo_triatlo' e 'modelo_triatlo_novo'
pd.DataFrame({'AUROC LOGIT':[round(roc_auc,4)],
              'AUROC LOGIT NOVO':[round(roc_auc3,4)]})


# In[ ]: likelihood ratio test para comparação de LL's entre os modelos

lrtest([modelo_triatlo, modelo_triatlo_novo])


# In[ ]:Plotando as curvas ROC para os modelos modelos 'modelo_triatlo' e
#'modelo_triatlo_novo'

plt.figure(figsize=(10,10))
plt.plot(fpr,tpr, '-o', color='navy', markersize=8) #modelo LOGIT inicial
plt.plot(fpr3,tpr3, '-o', color='lightsalmon', markersize=8) #modelo LOGIT novo
plt.plot(fpr,fpr, color='gray')
plt.title('Curvas ROC', fontsize=20)
plt.xlabel('1 - Especificidade', fontsize=15)
plt.ylabel('Sensitividade', fontsize=15)
plt.legend(['Modelo LOGIT: AUROC = %g' % round(roc_auc,4),
            'Modelo LOGIT NOVO: AUROC = %g' % round(roc_auc3,4)],
           fontsize=16, loc='lower right')
plt.show()
