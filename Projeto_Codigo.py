# Projeto de IA de Qualidade do Ar – ODS 13
# Aluno: Nicolas David da Cruz Santos - 176612
#---------------------------------------------------
# Por que esse tema?
# Basicamente, a poluição do ar é um problema global
# que afeta a todos e fazer uma análise crítica sobre a qualidade do ar
# E fazer previsões sobre concentração de poluentes é
#fundamental para a saúde pública e para políticas ambientais, quando feitas.
# -------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import time
import warnings


#Usando o sklearn para regressão linear, polinomial e ridge pra facilitar a vida.
from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.preprocessing   import PolynomialFeatures, StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size']  = 10


#Carregar dados e mostrar informações iniciais

print("=" * 60)
print("Carregando os dados e mostrando informações iniciais")
print("=" * 60)

df = pd.read_csv('AirQualityUCI.csv') # Arquivo que está no diretório

print(f"\nDimensões do dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")
print(f"\nVariáveis disponíveis:\n{df.dtypes}")
print(f"\nEstatísticas descritivas:\n{df.describe().round(2)}")
print(f"\nValores ausentes por coluna:\n{df.isnull().sum()}")


# Fazendo o pré processamento dos dados
print("\n" + "=" * 60)
print("Pré processamento dos dados - vamo lá")
print("=" * 60)

# Aqui tem fazer a remoção de linhas com valores ausentes nas colunas alvo e features
df_clean = df.dropna(subset=['CO', 'NO2', 'Temperatura', 'Umidade', 'Vento'])
print(f"\nAqui estão as linhas após remoção dos valores sem nada: {len(df_clean)} (removidas: {len(df) - len(df_clean)})")

# Remoção de outliers via IQR em CO e NO2

def remover_outliers_iqr(dataframe, coluna, fator=2.5):
    """
    Remove outliers usando o intervalo interquartil (IQR).
    Fator 2.5 foi escolhido pra não ser agressivo demais com os dados.
    """
    Q1 = dataframe[coluna].quantile(0.25)# Aqui é calculado o primerio quartil
    Q3 = dataframe[coluna].quantile(0.75)# Aqui é calculado o terceiro quartil
    IQR = Q3 - Q1 #Conta básica do IQR que a gente vê em estatística
    mask = (dataframe[coluna] >= Q1 - fator * IQR) & (dataframe[coluna] <= Q3 + fator * IQR) # Aqui criamos o filtro dos dados que queremos
    return dataframe[mask] #Retorna-se a os dados filtrados

antes = len(df_clean)
df_clean = remover_outliers_iqr(df_clean, 'CO')
df_clean = remover_outliers_iqr(df_clean, 'NO2')
print(f"Linhas depois de remover o outlier: {len(df_clean)} (número de linhas removidas: {antes - len(df_clean)})")

# Features e o alvo que queremos prever (CO)
FEATURES = ['Temperatura', 'Umidade', 'Vento', 'Hora']
ALVO     = 'CO'

X = df_clean[FEATURES].values
y = df_clean[ALVO].values

# Aqui fazemos a divisão treino/teste (80/20)
# Isso aqui foi falado em aula, mas
#basicamente, a gente separa os dados em dois conjuntos: um para treinar o modelo e outro para testar se ele aprendeu bem.
# Usamos isso quando temos bastante dados e como temos mais de 9000 linhas...

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTreino: {len(X_train)} amostras | Teste: {len(X_test)} amostras") 

#----------------------------------------------------------
#Fazendo a visualização exploratória dos dados. 
# Para vermos se há anomalias, tendências ou padrões que possam influenciar na modelagem.
print("\n" + "=" * 60)
print("Fazendo a visualização exploratória dos dados")
print("=" * 60)

fig = plt.figure(figsize=(16, 12))
fig.suptitle('Visualização Exploratória dos Dados', fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Distribuição de CO
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df_clean['CO'], bins=40, color='steelblue', edgecolor='white')
ax1.set_title('Distribuição de CO')
ax1.set_xlabel('CO (mg/m³)')
ax1.set_ylabel('Frequência')

# Distribuição de NO2
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(df_clean['NO2'], bins=40, color='coral', edgecolor='white')
ax2.set_title('Distribuição de NO₂')
ax2.set_xlabel('NO₂ (µg/m³)')
ax2.set_ylabel('Frequência')

# CO médio por hora
ax3 = fig.add_subplot(gs[0, 2])
co_hora = df_clean.groupby('Hora')['CO'].mean()
ax3.plot(co_hora.index, co_hora.values, marker='o', color='steelblue', linewidth=2)
ax3.set_title('CO Médio por Hora do Dia')
ax3.set_xlabel('Hora')
ax3.set_ylabel('CO médio (mg/m³)')
ax3.set_xticks(range(0, 24, 3))

# Correlação CO x Temperatura para ver a relação entre as duas variáveis.
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(df_clean['Temperatura'], df_clean['CO'], alpha=0.3, s=5, color='steelblue')
ax4.set_title('CO x Temperatura')
ax4.set_xlabel('Temperatura (°C)')
ax4.set_ylabel('CO (mg/m³)')

# Correlação CO x Umidade
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(df_clean['Umidade'], df_clean['CO'], alpha=0.3, s=5, color='seagreen')
ax5.set_title('CO x Umidade')
ax5.set_xlabel('Umidade (%)')
ax5.set_ylabel('CO (mg/m³)')

# Correlação CO x Vento
ax6 = fig.add_subplot(gs[1, 2])
ax6.scatter(df_clean['Vento'], df_clean['CO'], alpha=0.3, s=5, color='darkorange')
ax6.set_title('CO x Vento')
ax6.set_xlabel('Vento (m/s)')
ax6.set_ylabel('CO (mg/m³)')

# Mapa de correlação
#Pelo que vimos em aula, serve para ver a relação entre as variáveis.
#Verificamos se há viés ou multicolinearidade entre as features, o que pode afetar a performance do modelo.

ax7 = fig.add_subplot(gs[2, :])
corr = df_clean[FEATURES + [ALVO]].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=ax7, linewidths=0.5, square=True)
ax7.set_title('Mapa de Correlação entre Variáveis')

plt.savefig('exploracao.png', bbox_inches='tight')
plt.close()
print ('Salvo o gráfico de exploração.png')

#Treinando o modelo

print("\n" + "=" * 60)
print("Treinando os modelos, fazendo a parte legal ")
print("=" * 60)

resultados = {}  # serve pra guardar métricas de todos os modelos

#----------------------------------------------------------


#Regressão Linear Simples (apenas Temperatura como feature)
print("\nRegressão Linear Simples para temperatura")

X_simples_train = X_train[:, 0].reshape(-1, 1)  # só Temperatura
X_simples_test  = X_test[:, 0].reshape(-1, 1)

t0    = time.time()
rl    = LinearRegression()
rl.fit(X_simples_train, y_train)
t_rl  = time.time() - t0

y_pred_rl = rl.predict(X_simples_test)

resultados['Linear Simples'] = {
    'R²'   : r2_score(y_test, y_pred_rl),
    'RMSE' : np.sqrt(mean_squared_error(y_test, y_pred_rl)),
    'MAE'  : mean_absolute_error(y_test, y_pred_rl),
    'Tempo': t_rl,
    'preds': y_pred_rl
}
print(f"  R²={resultados['Linear Simples']['R²']:.4f}  "
      f"RMSE={resultados['Linear Simples']['RMSE']:.4f}  "
      f"Tempo={t_rl*1000:.2f} ms")


#----------------------------------------------------------

#Regressão Linear Múltipla (todas as features)
print("\nRegressão Linear Múltipla para todas as features")

scaler_rlm = StandardScaler()
X_train_sc = scaler_rlm.fit_transform(X_train)
X_test_sc  = scaler_rlm.transform(X_test)

t0      = time.time()
rlm     = LinearRegression()
rlm.fit(X_train_sc, y_train)
t_rlm   = time.time() - t0

y_pred_rlm = rlm.predict(X_test_sc)

resultados['Linear Múltipla'] = {
    'R²'   : r2_score(y_test, y_pred_rlm),
    'RMSE' : np.sqrt(mean_squared_error(y_test, y_pred_rlm)),
    'MAE'  : mean_absolute_error(y_test, y_pred_rlm),
    'Tempo': t_rlm,
    'preds': y_pred_rlm
}
print(f"  R²={resultados['Linear Múltipla']['R²']:.4f}  "
      f"RMSE={resultados['Linear Múltipla']['RMSE']:.4f}  "
      f"Tempo={t_rlm*1000:.2f} ms")
print(f"  Coeficientes: { {f: round(c,4) for f,c in zip(FEATURES, rlm.coef_)} }")

#----------------------------------------------------------
#Regressão Polinomial (grau 2 e 3)
for grau in [2, 3]:
    nome = f'Polinomial (grau {grau})'
    print(f"\n{grau-1}] Regressão Polinomial grau {grau}")

    pipe = Pipeline([
        ('poly',   PolynomialFeatures(degree=grau, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model',  LinearRegression())
    ])

    t0   = time.time()
    pipe.fit(X_train, y_train)
    t_p  = time.time() - t0

    y_pred_p = pipe.predict(X_test)

    resultados[nome] = {
        'R²'   : r2_score(y_test, y_pred_p),
        'RMSE' : np.sqrt(mean_squared_error(y_test, y_pred_p)),
        'MAE'  : mean_absolute_error(y_test, y_pred_p),
        'Tempo': t_p,
        'preds': y_pred_p
    }
    print(f"  R²={resultados[nome]['R²']:.4f}  "
          f"RMSE={resultados[nome]['RMSE']:.4f}  "
          f"Tempo={t_p*1000:.2f} ms")

# ----------------------------------------------------------
#Regressão Ridge (L2) – variação de parâmetro alpha

print("\nRidge")

alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
ridge_r2 = []

for alpha in alphas:
    ridge_tmp = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  Ridge(alpha=alpha))
    ])
    ridge_tmp.fit(X_train, y_train)
    r2_tmp = r2_score(y_test, ridge_tmp.predict(X_test))
    ridge_r2.append(r2_tmp)

# Melhor alpha
melhor_alpha = alphas[np.argmax(ridge_r2)]
print(f"  Melhor alpha: {melhor_alpha} → R²={max(ridge_r2):.4f}")

t0    = time.time()
ridge = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=melhor_alpha))])
ridge.fit(X_train, y_train)
t_ridge = time.time() - t0

y_pred_ridge = ridge.predict(X_test)
resultados['Ridge'] = {
    'R²'   : r2_score(y_test, y_pred_ridge),
    'RMSE' : np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
    'MAE'  : mean_absolute_error(y_test, y_pred_ridge),
    'Tempo': t_ridge,
    'preds': y_pred_ridge
}

# =============================================================================
#Fazendo o cross-validation
print("\n" + "=" * 60)
print("cross-validation")
print("=" * 60)

modelos_cv = {
    'Linear Múltipla': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
    'Ridge'          : Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=melhor_alpha))]),
    'Polinomial g2'  : Pipeline([('poly', PolynomialFeatures(2)), ('scaler', StandardScaler()), ('model', LinearRegression())]),
}#Aqui a gente faz o cross-validation para ver como os modelos se comportam em diferentes subconjuntos dos dados, 
#garantindo que não estamos apenas "sortudos" com a divisão treino/teste.

for nome, modelo in modelos_cv.items():
    scores = cross_val_score(modelo, X, y, cv=5, scoring='r2')
    print(f"  {nome:20s}  R² médio={scores.mean():.4f}  ± {scores.std():.4f}")

# =============================================================================
#Resultados e comparação.

print("\n" + "=" * 60)
print("Resultados e comparação")
print("=" * 60)

nomes   = list(resultados.keys())
r2s     = [resultados[n]['R²']   for n in nomes]
rmses   = [resultados[n]['RMSE'] for n in nomes]
maes    = [resultados[n]['MAE']  for n in nomes]
tempos  = [resultados[n]['Tempo'] * 1000 for n in nomes]  # em ms
cores   = ['#4C9BE8', '#2ECC71', '#E67E22', '#E74C3C', '#9B59B6']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Comparação de Modelos', fontsize=13, fontweight='bold')

# R²
axes[0, 0].bar(nomes, r2s, color=cores, edgecolor='white')
axes[0, 0].set_title('R² (quanto maior, melhor)')
axes[0, 0].set_ylabel('R²')
axes[0, 0].set_ylim(0, 1)
axes[0, 0].tick_params(axis='x', rotation=20)
for i, v in enumerate(r2s):
    axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

# RMSE
axes[0, 1].bar(nomes, rmses, color=cores, edgecolor='white')
axes[0, 1].set_title('RMSE (quanto menor, melhor)')
axes[0, 1].set_ylabel('RMSE (mg/m³)')
axes[0, 1].tick_params(axis='x', rotation=20)
for i, v in enumerate(rmses):
    axes[0, 1].text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=9)

# MAE
axes[1, 0].bar(nomes, maes, color=cores, edgecolor='white')
axes[1, 0].set_title('MAE (quanto menor, melhor)')
axes[1, 0].set_ylabel('MAE (mg/m³)')
axes[1, 0].tick_params(axis='x', rotation=20)
for i, v in enumerate(maes):
    axes[1, 0].text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=9)

# Tempo de execução
axes[1, 1].bar(nomes, tempos, color=cores, edgecolor='white')
axes[1, 1].set_title('Tempo de Treinamento (ms)')
axes[1, 1].set_ylabel('Tempo (ms)')
axes[1, 1].tick_params(axis='x', rotation=20)
for i, v in enumerate(tempos):
    axes[1, 1].text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('comparacao_modelos.png', bbox_inches='tight')
plt.close()
print("salvo a o png de comparação_modelos.png")

# ----------------------------------------------------------
# Gráfico: Real vs Predito para o melhor modelo, somente

melhor_nome  = nomes[np.argmax(r2s)]
melhor_preds = resultados[melhor_nome]['preds']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Real vs Predito – {melhor_nome}', fontsize=12, fontweight='bold')

# Scatter real vs predito
limite = max(y_test.max(), melhor_preds.max()) * 1.05
axes[0].scatter(y_test, melhor_preds, alpha=0.3, s=10, color='steelblue')
axes[0].plot([0, limite], [0, limite], 'r--', linewidth=1.5, label='Ideal')
axes[0].set_xlabel('CO Real (mg/m³)')
axes[0].set_ylabel('CO Predito (mg/m³)')
axes[0].set_title('Predito x Real')
axes[0].legend()

# Erro do modelo
residuos = y_test - melhor_preds
axes[1].scatter(melhor_preds, residuos, alpha=0.3, s=10, color='coral')
axes[1].axhline(0, color='black', linewidth=1.5, linestyle='--')
axes[1].set_xlabel('CO Predito (mg/m³)')
axes[1].set_ylabel('Erro (mg/m³)')
axes[1].set_title('Análise de Erro')

plt.tight_layout()
plt.savefig('real_vs_predito.png', bbox_inches='tight')
plt.close()
print("Figura salva: real_vs_predito.png")

# ----------------------------------------------------------
# Variação do alfa no ridge e seu impacto no R²
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogx(alphas, ridge_r2, marker='o', color='purple', linewidth=2)
ax.set_xlabel('Alpha (escala log)')
ax.set_ylabel('R²')
ax.set_title('Influência do Parâmetro Alpha – Ridge')
ax.axvline(melhor_alpha, color='red', linestyle='--', label=f'Melhor α={melhor_alpha}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ridge_alpha.png', bbox_inches='tight')
plt.close()
print("Figura salva: ridge_alpha.png")

# =============================================================================
#Resumo final

print("\n" + "=" * 60)
print("RESUMO FINAL DOS MODELOS")
print("=" * 60)

resumo = pd.DataFrame({
    'Modelo' : nomes,
    'R²'     : [f"{resultados[n]['R²']:.4f}"   for n in nomes],
    'RMSE'   : [f"{resultados[n]['RMSE']:.4f}"  for n in nomes],
    'MAE'    : [f"{resultados[n]['MAE']:.4f}"   for n in nomes],
    'Tempo'  : [f"{resultados[n]['Tempo']*1000:.2f} ms" for n in nomes],
})
print(resumo.to_string(index=False))
print(f"\n o melhor modelo: {melhor_nome} (R²={max(r2s):.4f})")
print(f" Ou seja, o modelo consegue explicar {max(r2s)*100:.1f}% da variação do CO.")
print(f"  O erro médio absoluto foi de {min(maes):.3f} mg/m³")
print("fim do projeto")

