# Projeto Final UC: Inteligência Artificial
## Predição de qualidade do ar (ODS 13: Ação Climática)
 Predizer a concentração de CO ou NO₂ com base em temperatura, umidade e dados de outros sensores, a fim de analisar como ocorrem as mudanças climáticas. 
 Dessa forma, este projeto se relaciona com a ODS 13: Ação Contra a Mudança Global do Clima.

## Algoritmos utilizados
- Regresão Linear Simples
- Regressão Linear Múltipla
- Regressão Polinomil de grau 2
- Regressão Polinomial de grau 3
- Ridge (L2)
> Obs: Pelas minhas pesquisas, o Ridge é uma abordagem muito boa para mitigar a multicolinearidade e evitar o overfitting em modelos de Machine Learning.
> Assim, esse método impede que os coeficientes assumam valores extremos quando há variáveis bastante correlacionadas entre si.

## Dados Utlizados 
Usei o Dataset Air Quality UCI que tem mais de 9000 registros e está disponível de forma gratuita e de fácil acesso. 
Basicamente, segundo a descrição do dataset:
Este contém 9358 instâncias de respostas médias horárias de uma matriz de 5 sensores químicos de óxido metálico integrados em um Dispositivo Multissensor Químico de Qualidade do Ar. 
O dispositivo foi posicionado em campo em uma área significativamente poluída, ao nível da estrada, dentro de uma cidade italiana. 
Os dados foram registrados de março de 2004 a fevereiro de 2005 (um ano), 
representando os registros de acesso livre mais longos disponíveis de respostas de dispositivos de sensores químicos de qualidade do ar implantados em campo. 
As concentrações médias horárias de referência (Ground Truth) para CO, Hidrocarbonetos Não Metânicos, Benzeno, Óxidos de Nitrogênio Totais (NOx) e Dióxido de Nitrogênio (NO2) foram fornecidas por um analisador certificado de referência localizado no mesmo ponto. 
Evidências de sensibilidades cruzadas, bem como de desvios de conceito (concept drifts) e de sensor estão presentes, conforme descrito em De Vito et al., Sens. And Act. B, Vol. 129,2,2008, 
afetando eventualmente as capacidades de estimativa de concentração dos sensores. Valores ausentes são identificados com o valor -200.

## Pré-Processamento
O pré-processamento dos dados envolveu a remoção de registros nulos (que foi aplicado nos dados como o valor -200) e a eliminação de outliers via IQR. 
Em seguida, foi aplicada uma divisão de treino e teste utilizando a proporção 80/20.

## Resultados
Os resultados indicaram que o modelo de regressão polinomial de grau 3 apresentou o melhor resultado.
O modelo alcançou $$R^2 = 0{,}8112$$. Os dados comparando os algoritmos estão logo abaixo:

| Modelo | R² | RMSE | MAE | Tempo |
| :--- | :---: | :---: | :---: | :---: |
| Linear Simples | 0.2323 | 1.1355 | 0.9621 | 2.08 ms |
| Linear Múltipla | 0.3498 | 1.0450 | 0.8727 | 1.76 ms |
| Polinomial (grau 2) | 0.7345 | 0.6678 | 0.5353 | 7.44 ms |
| Polinomial (grau 3) | 0.8112 | 0.5632 | 0.4558 | 19.29 ms |
| Ridge | 0.3501 | 1.0448 | 0.8731 | 5.03 ms |

Acredito que esse resultado tenha ocorrido porque a regressão polinomial de grau 3 coresponde às variáveis meteorológicas de forma curvilínea, uma vez que, tratando da variação do CO ao longo do dia, ue apresenta picos e quedas nesse período. Assim, basicamente, a regressão polinomial de grau 3 é capaz de capturar essa complexidade sem decorar dados, justamente por conseguir ajustar uma curva com dois pontos de inflexão.
