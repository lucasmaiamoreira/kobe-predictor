# import mlflow
# import mlflow.sklearn
# from sklearn.metrics import log_loss, f1_score
# from pycaret.classification import setup, create_model, predict_model
# import pandas as pd
# from sklearn.model_selection import train_test_split

# # Iniciar uma nova run do MLflow com o nome "Treinamento"
# mlflow.start_run(run_name="Treinamento")

# # Carregar os dados de treinamento
# data_train = pd.read_parquet("data/processed/base_train.parquet")

# # Remover linhas com valores faltantes na variável alvo
# data_train = data_train.dropna(subset=['shot_made_flag'])

# # Configurar ambiente PyCaret
# clf = setup(data=data_train, target='shot_made_flag')

# # Treinar modelo de regressão logística
# log_reg = create_model('lr')

# # Avaliar o modelo de regressão logística usando a base de teste
# data_test = pd.read_parquet("data/processed/base_test.parquet")
# y_test = data_test['shot_made_flag']
# y_pred_lr = predict_model(log_reg, data=data_test)['shot_made_flag']
# log_loss_lr = log_loss(y_test, y_pred_lr)

# # Registrar a função custo "log loss" no MLflow para regressão logística
# mlflow.log_metric("log_loss_logistic_regression", log_loss_lr)

# # Treinar modelo de árvore de decisão
# dt = create_model('dt')

# # Avaliar o modelo de árvore de decisão usando a base de teste
# y_pred_dt = predict_model(dt, data=data_test)['shot_made_flag']
# log_loss_dt = log_loss(y_test, y_pred_dt)
# f1_score_dt = f1_score(y_test, predict_model(dt, data=data_test)['shot_made_flag'], average='binary')

# # Registrar a função custo "log loss" e F1_score no MLflow para árvore de decisão
# mlflow.log_metric("log_loss_decision_tree", log_loss_dt)
# mlflow.log_metric("f1_score_decision_tree", f1_score_dt)

# # Selecionar o modelo com melhor desempenho (menor log loss)
# if log_loss_lr < log_loss_dt:
#     final_model = log_reg
#     model_type = "Regressão Logística"
# else:
#     final_model = dt
#     model_type = "Árvore de Decisão"

# # Finalizar a run do MLflow
# mlflow.end_run()

# # Salvar o modelo selecionado
# mlflow.sklearn.save_model(final_model, "final_model")

import mlflow
import mlflow.sklearn
from sklearn.metrics import log_loss, f1_score
from pycaret.classification import setup, create_model, predict_model
import pandas as pd
from sklearn.model_selection import train_test_split

# Iniciar uma nova run do MLflow com o nome "Treinamento"
mlflow.start_run(run_name="Treinamento")

# Carregar os dados de treinamento
data_train = pd.read_parquet("data/processed/base_train.parquet")

# Remover linhas com valores faltantes na variável alvo
data_train = data_train.dropna(subset=['shot_made_flag'])

# Configurar ambiente PyCaret
clf = setup(data=data_train, target='shot_made_flag')

# Treinar modelo de regressão logística
log_reg = create_model('lr')

# Avaliar o modelo de regressão logística usando a base de teste
data_test = pd.read_parquet("data/processed/base_test.parquet")
y_test = data_test['shot_made_flag']
y_pred_lr = predict_model(log_reg, data=data_test)['shot_made_flag']
log_loss_lr = log_loss(y_test, y_pred_lr)

# Registrar a função custo "log loss" no MLflow para regressão logística
mlflow.log_metric("log_loss_logistic_regression", log_loss_lr)

# Treinar modelo de árvore de decisão
dt = create_model('dt')

# Avaliar o modelo de árvore de decisão usando a base de teste
y_pred_dt = predict_model(dt, data=data_test)['shot_made_flag']
log_loss_dt = log_loss(y_test, y_pred_dt)
f1_score_dt = f1_score(y_test, predict_model(dt, data=data_test)['shot_made_flag'], average='binary')

# Registrar a função custo "log loss" e F1_score no MLflow para árvore de decisão
mlflow.log_metric("log_loss_decision_tree", log_loss_dt)
mlflow.log_metric("f1_score_decision_tree", f1_score_dt)

# Selecionar o modelo com melhor desempenho (menor log loss)
if log_loss_lr < log_loss_dt:
    final_model = log_reg
    model_type = "Regressão Logística"
else:
    final_model = dt
    model_type = "Árvore de Decisão"

# Salvar o modelo selecionado usando mlflow.sklearn.save_model
mlflow.sklearn.save_model(sk_model=final_model, path="final_model")

# Finalizar a run do MLflow
mlflow.end_run()
