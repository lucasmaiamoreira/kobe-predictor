import mlflow
import pandas as pd
from sklearn.metrics import log_loss, f1_score
from sklearn.pipeline import Pipeline

# Iniciar uma nova run do MLflow com o nome "PipelineAplicacao"
mlflow.start_run(run_name="PipelineAplicacao")

# Carregar o modelo treinado
loaded_model = mlflow.sklearn.load_model("final_model")

# Carregar a base de produção
data_prod = pd.read_parquet("data/dataset_kobe_prod.parquet")

# Remover linhas com valores NaN na variável alvo 'shot_made_flag'
data_prod_cleaned = data_prod.dropna(subset=['shot_made_flag'])

# Selecionar apenas as colunas relevantes para fazer previsões
relevant_columns = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance']
data_prod_relevant = data_prod_cleaned[relevant_columns]

# Aplicar o modelo na base de produção
predictions = loaded_model.predict(data_prod_relevant)

# Calcular as probabilidades previstas (necessárias para o log loss)
probabilities = loaded_model.predict_proba(data_prod_relevant)[:, 1]

# Calcular métricas de desempenho do modelo na base de produção
y_true = data_prod_cleaned['shot_made_flag']
log_loss_prod = log_loss(y_true, probabilities)
f1_score_prod = f1_score(y_true, predictions)

# Registrar as métricas no MLflow
mlflow.log_metric("log_loss_prod", log_loss_prod)
mlflow.log_metric("f1_score_prod", f1_score_prod)

# Salvar os resultados obtidos em uma tabela como um arquivo .parquet
results = pd.DataFrame({'prediction': predictions})
results.to_parquet("output/predictions.parquet")

# Finalizar a run do MLflow
mlflow.end_run()
