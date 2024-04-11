import mlflow
import pandas as pd
from sklearn.metrics import log_loss, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://localhost:5000")

with mlflow.start_run(run_name="PipelineAplicacao"):

    loaded_model = mlflow.sklearn.load_model("final_model")

    data_prod = pd.read_parquet("data/raw/dataset_kobe_prod.parquet")

    data_prod_cleaned = data_prod.dropna(subset=['shot_made_flag'])

    relevant_columns = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance']
    data_prod_relevant = data_prod_cleaned[relevant_columns]

    predictions = loaded_model.predict(data_prod_relevant)

    probabilities = loaded_model.predict_proba(data_prod_relevant)[:, 1]

    y_true = data_prod_cleaned['shot_made_flag']
    log_loss_prod = log_loss(y_true, probabilities)
    f1_score_prod = f1_score(y_true, predictions)

    mlflow.log_metric("log_loss_prod", log_loss_prod)
    mlflow.log_metric("f1_score_prod", f1_score_prod)

    results = pd.DataFrame({'prediction': predictions})
    results.to_parquet("output/predictions.parquet")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(probabilities, kde=True)
    plt.title('Distribuição das probabilidades previstas')
    plt.xlabel('Probabilidade')
    plt.ylabel('Frequência')

    plt.subplot(1, 2, 2)
    sns.countplot(x=predictions)
    plt.title('Contagem das previsões')
    plt.xlabel('Previsão')
    plt.ylabel('Contagem')

    plt.savefig("output/predictions_plots.png")
    mlflow.log_artifact("output/predictions_plots.png")

    mlflow.sklearn.log_model(
        loaded_model, 
        "final_model",
        registered_model_name="koube-predictor"
    )
