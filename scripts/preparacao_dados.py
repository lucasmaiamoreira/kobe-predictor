import pandas as pd
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

mlflow.set_tracking_uri("http://localhost:5000")

artifacts_dir = "artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

def plot_histograms_and_countplots(data):
    for column in ['lat', 'lon', 'minutes_remaining', 'period', 'shot_distance']:
        plt.figure()
        sns.histplot(data[column], kde=True)
        plt.title(f'Histogram for {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(artifacts_dir, f"histogram_{column}.png"))
        plt.close()

    for column in ['playoffs', 'shot_made_flag']:
        plt.figure()
        sns.countplot(x=column, data=data)
        plt.title(f'Countplot for {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.savefig(os.path.join(artifacts_dir, f"countplot_{column}.png"))
        plt.close()

def plot_parameter_validation_curve(param_name, grid_search, model, model_name, scoring, logx, X_train, y_train):
    plt.figure(figsize=(6,4))
    train_scores, test_scores = validation_curve(model, X=X_train, y=y_train, param_name=param_name, param_range=grid_search[param_name], scoring=scoring, cv=5, n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title("Validation Curve for " + model_name)
    plt.xlabel(param_name)
    plt.ylabel("Score ("+scoring+")")
    if logx:
        plt.semilogx(grid_search[param_name], train_scores_mean,'-o', label="Train", color="darkorange", lw=2)
        plt.semilogx(grid_search[param_name], test_scores_mean,'-o', label="Cross-Validation", color="navy", lw=2)
    else:
        plt.plot(grid_search[param_name], train_scores_mean,'-o', label="Train", color="darkorange", lw=2)
        plt.plot(grid_search[param_name], test_scores_mean,'-o', label="Cross-Validation", color="navy", lw=2)
    plt.fill_between(grid_search[param_name], train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=2)
    plt.fill_between(grid_search[param_name], test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=2)
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(artifacts_dir, f"validation_curve_{model_name}.png"))

def plot_learning_curve(model, model_name, scoring, train_sizes, X_train, y_train):
    plt.figure(figsize=(6,4))
    ax = plt.gca()
    ax.set_title('Learning Curve (%s)'%model_name)
    ax.set_xlabel("Train Examples")
    ax.set_ylabel("Score (" + scoring + ")")
    train_sizes, train_scores, test_scores = learning_curve(model, X=X_train, y=y_train, cv=5, n_jobs=-1, train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax.grid()
    ax.plot(train_sizes, train_scores_mean, 'o-', color="darkorange", label="Train")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="navy", label="Cross-Validation")
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="darkorange")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="navy")
    ax.legend(loc="best")
    plt.savefig(os.path.join(artifacts_dir, f"learning_curve_{model_name}.png"))

def realizar_subamostragem(X, y):
    under_sampler = RandomUnderSampler(random_state=42)
    X_under, y_under = under_sampler.fit_resample(X, y)
    return X_under, y_under

def realizar_sobreamostagem(X, y):
    over_sampler = RandomOverSampler(random_state=42)
    X_over, y_over = over_sampler.fit_resample(X, y)
    return X_over, y_over


def preparacao_dados():
    
    with mlflow.start_run(run_name="PreparacaoDados"):
        data_dev = pd.read_parquet("data/raw/dataset_kobe_dev.parquet")
        data_prod = pd.read_parquet("data/raw/dataset_kobe_prod.parquet")

        relevant_columns = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']
        data_dev_filtered = data_dev[relevant_columns].dropna()
        data_prod_filtered = data_prod[relevant_columns].dropna()
        print(data_prod_filtered)

        data_dev_filtered.to_parquet("data/processed/data_filtered_dev.parquet")
        data_prod_filtered.to_parquet("data/processed/data_filtered_prod.parquet")

        X = data_dev_filtered.drop(columns=['shot_made_flag'])
        y = data_dev_filtered['shot_made_flag']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        print("Balanceamento antes da subamostragem:")
        print(y_train.value_counts())

        X_under, y_under = realizar_subamostragem(X_train, y_train)

        print("Balanceamento após a subamostragem:")
        print(pd.Series(y_under).value_counts())

        X_over, y_over = realizar_sobreamostagem(X_train, y_train)

        print("Balanceamento após a sobreamostragem:")
        print(pd.Series(y_over).value_counts())

        mlflow.log_param("balanceamento_antes_subamostragem", y_train.value_counts().to_dict())
        mlflow.log_param("balanceamento_apos_subamostragem", pd.Series(y_under).value_counts().to_dict())
        mlflow.log_param("balanceamento_apos_sobreamostagem", pd.Series(y_over).value_counts().to_dict())

        sk_model = RandomForestClassifier()
        sk_model.fit(X_over, y_over)

        mlflow.sklearn.log_model(
            sk_model, 
            "PreparacaoDados",
            registered_model_name="PreparacaoDados")
        
        y_pred = sk_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc_score", roc_auc)

        percent_test = len(X_test) / (len(X_train) + len(X_test))

        mlflow.log_param("percent_test", percent_test)
        mlflow.log_metric("train_size", len(X_train))
        mlflow.log_metric("test_size", len(X_test))

        plot_histograms_and_countplots(pd.concat([X_train, y_train], axis=1))
        plot_histograms_and_countplots(pd.concat([X_test, y_test], axis=1))

        plot_parameter_validation_curve('n_estimators', {'n_estimators': [10, 50, 100, 200]}, sk_model, "RandomForestClassifier", "accuracy", logx=True, X_train=X_over, y_train=y_over)
        plot_learning_curve(sk_model, "RandomForestClassifier", "accuracy", np.linspace(.1, 1.0, 10), X_over, y_over)

        mlflow.log_artifacts(artifacts_dir)
