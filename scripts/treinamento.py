import mlflow
import mlflow.sklearn
from sklearn.metrics import log_loss, f1_score, roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
from pycaret.classification import setup, create_model, predict_model
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

mlflow.set_tracking_uri("http://localhost:5000")

def treinamento():

    print("Treinamento")

    with mlflow.start_run(run_name="Treinamento"):
        data_train = pd.read_parquet("data/processed/data_filtered_dev.parquet")

        feature_names = data_train.dropna(subset=['shot_made_flag']).columns.tolist()

        clf = setup(data=data_train, target='shot_made_flag')

        log_reg = create_model('lr')

        data_test = pd.read_parquet("data/processed/data_filtered_prod.parquet")
        y_test = data_test['shot_made_flag']

        data_test = data_test[feature_names]

        y_pred_lr = predict_model(log_reg, data=data_test)['shot_made_flag']
        log_loss_lr = log_loss(y_test, y_pred_lr)

        mlflow.log_metric("log_loss_logistic_regression", log_loss_lr)

        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        precision_lr = precision_score(y_test, y_pred_lr)
        recall_lr = recall_score(y_test, y_pred_lr)
        f1_lr = f1_score(y_test, y_pred_lr)
        roc_auc_lr = roc_auc_score(y_test, y_pred_lr)

        mlflow.log_metric("accuracy_lr", accuracy_lr)
        mlflow.log_metric("precision_lr", precision_lr)
        mlflow.log_metric("recall_lr", recall_lr)
        mlflow.log_metric("f1_lr", f1_lr)
        mlflow.log_metric("roc_auc_lr", roc_auc_lr)

        plt.figure()
        sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues')
        plt.title("Matriz de Confusão para Regressão Logística")
        plt.xlabel("Previsto")
        plt.ylabel("Verdadeiro")
        plt.savefig("output/confusion_matrix_logistic_regression.png")
        plt.close()
        mlflow.log_artifact("output/confusion_matrix_logistic_regression.png")

        fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr)
        roc_auc_lr = auc(fpr_lr, tpr_lr)
        plt.figure()
        plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_lr)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falso Positivo')
        plt.ylabel('Taxa de Verdadeiro Positivo')
        plt.title('Curva ROC para Regressão Logística')
        plt.legend(loc="lower right")
        plt.savefig("output/roc_curve_logistic_regression.png")
        plt.close()
        mlflow.log_artifact("output/roc_curve_logistic_regression.png")

        dt = create_model('dt')

        y_pred_dt = predict_model(dt, data=data_test)['shot_made_flag']
        log_loss_dt = log_loss(y_test, y_pred_dt)
        f1_score_dt = f1_score(y_test, y_pred_dt, average='binary')

        mlflow.log_metric("log_loss_decision_tree", log_loss_dt)
        mlflow.log_metric("f1_score_decision_tree", f1_score_dt)

        plt.figure()
        sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Blues')
        plt.title("Matriz de Confusão para Árvore de Decisão")
        plt.xlabel("Previsto")
        plt.ylabel("Verdadeiro")
        plt.savefig("output/confusion_matrix_decision_tree.png")
        plt.close()
        mlflow.log_artifact("output/confusion_matrix_decision_tree.png")

        fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_dt)
        roc_auc_dt = auc(fpr_dt, tpr_dt)
        plt.figure()
        plt.plot(fpr_dt, tpr_dt, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_dt)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falso Positivo')
        plt.ylabel('Taxa de Verdadeiro Positivo')
        plt.title('Curva ROC para Árvore de Decisão')
        plt.legend(loc="lower right")
        plt.savefig("output/roc_curve_decision_tree.png")
        plt.close()
        mlflow.log_artifact("output/roc_curve_decision_tree.png")

        if log_loss_lr < log_loss_dt:
            final_model = log_reg
            model_type = "Regressão Logística"
        else:
            final_model = dt
            model_type = "Árvore de Decisão"
        try:
            mlflow.sklearn.save_model(final_model, "final_model")
        except:
            pass
        
        mlflow.sklearn.log_model(final_model, "Treinamento", registered_model_name="Treinamento")