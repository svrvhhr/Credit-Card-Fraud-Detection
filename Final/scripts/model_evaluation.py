# Importation des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import joblib
from sklearn.metrics import accuracy_score, f1_score

# Fonction pour charger les modèles
def load_model(model_path):
    return joblib.load(model_path)

# Fonction pour générer et afficher la matrice de confusion
def plot_confusion_matrix(cm, model_name, file_path):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title(f'Matrice de confusion - {model_name}')
    plt.xlabel('Prédictions')
    plt.ylabel('Vérités')
    plt.savefig(file_path)
    plt.show()

# Fonction pour afficher les rapports de classification
def print_classification_report(model_name, y_true, y_pred):
    print(f"=== {model_name} ===")
    print(classification_report(y_true, y_pred))

# Fonction pour générer et afficher la courbe ROC
def plot_roc_curve(fpr, tpr, auc_score, model_name):
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})")

# Fonction pour générer et afficher la courbe Précision-Rappel
def plot_precision_recall_curve(precision, recall, model_name):
    plt.plot(recall, precision, label=model_name)

# Fonction pour calculer et afficher les métriques des modèles
def evaluate_models(X_test, y_test, models):
    # Initialisation des variables pour les courbes et scores
    results = []
    
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, model_name, f'../results/report/evaluation/mc-{model_name}.png')

        # Rapport de classification
        print_classification_report(model_name, y_test, y_pred)

        # Calcul des courbes ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        auc_score = auc(fpr, tpr)
        plot_roc_curve(fpr, tpr, auc_score, model_name)

        # Calcul des courbes précision-rappel
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        plot_precision_recall_curve(precision, recall, model_name)

        # Ajout des résultats au tableau
        results.append({
            'Modèle': model_name,
            'Précision': accuracy_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'AUC': auc_score
        })
    
    # Affichage des résultats sous forme de DataFrame
    results_df = pd.DataFrame(results)
    print(results_df)
    
    # Visualisation des comparaisons des modèles
    results_df.set_index('Modèle').plot(kind='bar', figsize=(10, 6))
    plt.title("Comparaison des modèles")
    plt.ylabel("Score")
    plt.savefig('../results/report/evaluation/model-compare.png')
    plt.show()

# Chargement des données
X_test = pd.read_csv('../data/X_test.csv')
y_test = pd.read_csv('../data/y_test.csv')

# Chargement des modèles
models = {
    'Régression Logistique': load_model('../models/logistic_model.pkl'),
    'Arbre de Décision': load_model('../models/decision_tree_model.pkl'),
    'Forêt Aléatoire': load_model('../models/random_forest_model.pkl')
}

# Évaluation des modèles
evaluate_models(X_test, y_test, models)
