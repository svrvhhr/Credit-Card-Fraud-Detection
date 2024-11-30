# Importation des bibliothèques
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import joblib

# Fonction pour charger les données
def load_data():
    X_train = pd.read_csv('../data/X_train.csv')
    X_test = pd.read_csv('../data/X_test.csv')
    y_train = pd.read_csv('../data/y_train.csv')
    y_test = pd.read_csv('../data/y_test.csv')
    
    print(f"Taille de l'ensemble d'entraînement : {X_train.shape}, {y_train.shape}")
    print(f"Taille de l'ensemble de test : {X_test.shape}, {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

# Fonction pour entraîner un modèle et faire des prédictions
def train_and_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train.values.ravel())  # Entraînement du modèle
    y_pred = model.predict(X_test)  # Prédictions
    return y_pred

# Fonction pour évaluer un modèle
def evaluate_model(model_name, y_test, y_pred):
    print(f"=== {model_name} ===")
    print(classification_report(y_test, y_pred))
    print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))

# Fonction pour calculer et afficher les courbes ROC
def plot_roc_curve(y_test, y_pred_logistic_prob, y_pred_tree_prob, y_pred_forest_prob):
    fpr_log, tpr_log, _ = roc_curve(y_test, y_pred_logistic_prob)
    fpr_tree, tpr_tree, _ = roc_curve(y_test, y_pred_tree_prob)
    fpr_forest, tpr_forest, _ = roc_curve(y_test, y_pred_forest_prob)

    auc_log = auc(fpr_log, tpr_log)
    auc_tree = auc(fpr_tree, tpr_tree)
    auc_forest = auc(fpr_forest, tpr_forest)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr_log, tpr_log, label=f"Régression Logistique (AUC = {auc_log:.2f})")
    plt.plot(fpr_tree, tpr_tree, label=f"Arbre de Décision (AUC = {auc_tree:.2f})")
    plt.plot(fpr_forest, tpr_forest, label=f"Forêt Aléatoire (AUC = {auc_forest:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Chance")
    plt.xlabel("Taux de Faux Positifs")
    plt.ylabel("Taux de Vrais Positifs")
    plt.title("Courbe ROC")
    plt.legend()
    plt.savefig('../results/figures/courbe-ROC.png')
    plt.show()

# Fonction pour entraîner plusieurs modèles et évaluer
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Instanciation des modèles
    logistic_model = LogisticRegression(random_state=42, max_iter=1000)
    decision_tree_model = DecisionTreeClassifier(random_state=42)
    random_forest_model = RandomForestClassifier(random_state=42)

    # Entraînement et prédictions
    y_pred_logistic = train_and_predict(logistic_model, X_train, y_train, X_test)
    y_pred_tree = train_and_predict(decision_tree_model, X_train, y_train, X_test)
    y_pred_forest = train_and_predict(random_forest_model, X_train, y_train, X_test)

    # Évaluation des modèles
    evaluate_model("Régression Logistique", y_test, y_pred_logistic)
    evaluate_model("Arbre de Décision", y_test, y_pred_tree)
    evaluate_model("Forêt Aléatoire", y_test, y_pred_forest)

    # Probabilités prédites pour les courbes ROC
    y_pred_logistic_prob = logistic_model.predict_proba(X_test)[:, 1]
    y_pred_forest_prob = random_forest_model.predict_proba(X_test)[:, 1]
    y_pred_tree_prob = decision_tree_model.predict_proba(X_test)[:, 1]

    # Affichage des courbes ROC
    plot_roc_curve(y_test, y_pred_logistic_prob, y_pred_tree_prob, y_pred_forest_prob)

    # Sauvegarde des modèles
    joblib.dump(logistic_model, '../models/logistic_model.pkl')
    joblib.dump(decision_tree_model, '../models/decision_tree_model.pkl')
    joblib.dump(random_forest_model, '../models/random_forest_model.pkl')

    print("Modèles sauvegardés avec succès.")


X_train, X_test, y_train, y_test = load_data()
train_and_evaluate_models(X_train, X_test, y_train, y_test)
