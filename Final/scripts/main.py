import pandas as pd
from data_exploration import explore_data
from data_preprocessing import preprocess_data
from smote_balancing import apply_smote
from model_training import load_data,train_and_evaluate_models
from model_evaluation import load_model,evaluate_models
import joblib


def main():
    # Chargement des données
    print("Chargement des données...")

    # Chargement et Exploration des données
    print("\n--- Chargement et Exploration des données ---")
    explore_data()

    # Prétraitement des données
    print("\n--- Prétraitement des données ---")
    input_file = '../../../data/creditcard.csv'
    output_file = '../data/preprocessed_creditcard.csv'

    preprocess_data(input_file, output_file)

    # Application du suréchantillonnage SMOTE
    print("\n--- Application du SMOTE pour équilibrer les classes ---")
    input_file = '../data/preprocessed_creditcard.csv'  # Fichier prétraité
    output_file = '../data/balanced_data.csv'    # Fichier équilibré
    apply_smote(input_file,output_file)
    

    # Entraînement des modèles
    print("\n--- Entraînement des modèles ---")
    
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


if __name__ == "__main__":
    main()