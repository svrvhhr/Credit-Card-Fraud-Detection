# smote_balancing.py

import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split

def apply_smote(input_file_path,output_file_path):
    # Chargement des données prétraitées
    data = pd.read_csv(input_file_path)
    
    # Séparation des caractéristiques et de la cible
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Vérification des proportions des classes avant équilibrage
    print(f"Répartition des classes avant équilibrage : {Counter(y)}")
    
    # Application de SMOTE pour équilibrer les classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Vérification des proportions des classes après équilibrage
    print(f"Répartition des classes après équilibrage : {Counter(y_resampled)}")
    
    # Sauvegarde du jeu de données équilibré
    balanced_data = pd.DataFrame(X_resampled, columns=X.columns)
    balanced_data['Class'] = y_resampled  # Ajout de la colonne cible
    balanced_data.to_csv(output_file_path, index=False)
    print(f"Jeu de données équilibré sauvegardé dans : {output_file_path}")
    
    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
    
    # Enregistrer les ensembles d'entraînement et de test pour une utilisation future.
    X_train.to_csv('../data/X_train.csv', index=False)
    X_test.to_csv('../data/X_test.csv', index=False)
    y_train.to_csv('../data/y_train.csv', index=False)
    y_test.to_csv('../data/y_test.csv', index=False)
    print("Données équilibrées et sauvegardées avec succès.")


input_file = '../data/preprocessed_creditcard.csv'  # Fichier prétraité
output_file = '../data/balanced_creditcard.csv'    # Fichier équilibré
apply_smote(input_file,output_file)
