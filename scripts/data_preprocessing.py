# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_file_path, output_file_path):
    # Chargement des données
    data = pd.read_csv(input_file_path)
    
    # Vérification des dimensions et aperçu des données
    print(f"Dimensions des données : {data.shape}")
    print(data.head())
    
    # Vérification des valeurs manquantes
    missing_values = data.isnull().sum()
    print("Valeurs manquantes par colonne :\n", missing_values[missing_values > 0])
    
    # Standardisation des colonnes 'Time' et 'Amount'
    scaler = StandardScaler()
    data[['Time', 'Amount']] = scaler.fit_transform(data[['Time', 'Amount']])
    
    # Vérification des données normalisées
    print(data[['Time', 'Amount']].describe())
    
    # Sauvegarde des données prétraitées
    data.to_csv(output_file_path, index=False)
    print(f"Données prétraitées et sauvegardées dans : {output_file_path}")


input_file = '../../../data/creditcard.csv'
output_file = '../data/preprocessed_creditcard.csv'

preprocess_data(input_file, output_file)
