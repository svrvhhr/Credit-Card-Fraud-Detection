# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Fonction pour charger les données
def load_data(file_path):
    """Charge les données à partir d'un fichier CSV."""
    return pd.read_csv(file_path)

# Fonction pour afficher un aperçu des données
def data_overview(data):
    """Affiche un aperçu des premières lignes, des informations et des statistiques descriptives des données."""
    print(data.head())
    print(data.info())
    print(data.describe())

# Fonction pour afficher la distribution des classes
def plot_class_distribution(data):
    """Affiche la distribution des classes dans les données."""
    class_distribution = data['Class'].value_counts(normalize=True) * 100
    print("Distribution des classes :")
    print(class_distribution)
    sns.countplot(x='Class', data=data)
    plt.title('Distribution des Classes (0 : Légitime, 1 : Frauduleux)')
    plt.show()

# Fonction pour afficher la distribution de la variable 'Time'
def plot_time_distribution(data):
    """Affiche la distribution de la variable 'Time'."""
    plt.figure(figsize=(10, 5))
    sns.histplot(data['Time'], bins=50, kde=True, color='blue')
    plt.title('Distribution de la variable Time')
    plt.xlabel('Temps écoulé (secondes)')
    plt.savefig('../results/figures/distribution-time.png')
    plt.show()

# Fonction pour afficher la distribution de la variable 'Amount'
def plot_amount_distribution(data):
    """Affiche la distribution de la variable 'Amount'."""
    plt.figure(figsize=(10, 5))
    sns.histplot(data['Amount'], bins=50, kde=True, color='green')
    plt.title('Distribution de la variable Amount')
    plt.xlabel('Montant de la transaction')
    plt.savefig('../results/figures/distribution-amount.png')
    plt.show()

# Fonction pour afficher les transactions frauduleuses contre le temps
def plot_fraudulent_transactions(data):
    """Affiche les transactions frauduleuses dans le temps."""
    fraudulent_data = data[data['Class'] == 1]
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Time', y='Amount', data=fraudulent_data, hue='Class', palette=['#FF6347'], alpha=0.6)
    plt.title('Transactions frauduleuses dans le temps')
    plt.xlabel('Temps (secondes depuis le début de la période)')
    plt.ylabel('Montant de la transaction')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig('../results/figures/transactions_frauduleuses.png')
    plt.legend(title='Class', labels=['Frauduleuses'])
    plt.show()

# Fonction pour afficher les transactions non frauduleuses contre le temps
def plot_non_fraudulent_transactions(data):
    """Affiche les transactions non frauduleuses dans le temps."""
    non_fraudulent_data = data[data['Class'] == 0]
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Time', y='Amount', data=non_fraudulent_data, hue='Class', palette=['#28A745'], alpha=0.6)
    plt.title('Transactions non frauduleuses dans le temps')
    plt.xlabel('Temps (secondes depuis le début de la période)')
    plt.ylabel('Montant de la transaction')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='Class', labels=['Non-frauduleuses'])
    plt.savefig('../results/figures/transactions_non_frauduleuses.png')
    plt.show()

# Fonction pour afficher un boxplot de la distribution des montants des transactions
def plot_amount_boxplot(data):
    """Affiche un boxplot de la distribution des montants des transactions."""
    plt.figure(figsize=(15, 6))
    sns.boxplot(x='Class', y='Amount', data=data, palette='coolwarm')
    plt.title('Distribution du montant des transactions (fraudes vs non-fraudes)')
    plt.xlabel('Classe (0 = Non-frauduleuse, 1 = Frauduleuse)')
    plt.ylabel('Montant de la transaction')
    plt.show()

# Fonction pour afficher la matrice de corrélation des caractéristiques
def plot_correlation_matrix(data):
    """Affiche la matrice de corrélation des caractéristiques V1 à V28."""
    corr_matrix = data.iloc[:, 1:30].corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Matrice de corrélation des caractéristiques (V1 à V28)')
    plt.savefig('../results/figures/matrice-corre.png')
    plt.show()

# Main : Fonction principale pour exécuter toutes les étapes
def explore_data():
    # Charger les données
    data = load_data('../../../data/creditcard.csv')
    
    # Afficher un aperçu des données
    data_overview(data)
    
    # Afficher la distribution des classes
    plot_class_distribution(data)
    
    # Afficher la distribution des variables Time et Amount
    plot_time_distribution(data)
    plot_amount_distribution(data)
    
    # Afficher les transactions frauduleuses et non frauduleuses
    plot_fraudulent_transactions(data)
    plot_non_fraudulent_transactions(data)
    
    # Afficher le boxplot des montants des transactions
    plot_amount_boxplot(data)
    
    # Afficher la matrice de corrélation des caractéristiques
    plot_correlation_matrix(data)

