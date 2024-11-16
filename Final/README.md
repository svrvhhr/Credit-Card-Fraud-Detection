# Détection de Fraude des Cartes Bancaires

## Qu'est-ce que la fraude des cartes bancaires ?  
La fraude des cartes bancaires désigne toute activité non autorisée utilisant une carte de crédit ou de débit pour effectuer des transactions financières illégitimes. Cela peut inclure l'utilisation de cartes volées, la création de fausses cartes ou l'exploitation de données de paiement compromises. La détection de ces activités est cruciale pour réduire les pertes financières et protéger les utilisateurs.

---

## Présentation de la problématique  
La détection des transactions frauduleuses dans des données financières est un défi majeur en raison de la forte déséquilibre des classes : les transactions frauduleuses représentent une part infime du total, mais leurs impacts financiers et sécuritaires sont importants. De plus, il est nécessaire d'identifier ces transactions avec précision, sans compromettre les transactions légitimes.

Dans ce projet, nous nous concentrons sur l'application de techniques d’apprentissage automatique pour détecter ces fraudes dans un jeu de données fortement déséquilibré. Nous devons également évaluer les modèles avec des métriques adaptées pour garantir leur performance.

---

## Présentation du jeu de données  
Le jeu de données utilisé contient des transactions effectuées par des porteurs de cartes européens en septembre 2013. Il se compose de **284,807 transactions**, dont **492 sont frauduleuses**, soit seulement **0,172%** de l’ensemble des données.

### **Caractéristiques des variables**  
- Les **variables d’entrée** (`V1` à `V28`) résultent d'une **Analyse en Composantes Principales (ACP)** pour préserver la confidentialité des données.  
- Les deux seules variables **non transformées** sont :  
  - `Time` : Temps écoulé entre la première transaction et la transaction en cours.  
  - `Amount` : Montant de la transaction.  
- La **variable cible** (`Class`) indique si une transaction est frauduleuse ou non :  
  - `0` : Transaction légitime.  
  - `1` : Transaction frauduleuse.  

---

## Étapes du projet  

- [ ] **Analyse exploratoire des données**  
  - Comprendre la structure et les statistiques du jeu de données.  
  - Visualiser le déséquilibre des classes.  

- [ ] **Prétraitement des données**  
  - Nettoyer et normaliser les données (`Time`, `Amount`).  
  - Appliquer la technique **SMOTE** pour équilibrer les classes.

- [ ] **Entraînement des modèles**  
  - Tester différents algorithmes : régression logistique, arbres de décision, forêts aléatoires, etc.  
  - Réaliser un réglage des hyperparamètres.

- [ ] **Évaluation des modèles**  
  - Calculer les métriques adaptées à un jeu de données déséquilibré :  
    - Précision, rappel, F1-score, courbe ROC-AUC.  
  - Comparer les performances des modèles.

- [ ] **Visualisation et interprétation des résultats**  
  - Générer des graphiques : matrices de confusion, courbes ROC.  
  - Résumer les résultats obtenus dans un rapport.

- [ ] **Industrialisation du projet**  
  - Structurer le code en modules réutilisables.  
  - Sauvegarder les modèles entraînés et fournir un script principal pour l’exécution.

---

Ce projet vise à fournir un système performant pour détecter les transactions frauduleuses, avec un pipeline clair et des résultats interprétables. 🎯
