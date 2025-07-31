# 📊 Churn Prediction Project

Ce projet a pour objectif de prédire si un client risque de se désabonné au service de l'entreprise, en se basant sur des données historiques. L’approche est axée sur la **science des données** et le **machine learning supervisé**.

---

##  Objectifs du projet

- Comprendre les facteurs qui influencent le churn client
- Appliquer des techniques de **prétraitement de données**
- Entraîner plusieurs modèles de classification (RandomForest, Logistic Regression, etc.)
- Évaluer les performances avec des métriques adaptées

---

## 🗂 Contenu du projet

- `churnProjet.ipynb` : Notebook principal contenant l'analyse complète
- `data/` : Dossier contenant les données 
- `models/`: Sauvegarde des modèles entraînés si applicable

---

## 📦 Librairies utilisées

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

---
# Projet est divisé en deux grandes parties

## Première partie : Recherche dans un notebook  à fin de prendre le meilleur modèle
## deuxième partie : moduler le code de la recherche dans un fichier .py 

## 🔍 Étapes principales 

1. **Exploration des données**
2. **Prétraitement**
   - Gestion des valeurs manquantes
   - Encodage des variables catégorielles
   - Normalisation
3. **Modélisation**
   - Régression Logistique
   - Forêt Aléatoire (Random Forest)
   - Comparaison via validation croisée
4. **Évaluation**
   - Matrice de confusion
   - Accuracy, Recall, Precision
   - ROC AUC

---

## 🚀 Résultats

Les modèles ont permis de :
- Identifier les variables clés du churn
- Obtenir une performance acceptable 
- Comparer les modèles sur des jeux d'entraînement/test

---

## 📁 À venir / améliorations possibles

- Ajout d’un dashboard (via Streamlit par exemple)
- Intégration dans une API 
- Ajout de tests unitaires
- Tester d'autres algo de classification

---

## ✅ Prérequis

```bash
pip install -r requirements.txt
