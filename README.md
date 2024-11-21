# ML 
# ML 
Ce code utilise un dataset de températures réelles provenant de données climatiques open-source pour prédire les températures futures à partir des 3 observations précédentes.

Chargement des données : Les températures moyennes sont extraites d'un dataset contenant des données climatiques mensuelles. Les valeurs manquantes sont remplies pour assurer la cohérence temporelle.
Préparation des données : Les températures sont organisées en séquences glissantes : 3 valeurs précédentes servent d'entrée pour prédire la suivante. Les données sont ensuite divisées en un ensemble d'entraînement (80 %) et un ensemble de test (20 %).
Normalisation : Les valeurs des températures sont mises à une échelle entre 0 et 1 pour optimiser le processus d'entraînement du modèle.
Création du modèle : Un modèle de réseau de neurones dense (MLP) est construit avec 3 couches : deux couches cachées avec 64 et 32 neurones et une couche de sortie pour prédire une température unique.
Entraînement : Le modèle est entraîné sur les données réelles pendant 20 cycles (epochs), en utilisant une fonction de perte pour minimiser l'erreur entre les prédictions et les valeurs réelles.
Prédictions et évaluation : Le modèle est testé sur les données de test, et ses prédictions sont comparées aux valeurs réelles. Un graphique visualise les différences entre les températures réelles et celles prédites pour évaluer les performances du modèle.