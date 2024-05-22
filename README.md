# Instructions

Suivez ces étapes exactement dans cet ordre pour exécuter les différents scripts et préparer le modèle :

1. **Téléchargement des données**
   - Exécutez le script `download_data.py` pour charger les images utilisées pour l'entraînement du modèle à partir de Google Drive. Assurez-vous que vous avez accès au drive contenant les données et que le script est correctement configuré pour télécharger les fichiers.
     ```bash
     python scripts/download_data.py
     ```

2. **Exploration des données**
   - Dans le dossier `notebook`, ouvrez et exécutez le notebook `exploration_file.ipynb` pour explorer les données et obtenir des informations sur les images présentes dans le dataset. Ce notebook vous aidera à comprendre la structure et la distribution de vos données.

3. **Renommage des images**
   - Exécutez le script `rename_images.py` pour renommer les images de manière à simplifier leur reconnaissance. Ce script permet de normaliser les noms des fichiers d'images pour faciliter les étapes suivantes.
     ```bash
     python scripts/rename_images.py
     ```

4. **Split des données**
   - Exécutez le script `split_data.py` pour diviser les données en ensembles d'entraînement, d'évaluation et de validation. Ce script s'assure que chaque ensemble contient une proportion représentative des différentes classes d'empreintes.
     ```bash
     python scripts/split_data.py
     ```

5. **Prétraitement des données**
   - Exécutez le script `data_preprocessing.py` pour préparer les images dans un format adapté aux modèles et effectuer des augmentations de données. Ce script inclut des étapes telles que le redimensionnement des images, la normalisation des pixels, et la génération d'images supplémentaires par des techniques d'augmentation de données.
     ```bash
     python scripts/data_preprocessing.py
     ```

6. **Entraînement du modèle**
   - Exécutez le script `train_model.py` pour entraîner le modèle sur les données prétraitées. Ce script contient la définition du modèle, la compilation, et les étapes d'entraînement incluant la sauvegarde du meilleur modèle trouvé.
     ```bash
     python scripts/train_model.py
     ```

7. **Évaluation du modèle**
   - Exécutez le script `evaluate_model.py` pour évaluer les performances du modèle sur l'ensemble de test. Ce script charge le modèle sauvegardé et calcule des métriques telles que l'exactitude, la précision, le rappel et le F1-score.
     ```bash
     python scripts/evaluate_model.py
     ```

8. **Test du modèle**
   - Exécutez le script `test_model.py` pour tester le modèle sur des images nouvelles ou spécifiques. Ce script permet de charger des images individuelles et d'obtenir des prédictions du modèle.
     ```bash
     python scripts/test_model.py
     ```

## Suivi des expérimentations avec MLflow

Pour suivre les diverses expérimentations sur votre modèle avec MLflow, tapez la commande suivante pour lancer l'interface utilisateur de MLflow :

```bash
mlflow ui
