import tensorflow as tf
import mlflow
import mlflow.tensorflow
import os
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import create_test_generator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import datetime

def plot_confusion_matrix(cm, class_names, filename='confusion_matrix.png'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    mlflow.log_artifact(filename)

def plot_confusion_matrix_fp_fn(cm, filename='confusion_matrix_fp_fn.png'):
    tn, fp, fn, tp = cm.ravel()
    cm_fp_fn = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm_fp_fn, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (FP/FN)')
    plt.savefig(filename)
    mlflow.log_artifact(filename)

def plot_learning_curves(history, filename='learning_curves.png'):
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.savefig(filename)
    mlflow.log_artifact(filename)

def save_metrics_to_file(metrics, filename):
    with open(filename, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    mlflow.log_artifact(filename)

def evaluate_model(model_path, test_dir, history, batch_size=32):
    # Charger le modèle sauvegardé
    model = tf.keras.models.load_model(model_path)

    # Générer les données de test
    test_generator = create_test_generator(test_dir, batch_size=batch_size)

    # Évaluation du modèle
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # Prédictions
    test_generator.reset()
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    # Classification report
    class_names = list(test_generator.class_indices.keys())
    print("Classification Report:")
    classification_report_str = classification_report(y_true, y_pred, target_names=class_names)
    print(classification_report_str)

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Afficher et enregistrer la matrice de confusion
    plot_confusion_matrix(cm, class_names)

    # Afficher et enregistrer la matrice de confusion FP/FN
    plot_confusion_matrix_fp_fn(cm)

    # Tracer et enregistrer les courbes d'apprentissage
    plot_learning_curves(history)

    # Enregistrer des métriques et des rapports
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    metrics_filename = f"metrics_{now}.txt"
    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "classification_report": classification_report_str,
        "confusion_matrix": np.array2string(cm)
    }
    save_metrics_to_file(metrics, metrics_filename)

    # Enregistrement des métriques dans MLflow
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_text(classification_report_str, "classification_report.txt")
    mlflow.log_text(np.array2string(cm), "confusion_matrix.txt")

if __name__ == "__main__":
    model_path = 'models/animal_footprint_classifier.h5'  # Chemin vers le modèle sauvegardé
    test_dir = 'data/test'  # Chemin vers les données de test
    mlflow.start_run()  # Démarrer une nouvelle run dans MLflow
    history = None  # Charger l'historique de l'entraînement si disponible, sinon None
    evaluate_model(model_path, test_dir, history)
    mlflow.end_run()  # Terminer la run
