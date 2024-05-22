import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import mlflow
import mlflow.tensorflow
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from datetime import datetime

from data_preprocessing import create_data_generators, create_test_generator

def build_model(num_classes, learning_rate=0.0001, fine_tune_start=15):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[fine_tune_start:]:
        layer.trainable = True

    model = Sequential([
        tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (224, 224))),
        base_model,
        Flatten(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(train_dir, val_dir, test_dir, epochs=100, batch_size=64):
    train_gen, val_gen, test_gen = create_data_generators(train_dir, val_dir, test_dir, batch_size=batch_size)
    num_classes = len(train_gen.class_indices)
    class_names = list(train_gen.class_indices.keys())

    model = build_model(num_classes)

    mlflow.tensorflow.autolog()

    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")


    # Define the file path for saving the best model
    best_model_path = f'../wildlens_api/backendapi/model_ia/modelvgg16_{start_time}.h5'

    checkpoint_cb = ModelCheckpoint(best_model_path, save_best_only=True, monitor='val_loss', mode='min')
    early_stopping_cb = EarlyStopping(patience=30, restore_best_weights=True)

    with mlflow.start_run() as run:
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=[checkpoint_cb, early_stopping_cb]
        )

        model_path = os.path.join('models', f'animal_footprint_classifier_{start_time}.h5')
        model.save(model_path)
        model.save(f'../wildlens_api/backendapi/model_ia/modelvgg16_{start_time}.h5')

        mlflow.log_artifact(model_path, artifact_path='models')

        test_loss, test_acc = model.evaluate(test_gen)
        print(f'Précision sur les données de test: {test_acc:.4f}')

        test_gen.reset()
        y_pred = np.argmax(model.predict(test_gen), axis=-1)
        y_true = test_gen.classes

        conf_matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Prédit')
        plt.ylabel('Réel')
        plt.title('Matrice de Confusion')
        plt.show()
        print(conf_matrix)

        report = classification_report(y_true, y_pred, target_names=class_names)
        print(report)

        plt.figure()
        plt.plot(history.history['accuracy'], label='Précision Entraînement')
        plt.plot(history.history['val_accuracy'], label='Précision Validation')
        plt.xlabel('Époque')
        plt.ylabel('Précision')
        plt.legend(loc='lower right')
        plt.title('Précision du Modèle')
        plt.show()

        plt.figure()
        plt.plot(history.history['loss'], label='Perte Entraînement')
        plt.plot(history.history['val_loss'], label='Perte Validation')
        plt.xlabel('Époque')
        plt.ylabel('Perte')
        plt.legend(loc='upper right')
        plt.title('Perte du Modèle')
        plt.show()

    return history

if __name__ == "__main__":
    train_dir = 'data/train'
    val_dir = 'data/val'
    test_dir = 'data/test'
    train_model(train_dir, val_dir, test_dir, epochs=100)
