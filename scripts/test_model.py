import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import mlflow
import os

def load_and_prep_image(img_path, img_size=224):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalisation
    return img

def predict_image(model_path, img_path, class_indices):
    # Charger le modèle sauvegardé
    model = tf.keras.models.load_model(model_path)

    # Prétraiter l'image
    img = load_and_prep_image(img_path)

    # Prédiction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    
    # Mapping des indices de classe aux noms de classe
    class_labels = {v: k for k, v in class_indices.items()}
    predicted_class_name = class_labels[predicted_class[0]]
    
    # Afficher l'image avec la prédiction
    plt.imshow(image.load_img(img_path))
    plt.title(f'Predicted: {predicted_class_name}')
    plt.show()

    # Afficher les probabilités
    for i, prob in enumerate(predictions[0]):
        print(f"{class_labels[i]}: {prob:.4f}")

if __name__ == "__main__":
    #model_path = 'models/animal_footprint_classifier.h5'  # Chemin vers le modèle sauvegardé
    model_path = 'best_model_20240521_214333.h5'  # Chemin vers le modèle sauvegardé
   
    img_path = 'D:/wildlens/Mammifères/Chat/original.jpeg'  # Chemin vers l'image à tester
    class_indices = {
        'Ours': 0,
        'Renard': 1,
        'Chien': 2,
        'Lynx': 3,
        'Raton laveur': 4,
        'Puma': 5,
        'Coyote': 6,
        'Castor': 7,
        'Ecureuil': 8,
        'Chat': 9,
        'Loup': 10,
        'background': 11
    }

    predict_image(model_path, img_path, class_indices)
