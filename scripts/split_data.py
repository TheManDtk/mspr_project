import os
import shutil
import random
from pathlib import Path

def split_data(source_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # Créer les répertoires d'entraînement, de validation et de test s'ils n'existent pas
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(val_dir).mkdir(parents=True, exist_ok=True)
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    
    for label in os.listdir(source_dir):
        label_path = os.path.join(source_dir, label)
        if os.path.isdir(label_path):
            images = [f for f in os.listdir(label_path) if f.endswith(('png', 'jpg', 'jpeg'))]
            random.shuffle(images)
            
            train_split = int(len(images) * train_ratio)
            val_split = int(len(images) * val_ratio) + train_split
            
            train_images = images[:train_split]
            val_images = images[train_split:val_split]
            test_images = images[val_split:]
            
            for img in train_images:
                src_path = os.path.join(label_path, img)
                dest_path = os.path.join(train_dir, label)
                Path(dest_path).mkdir(parents=True, exist_ok=True)
                shutil.copy(src_path, os.path.join(dest_path, img))
                
            for img in val_images:
                src_path = os.path.join(label_path, img)
                dest_path = os.path.join(val_dir, label)
                Path(dest_path).mkdir(parents=True, exist_ok=True)
                shutil.copy(src_path, os.path.join(dest_path, img))
                
            for img in test_images:
                src_path = os.path.join(label_path, img)
                dest_path = os.path.join(test_dir, label)
                Path(dest_path).mkdir(parents=True, exist_ok=True)
                shutil.copy(src_path, os.path.join(dest_path, img))
                
            print(f'Classe {label}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test')

if __name__ == "__main__":
    source_dir = 'data/raw/Mammiferes'  # Chemin vers les données brutes
    train_dir = 'data/train'  # Chemin vers les données d'entraînement
    val_dir = 'data/val'      # Chemin vers les données de validation
    test_dir = 'data/test'    # Chemin vers les données de test

    split_data(source_dir, train_dir, val_dir, test_dir)
    print("Division des données terminée.")
