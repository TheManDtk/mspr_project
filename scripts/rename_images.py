import os

def rename_images(data_dir):
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            for i, img_file in enumerate(os.listdir(label_path)):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    new_name = f"{label}{i + 1}.jpg"
                    new_path = os.path.join(label_path, new_name)
                    old_path = os.path.join(label_path, img_file)
                    os.rename(old_path, new_path)
                    print(f"Renommé {old_path} en {new_path}")

if __name__ == "__main__":
    data_dir = 'data/raw/Mammiferes'
    rename_images(data_dir)
    print(f"Renommage des images dans {data_dir} terminé.")

