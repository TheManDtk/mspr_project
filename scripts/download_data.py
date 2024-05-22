import os
import requests
import zipfile

def download_and_unzip(url, extract_to='.'):
    local_zip = 'data.zip'
    
    # Télécharger le fichier
    print(f"Téléchargement du fichier depuis {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_zip, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    # Vérifier le fichier téléchargé
    if not zipfile.is_zipfile(local_zip):
        print(f"Le fichier téléchargé {local_zip} n'est pas un fichier ZIP valide.")
        return
    
    # Décompresser le fichier
    print(f"Décompression de {local_zip} vers {extract_to}")
    with zipfile.ZipFile(local_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # Supprimer le fichier zip
    os.remove(local_zip)
    print(f"Suppression de {local_zip}")

if __name__ == "__main__":
    url = 'https://www.dropbox.com/scl/fi/sjmbilsh04l8vi0gzg73z/Mammiferes.zip?rlkey=e0tlvroeifmratf94sr3xbk7u&dl=1'  # Remplacement du lien
    extract_to = 'data/raw'  # Ajustement du chemin
    os.makedirs(extract_to, exist_ok=True)
    download_and_unzip(url, extract_to)
    print(f'Data downloaded and extracted to {extract_to}')
