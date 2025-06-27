import os
import glob
from config import DATASET_PATH

def rename_files(directory, base_name_length=6):
    # Encontrar todos os arquivos .jpg no diretório
    image_files = glob.glob(os.path.join(directory, '*.jpg'))
    
    # Ordenar os arquivos para garantir ordem consistente
    image_files.sort()
    
    # Renomear cada arquivo
    for i, image_file in enumerate(image_files, 1):
        # Gerar novo nome numérico
        new_name = f"{i:0{base_name_length}d}"
        
        # Construir novos paths
        new_image_path = os.path.join(directory, f"{new_name}.jpg")
        annotation_file = image_file.replace('.jpg', '.txt')
        new_annotation_path = os.path.join(directory, f"{new_name}.txt")
        
        # Renomear arquivos
        os.rename(image_file, new_image_path)
        if os.path.exists(annotation_file):
            os.rename(annotation_file, new_annotation_path)
        
        print(f"Renamed: {image_file} -> {new_image_path}")

# Diretórios a serem processados
directories = [DATASET_PATH + '/train', DATASET_PATH + '/valid', DATASET_PATH + '/test']

for dir_name in directories:
    print(f"Processing directory: {dir_name}")
    rename_files(dir_name)
    print()

print("Renaming completed!")