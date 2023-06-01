## DATA AUGMENTATION ONLINE CONTROLE - OBTENIR LA LISTE DES PATCHES AVEC LA DATE ET L'HEURE DE CREATION
# Les patches ont été sauvegardés avec le paramètre save_to_dir de la fonction ImageDataGenerator hors du projet.
# Ce script permet d'obtenir la liste avec l'ID de l'augmentation ainsi que la date et l'heure de création.

import os
import sys
from datetime import datetime

def get_sample(path, subfolder):
    sample_paths = sorted(
        [
            os.path.join(path, fname)
            for fname in os.listdir(path)
            if not fname.startswith('.') and fname.find(subfolder)
        ]
    )
    return sample_paths

sys.stdout = open('../DATASETS/augmentation_online/list_' + datetime.now().strftime('%Y%m%d-%H%M%S') + '.txt', 'w')
list_augmentation = get_sample(path='../DATASETS/augmentation_online/iteration_09-01_batch-1/images/', subfolder='None')
for i in list_augmentation:
    date = datetime.fromtimestamp(os.path.getctime(i)).strftime('%Y-%m-%d %H:%M:%S:%f')
    print(i + ';' + date)
sys.stdout.close()

# print(datetime.fromtimestamp(os.path.getctime('../DATASETS/augmentation_online/iteration_09-01_batch-1/images/aug_0_7558158.png')).strftime('%Y-%m-%d %H:%M:%S'))