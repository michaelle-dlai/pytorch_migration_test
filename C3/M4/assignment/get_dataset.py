# Script that dowloads the dataset from kaggle and puts it in the correct folder.
# The dataset is called "Street Cleanliness Classification" and is available at https://www.kaggle.com/datasets/nw8vqlafd/street-classification-dataset/data
# The dataset is in the folder "CleanStreetDataset" and is in the format of a zip file.
# The script should download the dataset, unzip it, and put it in the correct folder.
# The script should be called "download_dataset.py" and should be in the root of the project.
# The script should be run with the command "python download_dataset.py"

import kaggle
import zipfile
import os

kaggle.api.dataset_download_files('nw8vqlafd/street-classification-dataset', path='./dataset', unzip=True)

# Get all zip files in the dataset directory
zip_files = [f for f in os.listdir('./dataset') if f.endswith('.zip')]

# Extract each zip file
for zip_file in zip_files:
    zip_path = os.path.join('./dataset', zip_file)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('./dataset')
    
    # Remove the zip file after extraction
    os.remove(zip_path)

    # Create train, dev, test directories if they don't exist
    for split in ['train', 'dev', 'test']:
        split_path = os.path.join('./dataset', split)
        if not os.path.exists(split_path):
            os.makedirs(split_path)
            # Create subdirectories for each class
            os.makedirs(os.path.join(split_path, 'clean'))
            os.makedirs(os.path.join(split_path, 'dirty'))

    # Get list of files for each class
    clean_files = [f for f in os.listdir('./dataset/clean') if f.endswith(('.jpg', '.jpeg', '.png'))]
    dirty_files = [f for f in os.listdir('./dataset/dirty') if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Helper function to move files
    def move_files(files, src_dir, splits):
        import random
        random.shuffle(files)
        
        n_files = len(files)
        train_idx = int(0.7 * n_files)
        dev_idx = int(0.85 * n_files)
        
        # Split files according to ratios
        train_files = files[:train_idx]
        dev_files = files[train_idx:dev_idx]
        test_files = files[dev_idx:]
        
        # Move files to respective directories
        for f, split in zip([train_files, dev_files, test_files], ['train', 'dev', 'test']):
            for file in f:
                src = os.path.join('./dataset', src_dir, file)
                dst = os.path.join('./dataset', split, src_dir, file)
                os.rename(src, dst)

    # Move files for each class
    move_files(clean_files, 'clean', ['train', 'dev', 'test'])
    move_files(dirty_files, 'dirty', ['train', 'dev', 'test'])

    # Remove now-empty class directories
    os.rmdir('./dataset/clean')
    os.rmdir('./dataset/dirty')

