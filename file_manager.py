import os
import csv

class FileManager:

    # Get the sample from path
    def get_sample(self, path, subfolder):
        sample_paths = sorted(
            [
                os.path.join(path, fname)
                for fname in os.listdir(path)
                if not fname.startswith('.') and fname.find(subfolder)
            ]
        )
        return sample_paths

    # Get the file name and file extension
    def get_filename_n_extension(self, path):
        gfe_split_name = path.split(os.path.sep)
        gfe_file = gfe_split_name[-1]
        gfe_filename = gfe_file.split('.')[0].split('/')[-1]
        gfe_ext = '.' + gfe_file.split('.')[1]
        return gfe_filename, gfe_ext

    # Remove existing patches
    def remove_patches(self, files_path, files_subfolder):
        sample_paths = self.get_sample(files_path + files_subfolder + '/', 'None')
        for path in sample_paths:
            os.remove(path)

        print('Successful patches removal from ' + files_path + files_subfolder + ' !')

    def get_classes_encoding():
        with open('classes_encoding.csv', newline='') as csvfile:
            data = list(csv.reader(csvfile, delimiter=","))
            palette = [ [ None for y in range(3) ] for x in range( len(data) - 1) ]
            for i in range(len(data)):
                # Fist line corresponds to the csv column labels
                if (i == 0):
                    continue
                paletteIndex = i-1
                palette[paletteIndex][0] = data[i][0] # Name
                palette[paletteIndex][1] = int(data[i][1]) # Label
                # RGB value
                palette[paletteIndex][2] = (float(data[i][2]), float(data[i][3]), float(data[i][4]))
            
            return palette

    def create_dataset_directories(self):
        mandatory_directories = ["train/images", "train_labels/labels", "validation/images", "validation_labels/labels", "data_augmentation/images", "data_augmentation_labels/labels"]
        dataset_directory = "./datasets/"

        for dir in mandatory_directories:
            full_path = dataset_directory + dir
            if (not os.path.exists(full_path)):
                os.mkdir(full_path)