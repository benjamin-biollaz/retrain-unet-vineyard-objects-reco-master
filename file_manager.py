import os

class FileManager:

    ## Get the sample from path
    def get_sample(self, path, subfolder):
        sample_paths = sorted(
            [
                os.path.join(path, fname)
                for fname in os.listdir(path)
                if not fname.startswith('.') and fname.find(subfolder)
            ]
        )
        return sample_paths

    ## Get the file name and file extension
    def get_filename_n_extension(self, path):
        gfe_split_name = path.split(os.path.sep)
        gfe_file = gfe_split_name[-1]
        gfe_filename = gfe_file.split('.')[0].split('/')[-1]
        gfe_ext = '.' + gfe_file.split('.')[1]
        return gfe_filename, gfe_ext

    ## Remove existing patches
    def remove_patches(self, files_path, files_subfolder):
        sample_paths = self.get_sample(files_path + files_subfolder + '/', 'None')
        for path in sample_paths:
            os.remove(path)

        print('Successful patches removal from ' + files_path + files_subfolder + ' !')