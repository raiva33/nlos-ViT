import os
import random
import shutil

def split_folders(training_directory, validation_directory, split_ratio=0.2, random_seed=42):
    # Create the validation directory if it doesn't exist
    os.makedirs(validation_directory, exist_ok=True)

    # Get the list of object folders from the training directory
    object_folders = os.listdir(training_directory)
    print(object_folders)
    # Set the random seed for consistent shuffling
    random.seed(random_seed)

    for object_folder in object_folders:
        print(object_folder)
        # Create the corresponding object folder in the validation directory if it doesn't exist
        validation_object_folder = os.path.join(validation_directory, object_folder)
        os.makedirs(validation_object_folder, exist_ok=True)

        # Get the list of files within the object folder
        object_folder_path = os.path.join(training_directory, object_folder)
        files = os.listdir(object_folder_path)
        print(files)

        # Calculate the number of files to move
        num_files_to_move = int(len(files) * split_ratio)

        # Randomly select files to move
        files_to_move = random.sample(files, num_files_to_move)
        print(files_to_move)

        # Move files to the validation directory
        for file in files_to_move:
            print(file)
            src_path = os.path.join(object_folder_path, file)
            dest_path = os.path.join(validation_object_folder, file)
            print(dest_path)
            shutil.move(src_path, dest_path)


if __name__=="__main__":
    # Usage example
    training_directory_path = './datasets/train/'
    validation_directory_path = './datasets/validation/'

    split_folders(training_directory_path, validation_directory_path, split_ratio=0.2)