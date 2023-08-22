import os
import random
import shutil

def select_random_files(source_dir, destination_dir, percentage):
    random.seed(42)
    file_list = os.listdir(source_dir)
    file_list.sort()  # Sort files alphabetically
    random.shuffle(file_list)
    num_files = int(len(file_list) * percentage)
    selected_files = file_list[:num_files]
    print(selected_files)
    if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
            print(f"Folder created: {destination_dir}")
    else:
        print(f"Folder already exists: {destination_dir}")

    for file_name in selected_files:
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(destination_dir, file_name)
        shutil.move(source_path, destination_path)
        print(f"Moved file: {file_name}")

if __name__=="__main__":
    percentage = 0.2
    path="/home/mao/Documents/code/mitsuba2-transient-nlos/datasets/train/feature"
    list_objects=["circle_hole"]#os.listdir(path)
    for object in list_objects:
        select_random_files("/home/mao/Documents/code/mitsuba2-transient-nlos/datasets/train/feature/{}/".format(object), "/home/mao/Documents/code/mitsuba2-transient-nlos/datasets/validation/feature/{}/".format(object), percentage)
        select_random_files("/home/mao/Documents/code/mitsuba2-transient-nlos/datasets/train/target/{}/".format(object), "/home/mao/Documents/code/mitsuba2-transient-nlos/datasets/validation/target/{}/".format(object), percentage)
        #select_random_files("/home/mao/Documents/code/mitsuba2-transient-nlos/datasets/train/target_jpg/{}/".format(object), "/home/mao/Documents/code/mitsuba2-transient-nlos/datasets/validation/target_jpg/{}/".format(object), percentage)
