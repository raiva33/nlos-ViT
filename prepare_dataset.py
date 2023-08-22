import os
import numpy as np


def prepare_data(input,output):
    save_path = "./datasets/"
    array = []
    obj=os.listdir(input)
    print(obj)
    for objects in obj:
        print(input+objects)
        files=os.listdir(input+objects)
        print(files)
        files.sort()
        print(len(os.listdir(input+objects)))
        for file in files:
            print(input+objects+"/"+file)
            array.append(input+objects+"/"+file)
    # Set the random seed for consistent shuffling
    np.random.seed(42)
    # Shuffle the array using the fixed random seed
    np.random.shuffle(array)
    
    np.save(save_path+output,array)


if __name__ == '__main__':
    prepare_data("/code/datasets/train/feature/",'train_patterns.npy')
    prepare_data("/code/datasets/train/target/",'train_targets.npy')
    prepare_data("/code/datasets/validation/feature/",'val_patterns.npy')
    prepare_data("/code/datasets/validation/target/",'val_targets.npy')

