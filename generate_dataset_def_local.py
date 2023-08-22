import numpy as np
from numpy import matlib
import scipy.io as sio
from scipy.sparse import lil_matrix, csr_matrix
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, ifftshift, fftn, ifftn
import math
from numpy import linalg
import torch
import time
import tal
import cv2
import os
import glob
import shutil
import sys
import h5py
from scipy.signal import find_peaks

x = 64
t = 4096
args = sys.argv

### GENERATE 
def render(x=0.0,y=0.0,z=0.5,obj="Z"):
    path = "datasets/feature_generation_yaml/"
    yaml_file="datasets/feature_generation_yaml/base.yaml"
    print(path)
    with open(yaml_file) as f:
        text = f.read()
    
    text_obj="datasets/xml/obj/Z.obj"
    text_x = "displacement_x: 0.0"
    text_y = "displacement_y: 0.0"
    text_z = "displacement_z: 1.0"

    new_text_obj = "datasets/xml/obj/{}.obj".format(obj)
    new_text_x = "displacement_x: " + str(x)
    new_text_y = "displacement_y: " + str(y)
    new_text_z = "displacement_z: " + str(z)

    new_text = text
    new_text = new_text.replace(text_obj, new_text_obj)
    new_text = new_text.replace(text_x,new_text_x)
    new_text = new_text.replace(text_y,new_text_y)
    new_text = new_text.replace(text_z,new_text_z)

    #print("new_text")
    #print(new_text)

    with open(yaml_file,"w") as f:
        f.write(new_text)
        f.close()
        
    code = "python3 -m tal render " + yaml_file

    os.system(code)

    files = os.listdir(path)
    print(files)
    files_dir = [f for f in files if os.path.isdir(os.path.join(path, f))]
    print(files_dir)
    files_dir.sort(key=lambda x: os.path.getctime(os.path.join(path, x)), reverse=True)
    # Select the newest folder (first element in the sorted list)
    newest_folder = files_dir[0]
    #  Construct the path to the newest folder
    created_folder = os.path.join(path, newest_folder)
    print(created_folder)
    hdf5_path = created_folder + "/base.hdf5" #Construct the path pattern to find HDF5 files within the created folder:
    print(hdf5_path)
    hdf5_file = glob.glob(hdf5_path)[0] #Find the first HDF5 file that matches the pattern:
    print(hdf5_file)

    with open(yaml_file,"w") as f: #Restore the original content of the YAML file by writing the initial text back to it:
        f.write(text)
        f.close()


    return hdf5_file,created_folder

def reconstruct(hdf5_file,obj_name,f_name):
    x = 64
    t = 4096
    # data = tal.io.read_capture(hdf5_file)
    # voxel = tal.reconstruct.filter_H(data, 'pf', wl_mean=0.005, wl_sigma=0.005 / np.sqrt(2))
    # voxel = np.transpose(voxel,(1, 2, 0))
    # voxel = voxel.astype(np.float)
    # print("func reconstruct")  
    # voxel = np.resize(voxel, (x,x,t))
    # print(voxel.shape)

    # Open the hdf5 file and read the images
    with h5py.File(hdf5_file, 'r') as f:
         images = np.array(f['H'])
    images_t=np.transpose(images,(1,2,0))

    # #Target generation
    # print("target generation")
    # nlos = tra.NLOS_numpy(bin_resolution=14e-12)
    # vol = nlos.transient_to_albedo(images_t)
    # vol = vol.transpose(1, 2, 0)
    # vol = np.flip(vol, axis=0)

    # filename = "/code/dataset_img/"  + f_name +".png"
    # img = np.amax(vol, axis=2)
    # fig, ax = plt.subplots(figsize=(img.shape[1]/10, img.shape[0]/10))
    # plt.imshow(img, aspect='auto')
    # plt.axis('tight')
    # plt.axis('off')
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # plt.savefig(filename)

    #Elimination of data after last peak and downsampling
    npy_file = "./train/feature/{}/{}".format(obj_name, f_name)
    g=reduce_image(images)

    #Save npy file
    print(g.shape)  
    np.save(npy_file, g)
    print(os.listdir("./train/feature/{}/".format(obj_name)))

def reduce_image(images):
    
    ### Find the last relevant information and delete the rest

    # Get the number of images and the image dimensions
    num_images, height, width = images.shape

    images_t=np.transpose(images,(1,2,0))

    # Loop through each image and plot a vertical line with a height proportional to the pixel intensity
    max_val=0
    intensities = []
    for i in range(num_images):
        intensity = np.sum(images[i]) / (height * width)  # calculate the average pixel intensity for the image
        intensities.append(intensity)
        if intensity > max_val:
            max_val=intensity

    # Smooth the intensity values using a moving average filter
    window_size = 3
    kernel = np.ones(window_size) / window_size
    intensities_smooth = np.convolve(intensities, kernel, mode='same')

    # Find all the local maxima in the smoothed intensities
    peaks, _ = find_peaks(intensities_smooth,height=2)
    small_peaks,_= find_peaks(intensities_smooth)
    print(peaks)
    print(small_peaks)

    if len(peaks) != len(small_peaks):
        #Find the last peak
        end_peak=np.where(small_peaks==peaks[-1])[0][0]+1
        last_peak=small_peaks[end_peak]
        print(last_peak)
        N=last_peak
    else:
        N=4096

    print(N)
    ### Downsampling: Average pooling
    # Assuming input_image is your 64x64xN image
    data=images_t.astype(complex)
    data_new=data[:,:,:N]
    input_image =  data_new+ 1j*data_new

    # Determine pooling size
    pooling_size = N // 64
    remainder = N % 64

    # If N is not an exact multiple of 64, you might want to truncate or pad your data
    # Here we'll truncate for simplicity
    if remainder != 0:
        input_image = input_image[:, :, :-remainder]

    # Reshape to prepare for pooling
    reshaped_image = input_image.reshape(64, 64, 64, pooling_size)

    # Perform average pooling across the last dimension
    g_new = reshaped_image.mean(axis=-1)

    print(g_new.shape)

    return g_new


if __name__ == '__main__':
    print(len(args),list(args))
    x=-0.45
    y=-0.45
    z=0.5
    l=0
    ## Select object to render
    list_object=["square","circle","circle_hole","traingle_hole","circle_extruded"]

    for obj_name in list_object:
        #Create folder in case it is the first time that you render this object
        folder_path="./train/feature/{}".format(obj_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder created: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")

        for i in range(0,19):
            for j in range(0,19):
                for e in range(0,6):
                    x = round(x,3)
                    y = round(y,3)
                    z = round(z,3)
                    print(x,y,z)
                    l=l+1
                    

                    #Create file name and check if it already exists
                    f_name = "x_{}_y_{}_z_{}.npy".format(x,y,z)
                    #file_with_extension = f_name + ".npy"  # Assuming the file has a .txt extension
                    directory = "./train/feature/{}/{}".format(obj_name,f_name)
                    print(directory)
                                
                    if os.path.exists(directory):
                        z = z + 0.25 
                        print("File already exists.")
                        continue
                    else:
                        print("File does not exist.")

                    #Render scene                
                    hdf5_file,created_folder = render(x,y,z,obj_name) #=args[2]
                    print("hdf5_file")
                    print(hdf5_file)
                    reconstruct(hdf5_file, obj_name ,f_name)

                    #Remove folder of rendered scene
                    shutil.rmtree(created_folder)
                    print(os.listdir("datasets/feature_generation_yaml"))
                    z = z + 0.25
                x= x + 0.05
                z=0.5
            y= y + 0.05
            x=-0.45
        y=-0.45
        print(l)