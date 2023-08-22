import os
import numpy as np
import mitsuba
from PIL import Image
import os
import glob
import shutil
import re
import random
import shutil
import cv2

# Set the desired mitsuba variant
mitsuba.set_variant('scalar_rgb')

from mitsuba.core import Bitmap, Struct, Thread
from mitsuba.core.xml import load_file

#importing the os module
import os

#to get the current working directory
directory = os.getcwd()

print(directory)

def ground_truth_render(xml_path, dir_purpose, f_name, obj_name):
    
    # Load the scene
    scene = load_file(xml_path)

    # Render the scene using a simple integrator
    scene.integrator().render(scene, scene.sensors()[0])

    # # Retrieve the film and save the output image
    film = scene.sensors()[0].film()
    # film.set_destination_file('output_image.exr')
    # film.develop()

    # Save the output as a PNG
    film.set_destination_file('output_image.exr')
    film.develop()

    # Load the output image for viewing
    # bitmap = Bitmap('output_image.exr')
    bitmap = film.bitmap(raw=True).convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, srgb_gamma=True)

    bitmap.write('output_image.jpg')

    # Display the image using matplotlib
    image = np.array(bitmap)

    # Convert the RGB image to grayscale
    gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    gray_image=cv2.GaussianBlur(gray_image, (3, 3), 0)
    gray_image=gray_image/gray_image.max()

    #Save file as npy
    npy_file = "./datasets/{}/target/{}/{}.npy".format(dir_purpose,obj_name,f_name)
    print(gray_image.shape)  
    np.save(npy_file, gray_image)

    #Save as jpg to check if it is rendering correctly
    new_p = Image.fromarray(gray_image*255)
    new_p = new_p.convert("L")
    new_p.save("./datasets/{}/target_jpg/{}/{}.jpg".format(dir_purpose,obj_name,f_name))

def create_ground_truth(x,y,z,r,xlm_path,dir_purpose,f_name,obj_name,scale):

    with open(xlm_path) as f:
        text = f.read()

    text_obj="./obj/Z.obj"
    text_scale='scale x="1" y="1" z="1"'
    text_trans = 'translate x="0.0" y="0.0" z="0.0"'
    text_rot_z= 'rotate z="1" angle="0.0"'
    new_text_obj="./obj/{}.obj".format(obj_name)
    new_text_scale='scale x="{}" y="{}" z="{}"'.format(scale,scale,scale)
    new_text_trans = 'translate x="{}" y="{}" z="-{}"'.format(x,y,z)
    new_text_rot_z= 'rotate z="1" angle="{}"'.format(r)    

    new_text = text
    new_text = new_text.replace(text_obj,new_text_obj)
    new_text = new_text.replace(text_scale,new_text_scale)
    new_text = new_text.replace(text_trans,new_text_trans)
    new_text = new_text.replace(text_rot_z,new_text_rot_z)

    with open(xlm_path,"w") as f:
        f.write(new_text)
        f.close()

    ## Code rendering the GT----------------------------------------
    ground_truth_render(xlm_path,dir_purpose,f_name, obj_name)

    #---------------------------------------------------------------
    with open(xlm_path,"w") as f: #Restore the original content of the YAML file by writing the initial text back to it:
        f.write(text)
        f.close()
    


if __name__ == '__main__':

    ## SELECT OBJECT THAT WANTS TO BE RENDERED-------------------------------------------##
    #Choose between "validation" or "train" to send the image to one folder or the other
    dir_purpose="train"

    # feature_path="./datasets/{}/feature/".format(dir_purpose)
    # list_objects= os.listdir(feature_path)
    list_objects=["circle_hole"]
    scale=0.01
    ## ----------------------------------------------------------------------------------##

    xlm_path="./datasets/xml/base.xml"
    npy_file = ""

    for object_name in list_objects:
        # if object_name in ["Z","bunny"]:
        #     continue
        folder_path=["./datasets/{}/target/{}".format(dir_purpose,object_name),"./datasets/{}/target_jpg/{}".format(dir_purpose,object_name)]
        for path in folder_path:
            print(path)
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"Folder created: {path}")
            else:
                print(f"Folder already exists: {path}")

        feature_path="./datasets/{}/feature/{}".format(dir_purpose,object_name)
        print(feature_path)
        arr = os.listdir(feature_path)
        for file in arr:
            # Extract the values between "x", "y","z" and "r" using regular expressions
            matches = re.findall(r'x_(.*?)_y_(.*?)_z_(.*?)_r_(.*?)\.npy', file)

            if matches:
                x = float(matches[0][0])
                y = float(matches[0][1])
                z = float(matches[0][2])
                r = float(matches[0][3])
        
            else:
                print("No matches found.")

            #Create file name and check if it already exists
            f_name = "x_" + str(x) + "_y_" + str(y) + "_z_" + str(z)+"_r_"+str(r)+"_target"
            file_with_extension = f_name + ".npy"  # Assuming the file has a .txt extension
            directory = "./datasets/{}/target/{}/{}.npy".format(dir_purpose,object_name,f_name)
                    
            if os.path.exists(directory):
                print("File already exists.")
                continue
            else:
                print("File does not exist.")

            #render scene
            create_ground_truth(x,y,z,r,xlm_path,dir_purpose,f_name,object_name,scale)  


