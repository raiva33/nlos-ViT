import os

import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
from model import Rec_Transformer
from utils.data_utils import get_loader
import time 

def load_model(load_model_dir, model):
    loaded_dict = torch.load(load_model_dir)
    model_dict = model.state_dict()
    loaded_dict = {k: v for k, v in loaded_dict.items() if k in model_dict}
    model_dict.update(loaded_dict)
    model.load_state_dict(model_dict)


def setup(load_model_dir,input_size, output_size):
    model = Rec_Transformer(input_size=input_size, rec_size=output_size)
    load_model(load_model_dir, model)
    return model

def predict(color, input_size, output_size, model,test_dir,rec_dir_obj):
    test_in = np.load(test_dir)
    print(test_in.shape)
    if color:
        b, g, r = cv2.split(test_in)
        test_in = np.dstack((r, g, b))
        test_out = np.zeros((output_size, output_size, 3))
        for c in range(3):
            test_in_one_channel = test_in[:, :, c]
            test_in_one_channel = np.reshape(test_in_one_channel,(1, 1, input_size, input_size))
            test_in_one_channel = (test_in_one_channel - test_in_one_channel.mean()) / test_in_one_channel.std()
            test_in_one_channel.astype(float)
            test_in_one_channel = torch.from_numpy(test_in_one_channel)
            test_in_one_channel.cuda()
            test_out_one_channel = model(test_in_one_channel)
            test_out_one_channel = test_out_one_channel.to('cpu').detach().numpy().copy()
            test_out_one_channel = np.reshape(test_out_one_channel,(output_size, output_size))
            test_out[:, :, c] = test_out_one_channel
    else:
        test_in=np.reshape(test_in,(1, 1, input_size, input_size))
        print(test_in.shape)
        test_in=(test_in - test_in.mean()) / test_in.std()
        test_in.astype(float)
        test_in = torch.from_numpy(test_in)
        test_in.cuda()
        test_out = model(test_in)
        test_out = test_out.to('cpu').detach().numpy().copy()
    test_out = np.squeeze(test_out)

    #plt.imshow(test_out)
    #plt.savefig(rec_dir)
    cv2.imwrite(rec_dir_obj,cv2.normalize(test_out, None, 0, 255, cv2.NORM_MINMAX))


def main():
    
    input_size=512
    output_size=256
    #test_dir="./presentation/matlab/rescale/x_0.0_y_0.0_z_0.5_r_0.0.npy"#'./datasets/validation/feature/'
    #list_objects=os.listdir(test_dir)#list_objects=["triangle_hole","Z","square_hole"]
    #print(test_dir,list_objects)
    rec_dir='./presentation/matlab/rescale'

    load_model_dir = './checkpoints/test_final_hole_bo_ssh.pth'
    model = setup(load_model_dir,input_size,output_size)
    model.cuda()

    time_sta = time.perf_counter()
    predict(False, input_size, output_size, model, "./presentation/matlab/rescale/x_0.0_y_0.0_z_0.5_r_0.0.npy", './presentation/matlab/rescale/predict.png')
    time_end = time.perf_counter()
    print(time_end- time_sta)
    
if __name__ == "__main__":
    main()
