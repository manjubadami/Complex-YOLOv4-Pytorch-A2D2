# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 09:56:04 2022

@author: vishw
"""
import os
from os.path import join
import glob
import numpy as np
import pandas as pd
import cv2

root_path = 'D:\Master_Thesis\Datasets\A2D2_dataset\camera_lidar_semantic_bboxes'
file_names = sorted(glob.glob(join(root_path, '*/lidar/cam_front_center/*.npz')))

def get_sample_id_list(file_names):    
    sample_id_list = []
    for file_name in file_names:    
        file_name = file_name.split('\\')        
        seq_name_ = file_name[5]        
        file_name = file_name[-1].split('.')[0]    
        file_name = file_name.split('_')    
        seq_name = file_name[0]
        uniq_id = file_name[3]
        sample_id_list.append([seq_name_, seq_name, uniq_id])
    return sample_id_list

def get_lidar(idx):
    list_of_lidar_ids_considered = [0,1,2,3,4]
    seq_name_ = idx[0]
    seq_name = idx[1]
    uniq_id = idx[2]
    file_name = seq_name + '_lidar_' + 'frontcenter_' + uniq_id + '.npz'    
    lidar_file_path = os.path.join(root_path,seq_name_,'lidar','cam_front_center',file_name) 
    
    # read the lidar data
    data = np.load(lidar_file_path)
    
    # to convert npz file into dataframe all the keys as columns.
    df= pd.DataFrame.from_dict({item: data[item] for item in data.files}, orient='index')
    df = pd.DataFrame.transpose(df)
    
    # Filter out the lidar points which are not needed.
    df = df[df['lidar_id'].isin (list_of_lidar_ids_considered)]
    #print(df)
    
    # Converting reflectance value in the range of [0 to 1]
    reflectance_df = df['reflectance'].div(255)
    
    # Converting list of points into individual point.
    split_df = pd.DataFrame(df['points'].tolist(), columns=['x', 'y', 'z'])
    
    # Concatinating the x, y, z and reflectance value of each lidar point
    df = pd.concat([split_df, reflectance_df], axis=1)
    
    # Dataframe to array
    lidarData = df.to_numpy()
    return lidarData

def get_label(idx):
    seq_name_ = idx[0]
    seq_name = idx[1]
    uniq_id = idx[2]
    file_name = seq_name + '_label3D_' + 'frontcenter_' + uniq_id + '.json'    
    label3D_file_path = os.path.join(root_path,seq_name_,'label3D','cam_front_center',file_name)
    return label3D_file_path

def get_image(idx):
    seq_name_ = idx[0]
    seq_name = idx[1]
    uniq_id = idx[2]
    file_name = seq_name + '_camera_' + 'frontcenter_' + uniq_id + '.png'    
    image_file_path = os.path.join(root_path,seq_name_,'camera','cam_front_center',file_name)
    return image_file_path

sample_id_list = get_sample_id_list(file_names)
image_file_path = get_image(sample_id_list[0])

    


                          