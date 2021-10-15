#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 17:08:47 2021

@author: Mireia Alenyà (mireia.alenya@upf.edu)
"""
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import os
import json
import re
import shutil
from glob import glob
import nibabel as nib
import numpy as np


if __name__ == "__main__":

    # CSF             1 
    # GrayMatter      2 
    # WhiteMatter     3
    # Ventricles      4 
    # Cerebellum      5
    # DeepGreyMatter  6 
    # BrainStem       7 

        
    training_folder = "./Dataset/Training" 
    testing_folder = "./Dataset/Testing"
    output_folder = "./Database_Physense/nnUNet_raw_data_base/nnUNet_raw_data/Task001_Brain/" # empty

   
    maybe_mkdir_p(join(output_folder, "imagesTr"))
    maybe_mkdir_p(join(output_folder, "imagesTs"))
    maybe_mkdir_p(join(output_folder, "labelsTr"))
    maybe_mkdir_p(join(output_folder, "labelsTs"))
   
    # train
    all_train_files = []
    data_files_train = [i for i in subfiles(training_folder, suffix="img.nii.gz")]  
    corresponding_seg_files = [i for i in subfiles(training_folder, suffix="seg.nii.gz")] 
    
    #gas = [27.9,28.2,27.4,25.5,22.6,24.9,22.8,25.2,29,27.3,27.6,25.9,27.5,26.7,23.7,23.3,22.8,28.5,29.2,25.8,26.1,20,23.7,30.4,24.2,27.8,26.5,31.1,32.5,33.4,31.4,32.3,30,28.7,32.8,22.7,23.4,26.9,24.3,27.3,34.8,23.6,22.9,27.9,24.7,23.9,28.1,27.9,31.1,33.1,29.6,21.2,30.3,33.1,27.1,26.6,28.2,29.2,34.8,31.7,33,24.4,21.7,27.8,20.9,21.8,29,31.5,27.4,20.1,22.4,25.9,27.2,23.3,29,23.2,26.9,24,29.1,26.9]

    for d, s in zip(data_files_train, corresponding_seg_files):
        print(d)
        patient_identifier = re.sub("img.nii.gz", "", d)
        patient_identifier = re.sub("[^0-9]","",patient_identifier) # [:-1] selects only the numbers. Removes the "2" of the T2 that appears in the name of the file.
        print(patient_identifier)
        all_train_files.append(str(patient_identifier) + "_0000.nii.gz")
        shutil.copy(d, join(output_folder, "imagesTr", patient_identifier + "_0000.nii.gz").replace("\\","/"))
        shutil.copy(s, join(output_folder, "labelsTr", patient_identifier + ".nii.gz").replace("\\","/"))
           
    # test
    all_test_files = []
    data_files_test = [i for i in subfiles(testing_folder, suffix="img.nii.gz")]     
    

    for m in data_files_test:
        patient_identifier = re.sub("-img.nii.gz", "", m)
        patient_identifier = re.sub("[^0-9]","", patient_identifier)
        print(patient_identifier)
        all_test_files.append(str(patient_identifier) + "_0000.nii.gz")
        shutil.copy(m, join(output_folder, "imagesTs", patient_identifier + "_0000.nii.gz").replace("\\","/"))
      

  
    
    json_dict = OrderedDict()
    json_dict['name'] = "FeTA_Physense"
    json_dict['description'] = "FeTA: Fetal Brain Annotation Challenge"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see MICCAI2021 challenge"
    json_dict['licence'] = "see MICCAI2021 challenge"
    json_dict['release'] = "2.1"
    json_dict['modality'] = {
        "0": "MRI"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "CSF",
        "2": "GM",
        "3": "WM",
        "4": "LV",
        "5": "CBM",
        "6": "SGM",
        "7": "BS"
    }    
    json_dict['numTraining'] = len(all_train_files)
    json_dict['numTest'] = len(all_test_files)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1][:-12], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1][:-12]} for i in
                             all_train_files]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1][:-12] for i in all_test_files]

    save_json(json_dict, os.path.join(output_folder, "dataset.json"))
