# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:59:49 2020

@author: javi485
"""
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import os
import json
import re
import shutil
import ntpath
from glob import glob


if __name__ == "__main__":

    # CSF             001    CSF --> Launch
    # GrayMatter      002    GM
    # WhiteMatter     003    WM --> Launch
    # Ventricles      004    LV --> Launch
    # Cerebellum      005    CBM --> Launch
    # DeepGreyMatter  006    SGM
    # BrainStem       007    BS
        
    folder_training = "/homedtic/malenya/nnUNet/DB_Feta/Training_Labels1" # already changed the path for re-labelled images 
    folder_test = "/homedtic/malenya/nnUNet/DB_Feta/Testing"
    output_folder = "/homedtic/malenya/nnUNet/FeTA_BS/nnUNet_raw_data_base/nnUNet_raw_data/Task001_Brain/" # empty
    # Podem fer els folders: FeTA_CSF, FeTA_WM... ?? Aquí CSF, per exemple. CAL CANVIAR-HO PER CADA TISSUE
   
    maybe_mkdir_p(join(output_folder, "imagesTr"))
    maybe_mkdir_p(join(output_folder, "imagesTs"))
    maybe_mkdir_p(join(output_folder, "labelsTr"))
    maybe_mkdir_p(join(output_folder, "labelsTs"))
   
    # train
    all_train_files = []
    data_files_train = [i for i in subfiles(folder_training, suffix="_T2w.nii.gz")]  
    corresponding_seg_files = [i for i in subfiles(folder_training, suffix="BS_mask_lb1.nii.gz")] # CAL CANVIAR-HO PER CADA TISSUE --------------
    
    
    for d, s in zip(data_files_train, corresponding_seg_files):
        print(d)
        patient_identifier = re.sub("_T2w.nii.gz", "", d)
        patient_identifier = re.sub("[^0-9]","",patient_identifier) # [:-1] selects only the numbers. Removes the "2" of the T2 that appears in the name of the file.
        print(patient_identifier)
        all_train_files.append(str(patient_identifier) + "_0000.nii.gz")
        shutil.copy(d, join(output_folder, "imagesTr", patient_identifier + "_0000.nii.gz").replace("\\","/"))
        shutil.copy(s, join(output_folder, "labelsTr", patient_identifier + ".nii.gz").replace("\\","/"))
           
    # test
    all_test_files = []
    data_files_test = [i for i in subfiles(folder_test, suffix="T2w.nii.gz")]    # if i.find("GM_mask.nii.gz")==-1]  
    #print(data_files_test)
    #corresponding_seg_files_ts = [i for i in subfiles(folder_test, suffix=".nii.gz") if i.find("_T2w.nii.gz")==-1] 
    
    
    #data_files_test = [i for i in subfiles(folder_test, suffix=".nii.gz") if i.find("GM_mask.nii.gz")==-1]  
    #corresponding_seg_files_ts = [i for i in subfiles(folder_test, suffix=".nii.gz") if i.find("_T2w.nii.gz")==-1] 
    #data_files_test = [i for i in subfiles(folder_test, suffix=".nii.gz") if i.find("_1.nii.gz")==-1 ]
    #corresponding_seg_files_ts = [folder_test + '/case' + re.sub("[^0-9]", "",ntpath.basename(i))+'_1.nii.gz' for i in data_files_test]
    #corresponding_seg_files_ts = glob(folder_test+"//*_1.nii.gz")
    #for d in data_files_test:
    
    #for d, s in zip(data_files_test, corresponding_seg_files_ts):
    for m in data_files_test:
        patient_identifier = re.sub("_T2w.nii.gz", "", m)
        patient_identifier = re.sub("[^0-9]","", patient_identifier)#[:-1]
        print(patient_identifier)
        all_test_files.append(str(patient_identifier) + "_0000.nii.gz")
        shutil.copy(m, join(output_folder, "imagesTs", patient_identifier + "_0000.nii.gz").replace("\\","/"))
       
    json_dict = OrderedDict()
    json_dict['name'] = "FeTest_001"
    json_dict['description'] = "FeTA_Dataset"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "see MICCAI2021 challenge"
    json_dict['licence'] = "see MICCAI2021 challenge"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "MRI"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "BS",
    }                               # CAL CANVIAR-HO PER CADA TISSUE -------------------------
    #   "1": "CSF",
    #   "1": "GM",
    #   "1": "WM",
    #   "1": "LV",
    #   "1": "CBM",
    #   "1": "SGM"
    
    json_dict['numTraining'] = len(all_train_files)
    json_dict['numTest'] = len(all_test_files)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1][:-12], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1][:-12]} for i in
                             all_train_files]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1][:-12] for i in all_test_files]

    save_json(json_dict, os.path.join(output_folder, "dataset.json"))
