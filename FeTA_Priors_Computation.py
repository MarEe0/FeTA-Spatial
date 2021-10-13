"""
Created on June 2021

@author: Mireia Alenya

Description: Read MRI segmentations and compute centroids of structures (labels 1-7)
Save centroid's values in centroids_test.npy

"""

import logging
import os
import nibabel as nib
import sys
import numpy as np
from scipy.ndimage.measurements import center_of_mass
import matplotlib.pyplot as plt
import glob

if __name__ == '__main__':
    orig_stdout = sys.stdout


    data_path = '/priors_centroids'
    out_path = data_path

    seg_list = glob.glob(data_path + "\\*-seg.nii.gz")
    img_list = glob.glob(data_path + "\\*-img.nii.gz")
    i = 0
    # 3 = coordinates; 7 = number of labels: 80 = number of images
    labels = np.zeros((3, 7, 80))
    centroids = np.zeros((3, 7))

    for segin, imgin in zip(seg_list, img_list):
        print(segin)
        seg_image = os.path.join(data_path, segin)
        seg = nib.load(seg_image)
        seg_data = seg.get_fdata()

        label_shape = np.zeros(np.shape(seg_data))

        nlab = int(np.array(np.max(seg_data)))
        # nlab = int(np.array(nlab))

        label1 = label2 = np.zeros(np.shape(seg_data))
        label3 = np.zeros(np.shape(seg_data))
        label4 = np.zeros(np.shape(seg_data))
        label5 = np.zeros(np.shape(seg_data))
        label6 = np.zeros(np.shape(seg_data))
        label7 = np.zeros(np.shape(seg_data))

        # find centroids of label structures
        label1[np.where(seg_data == 1)] = 1  # csf
        label2[np.where(seg_data == 2)] = 2  # gm
        label3[np.where(seg_data == 3)] = 3  # wm
        label4[np.where(seg_data == 4)] = 4  # vntr
        label5[np.where(seg_data == 5)] = 5  # crb
        label6[np.where(seg_data == 6)] = 6  # dgm
        label7[np.where(seg_data == 7)] = 7  # bs

        for j in range(3):
            labels[j][0][i] = center_of_mass(label1)[j]
            labels[j][1][i] = center_of_mass(label2)[j]
            labels[j][2][i] = center_of_mass(label3)[j]
            labels[j][3][i] = center_of_mass(label4)[j]
            labels[j][4][i] = center_of_mass(label5)[j]
            labels[j][5][i] = center_of_mass(label6)[j]
            labels[j][6][i] = center_of_mass(label7)[j]

        # if i is 0:
            # print("should print imshow")
            # plt.imshow(seg_data[:, :, 100], cmap='viridis')
            # plt.scatter(labels[1][0][i], labels[0][0][i], c='coral')
            # plt.scatter(labels[1][1][i], labels[0][1][i], c='red')
            # plt.scatter(labels[1][2][i], labels[0][2][i], c='lightblue')
            # plt.scatter(labels[1][3][i], labels[0][3][i], c='navy')
            # plt.scatter(labels[1][4][i], labels[0][4][i], c='green')
            # plt.scatter(labels[1][5][i], labels[0][5][i], c='orange')
            # plt.scatter(labels[1][6][i], labels[0][6][i], c='black')
            # plt.show()
            #
            # plt.imshow(seg_data[100, :, :], cmap='viridis')
            # plt.scatter(labels[2][0][i], labels[1][0][i], c='coral')
            # plt.scatter(labels[2][1][i], labels[1][1][i], c='red')
            # plt.scatter(labels[2][2][i], labels[1][2][i], c='lightblue')
            # plt.scatter(labels[2][3][i], labels[1][3][i], c='navy')
            # plt.scatter(labels[2][4][i], labels[1][4][i], c='green')
            # plt.scatter(labels[2][5][i], labels[1][5][i], c='orange')
            # plt.scatter(labels[2][6][i], labels[1][6][i], c='black')
            # plt.show()

        i += 1

        nlab = np.array(7)
        for lab_n in range(nlab):
            for it_n in range(3):
                centroids[it_n][lab_n] = np.mean(labels[it_n][lab_n])

        np.save(os.path.join(data_path, "centroids_test"), centroids)

        print(centroids)

        # Plotting scattered points + MRI image:
        plt.scatter(centroids[2][0], centroids[0][0], c='coral')
        plt.scatter(centroids[2][1], centroids[0][1], c='red')
        plt.scatter(centroids[2][2], centroids[0][2], c='lightblue')
        plt.scatter(centroids[2][3], centroids[0][3], c='navy')
        plt.scatter(centroids[2][4], centroids[0][4], c='green')
        plt.scatter(centroids[2][5], centroids[0][5], c='orange')
        plt.scatter(centroids[2][6], centroids[0][6], c='black')

