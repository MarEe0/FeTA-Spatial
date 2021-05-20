#!/bin/bash
#SBATCH -J FTMulti
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH -o /homedtic/malenya/nnUNet/FeTA_Multilabel_3D_0/outs/job_%x-%j.out 
#SBATCH -e /homedtic/malenya/nnUNet/FeTA_Multilabel_3D_0/outs/job_%x-%j.err 

#abort on error
set -e

source ~/anaconda3/bin/activate "";
conda activate nunet;

  
export nnUNet_raw_data_base="/homedtic/malenya/nnUNet/FeTA_Multilabel_3D_0/nnUNet_raw_data_base"
export nnUNet_preprocessed="/homedtic/malenya/nnUNet/FeTA_Multilabel_3D_0/nnUNet_preprocessed"
export RESULTS_FOLDER="/homedtic/malenya/nnUNet/FeTA_Multilabel_3D_0/nnUNet_RESULTS"

cd /homedtic/malenya/nnUNet/

nnUNet_train 3d_fullres nnUNetTrainerV2 Task001_Brain 0