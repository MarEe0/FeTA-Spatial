#!/bin/bash
#SBATCH -J I_M
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH -o /homedtic/malenya/nnUNet/FeTA_all_Multilabel/outs/job_%x-%j.out 
#SBATCH -e /homedtic/malenya/nnUNet/FeTA_all_Multilabel/outs/job_%x-%j.err 

#abort on error
set -e

source ~/anaconda3/bin/activate "";
conda activate nunet;
  
export nnUNet_raw_data_base="/homedtic/malenya/nnUNet/FeTA_all_Multilabel/nnUNet_raw_data_base"
export nnUNet_preprocessed="/homedtic/malenya/nnUNet/FeTA_all_Multilabel/nnUNet_preprocessed"
export RESULTS_FOLDER="/homedtic/malenya/nnUNet/FeTA_all_Multilabel/nnUNet_RESULTS"

cd /homedtic/malenya/nnUNet/

nnUNet_predict -i "/homedtic/malenya/nnUNet/FeTA_all_Multilabel/nnUNet_raw_data_base/nnUNet_raw_data/Task001_Brain/imagesTs" -o "/homedtic/malenya/nnUNet/FeTA_all_Multilabel/nnUNet_prediction_results_cv" -t 001 -m 3d_fullres -tr nnUNetTrainerV2

