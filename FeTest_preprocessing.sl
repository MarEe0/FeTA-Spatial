#!/bin/bash
#SBATCH -J P_M
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -o /homedtic/malenya/nnUNet/FeTA_all_Multilabel/outs/job_%x-%j.out 
#SBATCH -e /homedtic/malenya/nnUNet/FeTA_all_Multilabel/outs/job_%x-%j.err 

#abort on error
set -e

source ~/anaconda3/bin/activate "";
conda activate nunet;

export nnUNet_raw_data_base="/homedtic/malenya/nnUNet/FeTA_all_Multilabel/nnUNet_raw_data_base"
export nnUNet_preprocessed="/homedtic/malenya/nnUNet/FeTA_all_Multilabel/nnUNet_preprocessed"
export nnUNet_RESULTS="/homedtic/malenya/nnUNet/FeTA_all_Multilabel/nnUNet_RESULTS"

cd /homedtic/malenya/nnUNet/

nnUNet_plan_and_preprocess -t 001 --verify_dataset_integrity

#nnUNet_plan_and_preprocess -t 001 -pl3d ExperimentPlanner3D_v21 -pl2d None
