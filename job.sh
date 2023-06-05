#!/bin/sh
#SBATCH --partition=p_nlp
#SBATCH --job-name=sd_coco
#SBATCH --mem=40G
#SBATCH --output=/nlp/data/rhuang99/fakenews/logs/%x.%j.out
#SBATCH --error=/nlp/data/rhuang99/fakenews/logs/%x.%j.err
#SBATCH --gpus=1
#SBATCH --nodelist=nlpgpu04                                                                                                                                                                                                                                             

srun python generate_images.py