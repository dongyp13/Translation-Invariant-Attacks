#!/bin/bash
#
# run_attack.sh is a script which executes the attack
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_attack.sh INPUT_DIR OUTPUT_DIR MAX_EPSILON
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_DIR - directory where adversarial images should be written
#   MAX_EPSILON - maximum allowed L_{\infty} norm of adversarial perturbation
#

INPUT_DIR=$1
OUTPUT_DIR=$2
MAX_EPSILON=$3

python attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=inception_v3.ckpt \
  --checkpoint_path_inception_v4=inception_v4.ckpt \
  --checkpoint_path_inception_resnet_v2=inception_resnet_v2_2016_08_30.ckpt \
  --checkpoint_path_resnet=resnet_v2_152.ckpt \
  --num_iter=10 \
  --momentum=1.0 \
  --prob=0.7
