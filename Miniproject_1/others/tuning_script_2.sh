#!/bin/bash
# Model Tuning

# INSIGHT FROM SCRIPT 1:
## FIXED PARAMETERS: no batchnorm, 1 conv per layer, SGD, no dropout
## UNCERTAIN: average/max pooling, data augmentation
## STILL TO PLAY WITH: number and shape of features

# template is:
# ./run_sessions.py <run_idx>  <conv_by_level>  <pooling_type>  <batch_norm>  <dropout>  <features>  <optimizer>  <loss_func>  <data_aug>

# NEW BASELINE
echo "STARTING RUN 1/8"
python ./tuning_session.py  10  1  average  0  0  [16,32,64]  SGD  MSE 0

# with MAX POOLING
echo "STARTING RUN 2/8"
python ./tuning_session.py  11  1  max  0  0  [16,32,64]  SGD  MSE 0

# with DATA AUGMENTATION
echo "STARTING RUN 3/8"
python ./tuning_session.py  12  1  average  0  0  [16,32,64]  SGD  MSE 1

# FEATURE conformation 1
echo "STARTING RUN 4/8"
python ./tuning_session.py  13  1  average  0  0  [4,8,16,32]  SGD  MSE 0

# FEATURE conformation 2
echo "STARTING RUN 5/8"
python ./tuning_session.py  14  1  average  0  0  [4,8,12,16]  SGD  MSE 0

# FEATURE conformation 3
echo "STARTING RUN 6/8"
python ./tuning_session.py  15  1  average  0  0  [4,8,16]  SGD  MSE 0

# FEATURE conformation 4
echo "STARTING RUN 7/8"
python ./tuning_session.py  16  1  average  0  0  [16,32]  SGD  MSE 0

# FEATURE conformation 5
echo "STARTING RUN 8/8"
python ./tuning_session.py  17  1  average  0  0  [4,8,12,16,18]  SGD  MSE 0

echo "END OF RUN SESSION"

echo "Creating performance comparison plot..."
python ./plot_performances.py
echo "New performance plot available in current folder."