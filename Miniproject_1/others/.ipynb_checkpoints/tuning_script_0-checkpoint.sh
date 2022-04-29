#!/bin/bash
# Model Tuning

# template is:
# ./run_sessions.py <run_idx>  <conv_by_level>  <pooling_type>  <batch_norm>  <dropout>  <features>  <optimizer>  <loss_func>  <data_aug>

# BASELINE
echo "STARTING RUN 1/9"
python ./tuning_session.py  1  2  average  1  0  [16,32,64]  SGD  MSE 0

# without BATCHNORM
echo "STARTING RUN 2/9"
python ./tuning_session.py  2  2  average  0  0  [16,32,64]  SGD  MSE 0

# using MAX POOLING instead of average
echo "STARTING RUN 3/9"
python ./tuning_session.py  3  2  max  1  0  [16,32,64]  SGD  MSE 0

# performing 1 CONVOLUTION per layer instead of 2
echo "STARTING RUN 4/9"
python ./tuning_session.py  4  1  average  1  0  [16,32,64]  SGD  MSE 0

# using ADAM as optimizer
echo "STARTING RUN 5/9"
python ./tuning_session.py  5  2  average  1  0  [16,32,64]  Adam  MSE 0

# using ADAGRAD as optimizer
echo "STARTING RUN 6/9"
python ./tuning_session.py  6  2  average  1  0  [16,32,64]  Adagrad  MSE 0

# using Data augmentation
echo "STARTING RUN 7/9"
python ./tuning_session.py  7  2  average  1  0  [16,32,64]  SGD  MSE 1

# using dropout
echo "STARTING RUN 8/9"
python ./tuning_session.py  8  2  average  1  0.5  [16,32,64]  SGD  MSE 0

# adding a FEATURE layer
echo "STARTING RUN 9/9"
python ./tuning_session.py  9  2  average  1  0  [8,16,32,64]  SGD  MSE 0

echo "END OF RUN SESSION"

echo "Creating performance comparison plot for script 0..."
python ./plot_performances.py 0
echo "New performance plot available in current folder."