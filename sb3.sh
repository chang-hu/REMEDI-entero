#!/bin/bash

ALGORITHM="PPO"
TRAIN_STEPS=2000000
LEARNING_RATE=0.002
N_ENVS=15
ADAPTATION_DAYS=240
DATA_ID="median"
MAX_BA_FLOW=10.0
GUT_DECONJ_FREQ_CO_MULTIPLIER=0.835
GUT_BIOTR_FREQ_CA_MULTIPLIER=0.2
# GUT_DECONJ_FREQ_CO_MULTIPLIER=1.0
# GUT_BIOTR_FREQ_CA_MULTIPLIER=1.0
# CONTINUE_TRAIN_SUFFIX=""



set -x
python REMEDI_model.py --algorithm $ALGORITHM \
                       --train_steps $TRAIN_STEPS \
                       --learning_rate $LEARNING_RATE \
                       --n_envs $N_ENVS \
                       --adaptation_days $ADAPTATION_DAYS \
                       --data_ID $DATA_ID \
                       --max_ba_flow $MAX_BA_FLOW \
                       --gut_deconj_freq_co_multiplier $GUT_DECONJ_FREQ_CO_MULTIPLIER \
                       --gut_biotr_freq_CA_multiplier $GUT_BIOTR_FREQ_CA_MULTIPLIER \
                       # --continue_train_suffix $CONTINUE_TRAIN_SUFFIX \
