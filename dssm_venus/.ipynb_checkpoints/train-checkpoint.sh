#!/bin/bash

export CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob)
python  dssm_train.py --conf=dssm_train_conf.ini  

