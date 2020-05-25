#!/bin/sh
source ~/.bashrc

ftime="$1"

dir="/data4/timmyqiu/qmkg/TensorFlowRec/LearningTools/Projects/Anchor/tfrecords"

rm -r $dir/${ftime}
mkdir -p $dir/${ftime}

tdfs -get $HDFS_IN_PATH/kevinshuang/qmkg/tfrecord/201901_anchor/${ftime} $dir
