#!/bin/bash
export JAVA_HOME="/usr/local/jdk"
export HADOOP_HOME="/data/home/hadoop-venus"
export HADOOP_HDFS_HOME="/data/home/hadoop-venus"
export LD_LIBRARY_PATH="$JAVA_HOME/jre/lib/amd64/server:/usr/local/nvidia/lib64:/usr/local/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob)
echo "litao1"
pwd
cd dssm_venus
echo "litao1"
rm -rf  /ceph/11027/dssm-anchor-20200624-checkpoint/$1
mkdir /ceph/11027/dssm-anchor-20200624-checkpoint/$1

cp -rf /ceph/11027/dssm-anchor-20200624-checkpoint/fix-checkpoint/* /ceph/11027/dssm-anchor-20200624-checkpoint/$1 
pwd
python  dssm_train.py --conf=dssm_train_conf.ini $1 
mkdir /ceph/11027/dssm-anchor-20200624-export-1/$1
cp -rf /ceph/11027/dssm-anchor-20200624-export/$1/1*/* /ceph/11027/dssm-anchor-20200624-export-1/$1/
