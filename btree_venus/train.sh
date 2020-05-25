#!/bin/bash
export JAVA_HOME="/usr/local/jdk"
export HADOOP_HOME="/data/home/hadoop-venus"
export HADOOP_HDFS_HOME="/data/home/hadoop-venus"
export LD_LIBRARY_PATH="$JAVA_HOME/jre/lib/amd64/server:/usr/local/nvidia/lib64:/usr/local/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob)
pwd
cd btree_venus
echo "clear old data"
pip uninstall tensorflow-gpu
pip install -i https://mirrors.tencent.com/pypi/simple/ tensorflow==1.13.1
old_date=`date -d "-1 month" +%Y%m%d`
mkdir -p /ceph/11027/bt-metric/$1
mkdir -p /ceph/11027/bt-checkpoint/$1
rm -rf /ceph/11027/bt-checkpoint/$old_date
rm -rf /ceph/11027/bt-export-1/$old_date
gap=$2
echo "graywang"
pwd
python bt_train_weight.py --conf=bt_train_conf.ini $1 /ceph/11027/bt-checkpoint/latest/ /ceph/11027/bt-export/$1 /ceph/11027/bt-metric/$1  $gap
mkdir -p /ceph/11027/bt-export-1/$1
cp -rf /ceph/11027/bt-export/$1/1*/* /ceph/11027/bt-export-1/$1/
rm -rf /ceph/11027/bt-export/$1
cp -r /ceph/11027/bt-checkpoint/latest/* /ceph/11027/bt-checkpoint/$1/
