#!/bin/sh
source ~/.bashrc

ftime="$1"

dir=./tfrecords

rm -r $dir/${ftime}
mkdir -p $dir/${ftime}
hdfs_path=/stage/interface/ecc/u_ecc_qqmusicaudio

#tdfs -get $HDFS_IN_PATH/janusyang/qmkg/tfrecord/201904_anchor/${ftime} $dir
#hadoop fs -Dfs.default.name=hdfs://tl-if-nn-tdw.tencent-distribute.com:54310 -Dhadoop.job.ugi=tdwadmin,usergroup -get $hdfs_path/janusyang/qmkg/tfrecord/201904_anchor/${ftime} $dir

for delta in $( seq 2 10 )
do
  data_date=`date --date="$delta days ago" +%Y%m%d`
  echo $data_date
done
