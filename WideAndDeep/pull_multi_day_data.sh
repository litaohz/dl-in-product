#!/bin/sh
source ~/.bashrc


#dir=./tfrecords
#dir=./tfrecords08
#dir=./tfrecords0808
#dir=./tfrecords0812
#dir=./tfrecords0812v2
#dir=./tfrecords0813
#dir=./tfrecords1112
#dir=./tfrecords1201
dir=./tfrecords1207

hdfs_path=/stage/interface/ecc/u_ecc_qqmusicaudio

#tdfs -get $HDFS_IN_PATH/janusyang/qmkg/tfrecord/201904_anchor/${ftime} $dir

for delta in $( seq 1 2)
do
  ftime=`date --date="$delta days ago" +%Y%m%d`
  echo "date is $ftime"
  rm -r $dir/${ftime}
  mkdir -p $dir/${ftime}
  
  #label decided by popup total time in a day  
  #hadoop fs -Dfs.default.name=hdfs://tl-if-nn-tdw.tencent-distribute.com:54310 -Dhadoop.job.ugi=tdwadmin,usergroup -get $hdfs_path/janusyang/qmkg/tfrecord/20190802_anchor/${ftime} $dir
  #label decided by popup total time in a day  and distinct by uid->touid in a day and sample negtive sample add follow sample as positive sample 
  #hadoop fs -Dfs.default.name=hdfs://tl-if-nn-tdw.tencent-distribute.com:54310 -Dhadoop.job.ugi=tdw_janusyang:yang@0613 -get $hdfs_path/janusyang/qmkg/tfrecord/20191201_anchor/${ftime} $dir
  
  #enterntainemnt page label decided by popup total time in a day  and distinct by uid->touid in a day and sample negtive sample add follow sample as positive sample 
  hadoop fs -Dfs.default.name=hdfs://tl-if-nn-tdw.tencent-distribute.com:54310 -Dhadoop.job.ugi=tdw_janusyang:yang@0613 -get $hdfs_path/janusyang/qmkg/tfrecord/20191207_anchor/${ftime} $dir
  
  #label decided by popup total time in a day  and distinct by uid->touid in a day
  #hadoop fs -Dfs.default.name=hdfs://tl-if-nn-tdw.tencent-distribute.com:54310 -Dhadoop.job.ugi=tdwadmin,usergroup -get $hdfs_path/janusyang/qmkg/tfrecord/20190812_anchor/${ftime} $dir
  
  #label decided by popup total time in a day  and at most 5 samples to uid->touid pair in a day
  #hadoop fs -Dfs.default.name=hdfs://tl-if-nn-tdw.tencent-distribute.com:54310 -Dhadoop.job.ugi=tdwadmin,usergroup -get $hdfs_path/janusyang/qmkg/tfrecord/20190813_anchor/${ftime} $dir
  
  #label decided by popup total time in a day  and distinct by uid->touid in a day and sample negtive sample
  #hadoop fs -Dfs.default.name=hdfs://tl-if-nn-tdw.tencent-distribute.com:54310 -Dhadoop.job.ugi=tdwadmin,usergroup -get $hdfs_path/janusyang/qmkg/tfrecord/20190812v2_anchor/${ftime} $dir
  
  
  #nearby lable with follow as positive sample
  #hadoop fs -Dfs.default.name=hdfs://tl-if-nn-tdw.tencent-distribute.com:54310 -Dhadoop.job.ugi=tdw_janusyang:yang@0613 -get $hdfs_path/janusyang/qmkg/tfrecord/20191112_anchor/${ftime} $dir
  
  #label decided by popup total time in a day and  add time weight 
  #hadoop fs -Dfs.default.name=hdfs://tl-if-nn-tdw.tencent-distribute.com:54310 -Dhadoop.job.ugi=tdwadmin,usergroup -get $hdfs_path/janusyang/qmkg/tfrecord/20190808_anchor/${ftime} $dir
  
  #label decided by all entry total time 
  #hadoop fs -Dfs.default.name=hdfs://tl-if-nn-tdw.tencent-distribute.com:54310 -Dhadoop.job.ugi=tdwadmin,usergroup -get $hdfs_path/janusyang/qmkg/tfrecord/201908_anchor/${ftime} $dir
  
  # label decided by popup sample watch time not time in a day 
  #hadoop fs -Dfs.default.name=hdfs://tl-if-nn-tdw.tencent-distribute.com:54310 -Dhadoop.job.ugi=tdwadmin,usergroup -get $hdfs_path/janusyang/qmkg/tfrecord/201904_anchor/${ftime} $dir
done
#hadoop fs -Dfs.default.name=hdfs://tl-if-nn-tdw.tencent-distribute.com:54310 -Dhadoop.job.ugi=tdwadmin,usergroup -get $hdfs_path/janusyang/qmkg/tfrecord/201904_anchor/20190403 $dir &
#hadoop fs -Dfs.default.name=hdfs://tl-if-nn-tdw.tencent-distribute.com:54310 -Dhadoop.job.ugi=tdwadmin,usergroup -get $hdfs_path/janusyang/qmkg/tfrecord/201904_anchor/20190327 $dir &
