#!/bin/sh
#source ~/.bashrc
##source ../../../tfenv/bin/activate
source activate /data2/timmyqiu/env

min="20191127"
max="20191207"

traindir="/data4/janusyang/KG/Anchor/WideAndDeep"

#for dir in `ls /data4/timmyqiu/qmkg/TensorFlowRec/LearningTools/Projects/Anchor/tfrecords/`
#for dir in `ls /data4/janusyang/KG/Anchor/WideAndDeep/tfrecords/`
#for dir in `ls /data4/janusyang/KG/Anchor/WideAndDeep/tfrecords08/`
#for dir in `ls /data4/janusyang/KG/Anchor/WideAndDeep/tfrecords1112/`
for dir in `ls /data4/janusyang/KG/Anchor/WideAndDeep/tfrecords1207/`
#for dir in `ls /data4/janusyang/KG/Anchor/WideAndDeep/tfrecords0812v2/`
do
  #echo $dir
  #if [[ $dir = [0-9][0-9]* ]]; then
  if [[ $dir = 20191[1-2][0-3][0-9] ]]; then
    if [ $dir -ge $min ]; then
      if [ $dir -le $max ]; then
        echo $dir
        echo `date`
        #echo "start train: $traindir/tfrecords08/$dir"
        #echo "start train: $traindir/tfrecords/$dir"
        #echo "start train: $traindir/tfrecords0808/$dir"
        echo "start train: $traindir/tfrecords1112/$dir"
        #echo "start train: $traindir/tfrecords0812v2/$dir"
        echo "-------------------------"
        #此处训练是选择最后一天的作为测试--dev参数，之前的作为训练--train参数 
        source activate /data2/timmyqiu/env &&  python ./dw_train_weight.py --conf dw_conf_1207.ini --train $traindir/tfrecords1207/$dir/train/ --dev $traindir/tfrecords1207/$max/test/ 
      fi
    fi
  fi
done
