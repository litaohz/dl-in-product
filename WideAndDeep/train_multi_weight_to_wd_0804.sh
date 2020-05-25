#!/bin/sh
#source ~/.bashrc
##source ../../../tfenv/bin/activate
source activate /data2/timmyqiu/env

min="20190801"
max="20190819"

traindir="/data4/janusyang/KG/Anchor/WideAndDeep"

#for dir in `ls /data4/timmyqiu/qmkg/TensorFlowRec/LearningTools/Projects/Anchor/tfrecords/`
#for dir in `ls /data4/janusyang/KG/Anchor/WideAndDeep/tfrecords/`
#for dir in `ls /data4/janusyang/KG/Anchor/WideAndDeep/tfrecords08/`
#for dir in `ls /data4/janusyang/KG/Anchor/WideAndDeep/tfrecords0808/`
for dir in `ls /data4/janusyang/KG/Anchor/WideAndDeep/tfrecords0820/`
#for dir in `ls /data4/janusyang/KG/Anchor/WideAndDeep/tfrecords0812v2/`
do
  #echo $dir
  #if [[ $dir = [0-9][0-9]* ]]; then
  if [[ $dir = 20190[7-8][0-3][0-9] ]]; then
    if [ $dir -ge $min ]; then
      if [ $dir -le $max ]; then
        echo $dir
        echo `date`
        #echo "start train: $traindir/tfrecords08/$dir"
        #echo "start train: $traindir/tfrecords/$dir"
        #echo "start train: $traindir/tfrecords0808/$dir"
        echo "start train: $traindir/tfrecords0820/$dir"
        #echo "start train: $traindir/tfrecords0812v2/$dir"
        echo "-------------------------"
        #此处训练是选择最后一天的作为测试--dev参数，之前的作为训练--train参数 
        #source activate /data2/timmyqiu/env &&  python ./dw_train_weight.py --conf dw_conf_0804.ini --train $traindir/tfrecords08/$dir/train/ --dev $traindir/tfrecords08/$max/test/ 
        #source activate /data2/timmyqiu/env &&  python ./dw_train_weight.py --conf dw_conf_0804.ini --train $traindir/tfrecords/$dir/train/ --dev $traindir/tfrecords/$max/test/ 
        #source activate /data2/timmyqiu/env &&  python ./dw_train_weight.py --conf dw_conf_0804.ini --train $traindir/tfrecords0808/$dir/train/ --dev $traindir/tfrecords0808/$max/test/ 
        #source activate /data2/timmyqiu/env &&  python ./dw_train_weight.py --conf dw_conf_0804.ini --train $traindir/tfrecords0812/$dir/train/ --dev $traindir/tfrecords0812/$max/test/ 
        source activate /data2/timmyqiu/env &&  python ./dw_train_weight.py --conf dw_conf_0804.ini --train $traindir/tfrecords0820/$dir/train/ --dev $traindir/tfrecords0820/$max/test/ 
        #source activate /data2/timmyqiu/env &&  python ./dw_train_weight.py --conf dw_conf_0804.ini --train $traindir/tfrecords0812v2/$dir/train/ --dev $traindir/tfrecords0812v2/$max/test/ 
        #source activate /data2/timmyqiu/env &&  python ./dw_train_weight.py --conf dw_conf.ini --train $traindir/tfrecords08/$dir/train/ --dev $traindir/tfrecords08/$max/test/ 
        #source activate /data2/timmyqiu/env &&  python ./dw_train_weight.py --conf dw_conf.ini --train $traindir/tfrecords/$dir/train/ --dev $traindir/tfrecords/$dir/test/ #用每天的训练数据里的测试集合来测试
      fi
    fi
  fi
done
