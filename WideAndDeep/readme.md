


1 先利用spec_bucket.py脚本生成连续特征的分桶的boundary
先利用 python spec_bucket.py --conf conf.ini来生成一些特征的分桶的边界值
特征原始配置文件名的修改在配置文件conf.ini的[input]这个section， 这里注意特征类型要跟 /tftools/FCGen下的GetFeatureSpec的类型一致，读取文
特征原始配置文件只要填充feature column的名字 类型等信息
 运行脚本脚本后会得到一份特征的分桶文件xxx_bucket.ini

 2 再运行train_multi_weight.sh来训练模型，这里会调用模型训练脚本dw_train_weight.py 注意修改路径参数

/tftools/FCGen下的GetFeatureSpec读入数据时涉及的特征数据类型：
ftype = numeric
ftype = bucketized
ftype = cat_hash
ftype = cat_hash_self
ftype = cat_vocab
ftype = cat_id
ftype = cross
ftype = shared_embedding
ftype = embedding

201907开始 dw_train_weight.py是用来训练wide and deep，搭配 janusyang_anchor_spec_bucket_v2.ini，dw_conf.ini

dw_train_weight_v3.py训练LR也就是说只有wide部分(分桶的边界配置是popup_live_lr_spec_v4_bucket.ini这个系列,dw_conf_v4.ini 数据是20190612-20190614这3天)

python ./dw_train_weight.py --conf dw_conf.ini --train tfrecords/2019061[2-3]/train/ --dev tfrecords/20190614/train/

从HDFS拉数据前，先用下面的命令看看，有多少天的数据，20190721去导最近20天的训练数据，结果只有19和20号有数据
hadoop fs -Dfs.default.name=hdfs://tl-if-nn-tdw.tencent-distribute.com:54310 -Dhadoop.job.ugi=tdwadmin,usergroup -ls  /stage/interface/ecc/u_ecc_qqmusicaudio/janusyang/qmkg/tfrecord/201904_anchor/
