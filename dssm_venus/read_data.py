"""doc string"""
import os
import tensorflow as tf
 
 
def read_hdfs_tfrecords(paths,
                        parse_spec,
                        shuffle=True,
                        batch_size=64,
                        epochs=1):
    """General input functions
 
      Args:
        path: (list) list of paths on HDFS
        parse_spec: (dict) feature parsing specification
        shuffle: (bool) whether to shuffle
        batch_size: (int) batch size
        epochs: (int) epochs
      Returns:
        (batched feature dicts, batched labels)
    """
 
    suffix = '/part-r-*'
    patterns = [p + suffix for p in paths]
    files = tf.data.Dataset.from_tensor_slices(patterns).flat_map(
        lambda p: tf.data.Dataset.list_files(p))
 
    if epochs > 1:
        files = dataset.repeat(epochs)
 
    if shuffle:
        files = files.shuffle(buffer_size=len(patterns))
 
    dataset = files.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset,
            cycle_length=15,  # 同时读取15个文件
            sloppy=True,  # 允许以不确定顺序从读取
            buffer_output_elements=1024,
            prefetch_input_elements=1024))
 
    def _map(example_proto):
        features = tf.parse_example(example_proto, parse_spec)
        labels = features.pop('label')
        return features, labels
 
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(_map, num_parallel_calls=36)
    dataset = dataset.prefetch(1)
 
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
 
    features, labels = next_batch[0], next_batch[1]
 
    return features, labels
 
 
def test_read():
    parse_spec = {
        'play': tf.FixedLenFeature((1, ), dtype=tf.float32, default_value=0.0),
        'sex': tf.VarLenFeature(dtype=tf.int64),
        'kid': tf.VarLenFeature(dtype=tf.string),
        'pub': tf.VarLenFeature(dtype=tf.string),
        'label': tf.FixedLenFeature((1, ), dtype=tf.float32, default_value=0.0)
    }
 
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
 
    path = [
        'hdfs://ss-sng-dc-v2/stage/interface/ecc/u_ecc_qqmusicaudio/linguan/qmkg/tfrecord/notyoung_aug_sample_v0/20200511/train'
    ]
 
    with tf.Session() as sess:
        features, labels = read_hdfs_tfrecords(path,
                                               parse_spec,
                                               False,
                                               batch_size=1)
        for i in range(5):
            feature, label = sess.run([features, labels])
            print(label.shape)
 
 
if __name__ == '__main__':
    test_read()
