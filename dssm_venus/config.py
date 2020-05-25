import tensorflow as tf


class Config(object):
    train_epochs = 1
    batch_size = 512
    hidden_units = [128, 64, 32]
    train_period = 1  # train data: 1 day
    model_dir = './model'  # path to save model
    model_type = 'wide_deep'  # 'wide': only wide, 'deep': only deep, 'wide_deep': wide and deep
    export_dir = './export/'  # path to save export model
    wide_learning_rate = 0.01
    deep_learning_rate = 0.01
    data_path = '/data4/graywang/KG/CTCVR/ESMM/tfrecords/offline_ai/'

    features = {
        'label': tf.FixedLenFeature([1], tf.float32, 0),
        'fjifen': tf.FixedLenFeature([1], tf.float32, 0),
        'all_sum_days': tf.FixedLenFeature([], tf.float32, 0),
        'and_sum_days': tf.FixedLenFeature([], tf.float32, 0),
        'and_login_day': tf.FixedLenFeature([], tf.float32, 0),
        'ios_sum_days': tf.FixedLenFeature([], tf.float32, 0),
        'ios_login_day': tf.FixedLenFeature([], tf.float32, 0),
        'publish_sum_days': tf.FixedLenFeature([], tf.float32, 0),
        'fans_num': tf.FixedLenFeature([], tf.float32, 0),
        'follow_num': tf.FixedLenFeature([], tf.float32, 0),
        'age': tf.FixedLenFeature([], tf.float32, 0),
        'sex': tf.FixedLenFeature([], tf.int64, 0),
        'degree': tf.FixedLenFeature([], tf.int64, 0),
        'money_level': tf.FixedLenFeature([], tf.float32, 0),

        'recent_province_id': tf.FixedLenFeature([], tf.int64, 0),
        'recent_city_id': tf.FixedLenFeature([], tf.int64, 0),
        'publish_day': tf.FixedLenFeature([], tf.float32, 0),

        'z_fjifen': tf.FixedLenFeature([1], tf.float32, 0),
        'z_all_sum_days': tf.FixedLenFeature([], tf.float32, 0),
        'z_and_sum_days': tf.FixedLenFeature([], tf.float32, 0),
        'z_and_login_day': tf.FixedLenFeature([], tf.float32, 0),
        'z_ios_sum_days': tf.FixedLenFeature([], tf.float32, 0),
        'z_ios_login_day': tf.FixedLenFeature([], tf.float32, 0),
        'z_publish_sum_days': tf.FixedLenFeature([], tf.float32, 0),
        'z_fans_num': tf.FixedLenFeature([], tf.float32, 0),
        'z_follow_num': tf.FixedLenFeature([], tf.float32, 0),
        'z_age': tf.FixedLenFeature([], tf.float32, 0),
        'z_sex': tf.FixedLenFeature([], tf.int64, 0),
        'z_degree': tf.FixedLenFeature([], tf.int64, 0),
        'z_money_level': tf.FixedLenFeature([], tf.float32, 0),
        'z_recent_province_id': tf.FixedLenFeature([], tf.int64, 0),
        'z_recent_city_id': tf.FixedLenFeature([], tf.int64, 0),
        'z_publish_day': tf.FixedLenFeature([], tf.float32, 0)
    }
