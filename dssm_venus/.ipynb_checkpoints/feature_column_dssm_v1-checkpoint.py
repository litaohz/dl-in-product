import tensorflow as tf


def build_model_columns():
    """Builds a set of wide and deep feature columns."""
    # label = tf.feature_column.numeric_column("label", dtype=tf.float32, default_value=0)
    label = tf.feature_column.numeric_column("label", dtype=tf.float32, default_value=0)
   
    user_column = []
    item_column = []

   
    # ----------------------- user meta data -----------------------------
    uage = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("age", dtype=tf.float32, default_value=0), boundaries=[1, 7, 13, 16, 19, 23, 26, 30, 35, 40, 45, 50, 55, 60, 70])
    fjifen = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("fjifen", dtype=tf.float32, default_value=0), boundaries=[1, 10, 30, 60, 100, 400, 600, 1000])

    # ------------------------ user history behavior statistical features -----------------------
    all_sum_days = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("all_sum_days", dtype=tf.float32, default_value=0), boundaries=[1, 4, 7, 10, 20, 40, 80, 100])
    and_sum_days = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("and_sum_days", dtype=tf.float32, default_value=0), boundaries=[1, 4, 8, 14, 28, 60])
    and_login_day = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("and_login_day", dtype=tf.float32, default_value=0), boundaries=[1, 5, 10, 20])
    ios_sum_days = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("ios_sum_days", dtype=tf.float32, default_value=0), boundaries=[1, 2, 10, 15, 100])
    ios_login_day = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("ios_login_day", dtype=tf.float32, default_value=0), boundaries=[1, 10, 30])
    publish_sum_days = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("publish_sum_days", dtype=tf.float32, default_value=0), boundaries=[1, 4, 10, 20, 100])
    publish_day = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("publish_day", dtype=tf.float32, default_value=0), boundaries=[1, 10, 30])
    fans_num = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("fans_num", dtype=tf.float32, default_value=0), boundaries=[1, 2, 5, 7, 10, 20, 50, 100])
    follow_num = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("follow_num", dtype=tf.float32, default_value=0), boundaries=[1, 2, 5, 8, 10, 20, 30, 100])

  
    user_bucketized_columns = [uage, fjifen, all_sum_days,
                          and_sum_days, and_login_day, ios_sum_days, ios_login_day,
                          publish_sum_days, publish_day, fans_num, follow_num]  
    for col in user_bucketized_columns:
        user_column.append(tf.feature_column.indicator_column(col))

   


    # ------------------ user meta data -----------------------
    #usex = tf.feature_column.categorical_column_with_vocabulary_list('sex', list(range(0, 3)), default_value=0, dtype=tf.int64)
    degree = tf.feature_column.categorical_column_with_vocabulary_list('degree', list(range(0, 8)), default_value=0, dtype=tf.int64)
    #money_level = tf.feature_column.categorical_column_with_vocabulary_list('money_level', list(range(0, 9)), default_value=0, dtype=tf.float32)
    #all_login_user_type = tf.feature_column.categorical_column_with_vocabulary_list('all_login_user_type', list(range(0, 4)), default_value=0, dtype=tf.float32)
    #account_source = tf.feature_column.categorical_column_with_vocabulary_list('account_source', list(range(0, 3)), default_value=0, dtype=tf.float32)

    user_categorical_column = [degree]
    for col in user_categorical_column:
        user_column.append(tf.feature_column.indicator_column(col))

    # ----------------------- hash_column (ids) ----------------------------
    
    recent_province_id = tf.feature_column.categorical_column_with_hash_bucket('recent_province_id', hash_bucket_size=34, dtype=tf.int64)
    recent_city_id = tf.feature_column.categorical_column_with_hash_bucket('recent_city_id', hash_bucket_size=1e3, dtype=tf.int64)
    
    recent_province_id_emb = tf.feature_column.embedding_column(recent_province_id, dimension=4)
    recent_city_id_emb = tf.feature_column.embedding_column(recent_city_id, dimension=8)
    
    user_column += [recent_province_id_emb, recent_city_id_emb]

    # ---------------------- item feature -----------------------
   
 # ----------------------- user meta data -----------------------------
    z_uage = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("z_age", dtype=tf.float32, default_value=0), boundaries=[1, 7, 13, 16, 19, 23, 26, 30, 35, 40, 45, 50, 55, 60, 70])
    z_fjifen = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("z_fjifen", dtype=tf.float32, default_value=0), boundaries=[1, 10, 30, 60, 100, 400, 600, 1000])

    # ------------------------ user history behavior statistical features -----------------------
    z_all_sum_days = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("z_all_sum_days", dtype=tf.float32, default_value=0), boundaries=[1, 4, 7, 10, 20, 40, 80, 100])
    z_and_sum_days = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("z_and_sum_days", dtype=tf.float32, default_value=0), boundaries=[1, 4, 8, 14, 28, 60])
    z_and_login_day = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("z_and_login_day", dtype=tf.float32, default_value=0), boundaries=[1, 5, 10, 20])
    z_ios_sum_days = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("z_ios_sum_days", dtype=tf.float32, default_value=0), boundaries=[1, 2, 10, 15, 100])
    z_ios_login_day = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("z_ios_login_day", dtype=tf.float32, default_value=0), boundaries=[1, 10, 30])
    z_publish_sum_days = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("z_publish_sum_days", dtype=tf.float32, default_value=0), boundaries=[1, 4, 10, 20, 100])
    z_publish_day = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("z_publish_day", dtype=tf.float32, default_value=0), boundaries=[1, 10, 30])
    z_fans_num = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("z_fans_num", dtype=tf.float32, default_value=0), boundaries=[1, 2, 5, 7, 10, 20, 50, 100])
    z_follow_num = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("z_follow_num", dtype=tf.float32, default_value=0), boundaries=[1, 2, 5, 8, 10, 20, 30, 100])

  
    item_bucketized_columns = [z_uage, z_fjifen, z_all_sum_days,
                          z_and_sum_days, z_and_login_day, z_ios_sum_days, z_ios_login_day,
                          z_publish_sum_days, z_publish_day, z_fans_num, z_follow_num]  
    for col in item_bucketized_columns:
        item_column.append(tf.feature_column.indicator_column(col))




    # ------------------ user meta data -----------------------
    #z_usex = tf.feature_column.categorical_column_with_vocabulary_list('z_sex', list(range(0, 3)), default_value=0, dtype=tf.int64)
    z_degree = tf.feature_column.categorical_column_with_vocabulary_list('z_degree', list(range(0, 8)), default_value=0, dtype=tf.int64)
    #z_money_level = tf.feature_column.categorical_column_with_vocabulary_list('z_money_level', list(range(0, 9)), default_value=0, dtype=tf.float32)
   
    item_categorical_column = [ z_degree]
    for col in item_categorical_column:
        item_column.append(tf.feature_column.indicator_column(col))


    
    # ----------------------- hash_column (ids) ----------------------------
    # uid_hash = tf.feature_column.categorical_column_with_hash_bucket('uid', hash_bucket_size=5e6, dtype=tf.float32)
    
    z_recent_province_id = tf.feature_column.categorical_column_with_hash_bucket('z_recent_province_id', hash_bucket_size=34, dtype=tf.int64)
    z_recent_city_id = tf.feature_column.categorical_column_with_hash_bucket('z_recent_city_id', hash_bucket_size=1e3, dtype=tf.int64)
    
    z_recent_province_id_emb = tf.feature_column.embedding_column(z_recent_province_id, dimension=4)
    z_recent_city_id_emb = tf.feature_column.embedding_column(z_recent_city_id, dimension=8)

    item_column += [z_recent_province_id_emb, z_recent_city_id_emb]

    return user_column, item_column
