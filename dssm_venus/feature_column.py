import tensorflow as tf


def build_model_columns():
    """Builds a set of wide and deep feature columns."""
    # label = tf.feature_column.numeric_column("label", dtype=tf.int64, default_value=0)
    click_flag = tf.feature_column.numeric_column("click_flag", dtype=tf.int64, default_value=0)
    sing_flag = tf.feature_column.numeric_column("sing_flag", dtype=tf.int64, default_value=0)
    # -------------- user profile, numeric_column -----------------
    uid_era = tf.feature_column.numeric_column('uid_era', dtype=tf.float32, shape=(9,), default_value=0.0)
    uid_genre = tf.feature_column.numeric_column('uid_genre', dtype=tf.float32, shape=(33,), default_value=0.0)
    uid_lang = tf.feature_column.numeric_column('uid_lang', dtype=tf.float32, shape=(8,), default_value=0.0)
    uid_bpm = tf.feature_column.numeric_column('uid_bpm', dtype=tf.float32, shape=(8,), default_value=0.0)
    uid_spm = tf.feature_column.numeric_column('uid_spm', dtype=tf.float32, shape=(7,), default_value=0.0)
    uid_pitch = tf.feature_column.numeric_column('uid_pitch', dtype=tf.float32, shape=(8,), default_value=0.0)
    uid_crowd = tf.feature_column.numeric_column('uid_crowd', dtype=tf.float32, shape=(8,), default_value=0.0)
    uid_emotion = tf.feature_column.numeric_column('uid_emotion', dtype=tf.float32, shape=(27,), default_value=0.0)
    uid_scene = tf.feature_column.numeric_column('uid_scene', dtype=tf.float32, shape=(16,), default_value=0.0)
    uid_festival = tf.feature_column.numeric_column('uid_festival', dtype=tf.float32, shape=(26,), default_value=0.0)
    uid_singer_gender = tf.feature_column.numeric_column('uid_singer_gender', dtype=tf.float32, shape=(7,), default_value=0.0)
    # -------------- song profile, numeric_column -----------------
    mid_age = tf.feature_column.numeric_column('mid_age', dtype=tf.float32, shape=(15,), default_value=0.0)
    mid_sex = tf.feature_column.numeric_column('mid_sex', dtype=tf.float32, shape=(3,), default_value=0.0)

    user_column = [uid_era, uid_genre, uid_lang, uid_bpm, uid_spm, uid_pitch, uid_crowd, uid_singer_gender, uid_emotion, uid_scene, uid_festival]
    item_column = [mid_age, mid_sex]

    # --------------------- song meta data ---------------------
    era = tf.feature_column.categorical_column_with_vocabulary_list('era', list(range(0, 9)), dtype=tf.int64, default_value=0)
    genre = tf.feature_column.categorical_column_with_vocabulary_list('genre', list(range(0, 33)), dtype=tf.int64, default_value=0)
    lang = tf.feature_column.categorical_column_with_vocabulary_list('lang', list(range(0, 8)), dtype=tf.int64, default_value=0)
    bpm = tf.feature_column.categorical_column_with_vocabulary_list('bpm', list(range(0, 8)), dtype=tf.int64, default_value=0)
    spm = tf.feature_column.categorical_column_with_vocabulary_list('spm', list(range(0, 7)), dtype=tf.int64, default_value=0)
    pitch = tf.feature_column.categorical_column_with_vocabulary_list('pitch', list(range(0, 8)), dtype=tf.int64, default_value=0)
    crowd = tf.feature_column.categorical_column_with_vocabulary_list('crowd', list(range(0, 8)), dtype=tf.int64, default_value=0)
    singer_gender = tf.feature_column.categorical_column_with_vocabulary_list('singer_gender', list(range(0, 7)), dtype=tf.int64, default_value=-1)
    emotion = tf.feature_column.categorical_column_with_vocabulary_list('emotion', list(range(0, 27)), dtype=tf.int64, default_value=0)
    scene = tf.feature_column.categorical_column_with_vocabulary_list('scene', list(range(0, 16)), dtype=tf.int64, default_value=0)
    festival = tf.feature_column.categorical_column_with_vocabulary_list('festival', list(range(0, 26)), dtype=tf.int64, default_value=0)

    varlen_columns = [era, genre, lang, bpm, spm, pitch, crowd, singer_gender, emotion, scene, festival]  # dimension: 157
    for col in varlen_columns:
        item_column.append(tf.feature_column.indicator_column(col))
    # item_column += [era, genre, lang, bpm, spm, pitch, crowd, singer_gender, emotion, scene, festival]  # dimension: 157

    # ----------------------- user meta data -----------------------------
    uage = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("uage", dtype=tf.int64, default_value=0), boundaries=[1, 7, 13, 16, 19, 23, 26, 30, 35, 40, 45, 50, 55, 60, 70])
    fjifen = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("fjifen", dtype=tf.int64, default_value=0), boundaries=[1, 10, 30, 60, 100, 400, 600, 1000])
    follow_avg_age = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("follow_avg_age", dtype=tf.float32, default_value=0), boundaries=[1.0, 7.0, 14.0, 18.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0])
    to_follow_avg_age = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("to_follow_avg_age", dtype=tf.float32, default_value=0.0), boundaries=[1.0, 7.0, 14.0, 18.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0])
    first_city_ratio = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("first_city_ratio", dtype=tf.float32, default_value=0.0), boundaries=[0.2, 0.4, 0.6, 0.8])
    ios_percent = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("ios_percent", dtype=tf.float32, default_value=0.0), boundaries=[0.02, 0.05, 0.07, 0.1, 0.12, 0.15, 0.18])
    pagerank_score = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("pagerank_score", dtype=tf.float32, default_value=0.0), boundaries=[0.15, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])

    # ------------------------ user history behavior statistical features -----------------------
    all_sum_days = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("all_sum_days", dtype=tf.int64, default_value=0), boundaries=[1, 4, 7, 10, 20, 40, 80, 100])
    all_sum_dacs = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("all_sum_dacs", dtype=tf.int64, default_value=0), boundaries=[1, 5, 10, 20, 30, 60, 100, 200, 300])
    all_login_day = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("all_login_day", dtype=tf.int64, default_value=0), boundaries=[1, 7, 10, 30])
    all_login_dacs = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("all_login_dacs", dtype=tf.int64, default_value=0), boundaries=[1, 2, 10, 20])
    and_sum_days = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("and_sum_days", dtype=tf.int64, default_value=0), boundaries=[1, 4, 8, 14, 28, 60])
    and_sum_dacs = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("and_sum_dacs", dtype=tf.int64, default_value=0), boundaries=[1, 6, 10, 20, 40, 90])
    and_login_day = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("and_login_day", dtype=tf.int64, default_value=0), boundaries=[1, 5, 10, 20])
    and_login_dacs = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("and_login_dacs", dtype=tf.int64, default_value=0), boundaries=[1, 5, 10, 20])
    all_kg_year = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("all_kg_year", dtype=tf.int64, default_value=0), boundaries=[1, 2, 3, 4])
    ios_sum_days = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("ios_sum_days", dtype=tf.int64, default_value=0), boundaries=[1, 2, 10, 15, 100])
    ios_sum_dacs = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("ios_sum_dacs", dtype=tf.int64, default_value=0), boundaries=[1, 10, 20])
    ios_login_day = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("ios_login_day", dtype=tf.int64, default_value=0), boundaries=[1, 10, 30])
    ios_login_dacs = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("ios_login_dacs", dtype=tf.int64, default_value=0), boundaries=[1, 10])
    publish_sum_days = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("publish_sum_days", dtype=tf.int64, default_value=0), boundaries=[1, 4, 10, 20, 100])
    publish_sum_dacs = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("publish_sum_dacs", dtype=tf.int64, default_value=0), boundaries=[1, 2, 6, 10, 20, 100])
    publish_day = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("publish_day", dtype=tf.int64, default_value=0), boundaries=[1, 10, 30])
    publish_dacs = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("publish_dacs", dtype=tf.int64, default_value=0), boundaries=[1, 10, 100])
    fans_num = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("fans_num", dtype=tf.int64, default_value=0), boundaries=[1, 2, 5, 7, 10, 20, 50, 100])
    follow_num = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("follow_num", dtype=tf.int64, default_value=0), boundaries=[1, 2, 5, 8, 10, 20, 30, 100])
    u_click_cnt = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("u_click_cnt", dtype=tf.int64, default_value=0), boundaries=[1, 2, 4, 10, 20, 100])
    uid_sing_cnt = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("uid_sing_cnt", dtype=tf.int64, default_value=0), boundaries=[1, 2, 4, 10, 20, 100])
    u_click_ratio = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("u_click_ratio", dtype=tf.float32, default_value=0.0), boundaries=[0.01, 0.02, 0.04, 0.05, 0.08, 0.1, 0.15, 0.18, 0.2])
    m_click_ratio = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("m_click_ratio", dtype=tf.float32, default_value=0.0), boundaries=[0.05, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.14, 0.16, 0.2])
    valid_sing_ratio = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("valid_sing_ratio", dtype=tf.float32, default_value=0.0), boundaries=[0.3683,0.5151,0.6153,0.6904,0.7446,0.7894,0.8253,0.859,0.8986])
    fgood_comment_user_all_ratio = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("fgood_comment_user_all_ratio", dtype=tf.float32, default_value=0.0), boundaries=[0,0.5964,0.7499,0.8333,0.8965,0.9638,0.9998])

    # ----------------------------ola-----------------------------
    ola_age = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("ola_age", dtype=tf.int64, default_value=0), boundaries=[1, 7, 13, 16, 19, 23, 26, 30, 35, 40, 45, 50, 55, 60, 70])
    ola_price = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("ola_price", dtype=tf.int64, default_value=0), boundaries=[1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000])
    ola_sex = tf.feature_column.categorical_column_with_vocabulary_list('ola_sex', list(range(0, 3)), default_value=0, dtype=tf.int64)
    ola_edu = tf.feature_column.categorical_column_with_vocabulary_list('ola_edu', list(range(10001, 10011)),  default_value=0, dtype=tf.int64)
    ola_occu = tf.feature_column.categorical_column_with_vocabulary_list('ola_occu', list(range(10037001, 10037013)), default_value=0, dtype=tf.int64)
    ola_status = tf.feature_column.categorical_column_with_vocabulary_list('ola_status',  [10038001, 10038002, 10038003, 10038004, 10038006, 10038012, 10038013, 10038051, 10038052, 10038053, 10038054, 10038055], default_value=0, dtype=tf.int64)

    # ------------------- song history data -----------------------
    m_click_cnt = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("m_click_cnt", dtype=tf.int64, default_value=0), boundaries=[1, 2, 6, 10, 20, 50, 70, 100, 200, 500, 1000, 2000, 5000])
    mid_sing_cnt = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("mid_sing_cnt", dtype=tf.int64, default_value=0), boundaries=[1, 10, 50, 100, 500, 1000, 5000])

    user_bucketized_columns = [ios_percent, pagerank_score, u_click_ratio, all_sum_days, all_sum_dacs,
                          and_sum_days, and_sum_dacs, and_login_day, and_login_dacs, all_login_day, all_login_dacs,
                          all_kg_year, ios_sum_days, ios_sum_dacs, ios_login_day, ios_login_dacs, fjifen,
                          publish_sum_days, publish_sum_dacs, publish_day, publish_dacs, fans_num, follow_num,
                          uage, follow_avg_age, to_follow_avg_age, u_click_cnt,
                          uid_sing_cnt, first_city_ratio, ola_age, ola_price]  # dimension: 247
    for col in user_bucketized_columns:
        user_column.append(tf.feature_column.indicator_column(col))

    item_bucketized_columns = [m_click_ratio, m_click_cnt, mid_sing_cnt, valid_sing_ratio, fgood_comment_user_all_ratio]
    for col in item_bucketized_columns:
        item_column.append(tf.feature_column.indicator_column(col))

    # ----------------------- song meta data -----------------------------
    is_ori_edition = tf.feature_column.categorical_column_with_vocabulary_list('is_ori_edition', list(range(0, 2)), dtype=tf.int64, default_value=0)
    is_HQ = tf.feature_column.categorical_column_with_vocabulary_list('is_HQ', list(range(0, 2)), dtype=tf.int64, default_value=0)

    item_categorical_columns = [is_ori_edition, is_HQ]
    for col in item_categorical_columns:
        item_column.append(tf.feature_column.indicator_column(col))

    # ------------------ user meta data -----------------------
    usex = tf.feature_column.categorical_column_with_vocabulary_list('usex', list(range(0, 3)), default_value=0, dtype=tf.int64)
    degree = tf.feature_column.categorical_column_with_vocabulary_list('degree', list(range(0, 8)), default_value=0, dtype=tf.int64)
    flevel = tf.feature_column.categorical_column_with_vocabulary_list('flevel', list(range(0, 15)), default_value=0, dtype=tf.int64)
    money_level = tf.feature_column.categorical_column_with_vocabulary_list('money_level', list(range(0, 9)), default_value=0, dtype=tf.int64)
    all_login_user_type = tf.feature_column.categorical_column_with_vocabulary_list('all_login_user_type', list(range(0, 4)), default_value=0, dtype=tf.int64)
    account_source = tf.feature_column.categorical_column_with_vocabulary_list('account_source', list(range(0, 3)), default_value=0, dtype=tf.int64)

    user_categorical_column = [all_login_user_type, usex, flevel, money_level, degree, account_source, ola_sex, ola_edu, ola_occu, ola_status]
    for col in user_categorical_column:
        user_column.append(tf.feature_column.indicator_column(col))

    # ----------------------- hash_column (ids) ----------------------------
    # uid_hash = tf.feature_column.categorical_column_with_hash_bucket('uid', hash_bucket_size=5e6, dtype=tf.int64)
    mid = tf.feature_column.categorical_column_with_hash_bucket('mid', hash_bucket_size=6e5, dtype=tf.int64)
    singerid = tf.feature_column.categorical_column_with_hash_bucket('singerid', hash_bucket_size=2e4, dtype=tf.int64)
    recent_province_id = tf.feature_column.categorical_column_with_hash_bucket('recent_province_id', hash_bucket_size=34, dtype=tf.int64)
    recent_city_id = tf.feature_column.categorical_column_with_hash_bucket('recent_city_id', hash_bucket_size=1e3, dtype=tf.int64)
    uid_sing_song = tf.feature_column.categorical_column_with_hash_bucket('uid_sing_song', hash_bucket_size=6e5, dtype=tf.int64)
    uid_sing_singer = tf.feature_column.categorical_column_with_hash_bucket('uid_sing_singer', hash_bucket_size=2e4, dtype=tf.int64)
    uid_search_song = tf.feature_column.categorical_column_with_hash_bucket('uid_search_song', hash_bucket_size=6e5, dtype=tf.int64)
    uid_search_singer = tf.feature_column.categorical_column_with_hash_bucket('uid_search_singer', hash_bucket_size=2e4, dtype=tf.int64)
    uid_listen_song = tf.feature_column.categorical_column_with_hash_bucket('uid_listen_song', hash_bucket_size=6e5, dtype=tf.int64)
    uid_listen_singer = tf.feature_column.categorical_column_with_hash_bucket('uid_listen_singer', hash_bucket_size=2e4, dtype=tf.int64)

    ola_brand = tf.feature_column.categorical_column_with_hash_bucket('ola_brand', hash_bucket_size=50, dtype=tf.int64)
    ola_no = tf.feature_column.categorical_column_with_hash_bucket('ola_no', hash_bucket_size=100, dtype=tf.int64)
    ola_film = tf.feature_column.categorical_column_with_hash_bucket('ola_film', hash_bucket_size=100, dtype=tf.int64)
    ola_show = tf.feature_column.categorical_column_with_hash_bucket('ola_show', hash_bucket_size=100, dtype=tf.int64)
    ola_comic = tf.feature_column.categorical_column_with_hash_bucket('ola_comic', hash_bucket_size=100, dtype=tf.int64)
    ola_game = tf.feature_column.categorical_column_with_hash_bucket('ola_game', hash_bucket_size=50, dtype=tf.int64)
    ola_games = tf.feature_column.categorical_column_with_hash_bucket('ola_games', hash_bucket_size=100, dtype=tf.int64)
    ola_school = tf.feature_column.categorical_column_with_hash_bucket('ola_school', hash_bucket_size=100, dtype=tf.int64)
    ola_resi = tf.feature_column.categorical_column_with_hash_bucket('ola_resi', hash_bucket_size=200, dtype=tf.int64)
    ola_office = tf.feature_column.categorical_column_with_hash_bucket('ola_office', hash_bucket_size=200, dtype=tf.int64)

    hash_wide_column = [singerid, uid_sing_song, recent_province_id, recent_city_id, uid_sing_singer]
    hash_wide_column += [ola_brand, ola_no, ola_film, ola_show, ola_comic, ola_game, ola_games, ola_school, ola_office]

    recent_province_id_emb = tf.feature_column.embedding_column(recent_province_id, dimension=4)
    recent_city_id_emb = tf.feature_column.embedding_column(recent_city_id, dimension=8)
    # song_history_emb = tf.feature_column.shared_embedding_columns([mid, uid_sing_song, uid_search_song, uid_listen_song], dimension=64, combiner='sum')  # it is a list
    # singer_history_emb = tf.feature_column.shared_embedding_columns([singerid, uid_sing_singer, uid_search_singer, uid_listen_singer], dimension=32, combiner='sum')

    ola_brand_emb = tf.feature_column.embedding_column(ola_brand, dimension=8)
    ola_no_emb = tf.feature_column.embedding_column(ola_no, dimension=16)
    ola_film_emb = tf.feature_column.embedding_column(ola_film, dimension=16)
    ola_show_emb = tf.feature_column.embedding_column(ola_show, dimension=16)
    ola_comic_emb = tf.feature_column.embedding_column(ola_comic, dimension=16)
    ola_game_emb = tf.feature_column.embedding_column(ola_game, dimension=16)
    ola_games_emb = tf.feature_column.embedding_column(ola_games, dimension=16)
    ola_school_emb = tf.feature_column.embedding_column(ola_school, dimension=16)
    ola_resi_emb = tf.feature_column.embedding_column(ola_resi, dimension=16)
    ola_office_emb = tf.feature_column.embedding_column(ola_office, dimension=16)

    # user_column += [song_history_emb, singer_history_emb, recent_province_id_emb, recent_city_id_emb, ola_brand_emb, ola_no_emb, ola_film_emb, ola_show_emb, ola_comic_emb, ola_game_emb, ola_games_emb, ola_school_emb, ola_resi_emb, ola_office_emb]
    user_column += [recent_province_id_emb, recent_city_id_emb, ola_brand_emb, ola_no_emb, ola_film_emb, ola_show_emb, ola_comic_emb, ola_game_emb, ola_games_emb, ola_school_emb, ola_resi_emb, ola_office_emb]

    # ---------------------- cross feature -----------------------
    # uage_usex = tf.feature_column.crossed_column([uage, usex], hash_bucket_size=30)
    # uage_usex = tf.feature_column.categorical_column_with_vocabulary_list(uage_usex, list(range(0, 45)), default_value=0, dtype=tf.int64)
    # user_column += [uage_usex]


    return user_column, item_column
