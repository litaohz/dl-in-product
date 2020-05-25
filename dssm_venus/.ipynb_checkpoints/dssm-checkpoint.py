import tensorflow as tf
import os
import sys
from feature_column_dssm import build_model_columns
from tensorflow import feature_column as fc
from config import Config
import datetime
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.logging.set_verbosity(tf.logging.INFO)

train_date = sys.argv[1]
test_date = sys.argv[2]
# mode = sys.argv[3]  # train or test or export


flags = tf.app.flags
flags.DEFINE_string("model_dir", "./model", "Base directory for the model.")
flags.DEFINE_string("output_model", "./export", "Path to the training data.")
flags.DEFINE_string("train_data", "/data4/graywang/KG/CTCVR/ESMM/tfrecords/offline_ai/" + train_date + "/train/", "Directory for storing mnist data")
flags.DEFINE_string("eval_data", "/data4/graywang/KG/CTCVR/ESMM/tfrecords/offline_ai/" + train_date + "/test/", "Path to the evaluation data.")
flags.DEFINE_string("hidden_units", "512,256,128", "Comma-separated list of number of units in each hidden layer of the NN")
flags.DEFINE_integer("train_steps", 10000, "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 512, "Training batch size")
flags.DEFINE_integer("shuffle_buffer_size", 10000, "dataset shuffle buffer size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_float("dropout_rate", 0, "Drop out rate")
flags.DEFINE_integer("num_parallel_readers", 5, "number of parallel readers for training data")
flags.DEFINE_integer("save_checkpoints_steps", 5000, "Save checkpoints every this many steps")
flags.DEFINE_boolean("train", True, "Whether to train")
flags.DEFINE_boolean("predict", True, "Whether to predict")
flags.DEFINE_boolean("evaluate", True, "Whether to evaluate")
flags.DEFINE_boolean("export", True, "Whether to export model")

FLAGS = flags.FLAGS

def get_ndaysago(n, work_day=None):
    dates = []
    if work_day:
        date = time.strptime(work_day, '%Y%m%d')
        for i in range(n):
            dates.append((datetime.datetime(
                date[0], date[1], date[2]) - datetime.timedelta(days=i)).strftime('%Y%m%d'))
    else:
        for i in range(n):
            dates.append((datetime.datetime.now() - datetime.timedelta(days=i)).strftime('%Y%m%d'))
    return dates


def parse_record(example_proto):
    parsed_features = tf.parse_single_example(example_proto, features=Config.features)
    # label = parsed_features.pop('label')
    labels = parsed_features.pop('label')
    # ctr = parsed_features.pop('click_flag')
    # cvr = parsed_features.pop('sing_flag')
    return parsed_features, labels


def input_fn(data_file, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""
    dataset = tf.data.TFRecordDataset([data_file])
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)  # buffer_size should be large enough to shuffle the data. if buffer_size=1, no shuffle.

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(parse_record, num_parallel_calls=10)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=10000)
    return dataset


def train_input_fn():
    date_list = get_ndaysago(Config.train_period, str(train_date))
    tf.logging.info('Train data list: %s' % date_list)
    training_files = tf.random_shuffle(tf.train.match_filenames_once(['%s/%s/train/part-r-*' % (Config.data_path, dt) for dt in date_list]))
    return input_fn(training_files, Config.train_epochs, True, Config.batch_size)

def eval_input_fn():
    tf.logging.info('Test data list: %s' % str(test_date))
    eval_files = tf.train.match_filenames_once('%s/%s/test/part-r-*0' % (Config.data_path, test_date))
    return input_fn(eval_files, 1, False, Config.batch_size)

def build_mode(features, mode, params, columns):
    # import pdb;pdb.set_trace()
    net = fc.input_layer(features, columns)
    # Build the hidden layers, sized according to the 'hidden_units' param.
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
          net = tf.layers.dropout(net, params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))
    return net

def get_cosine_score(user_vector, item_vector):
    # query_norm = sqrt(sum(each x^2))
    pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.square(user_vector), 1))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.square(item_vector), 1))
    pooled_mul_12 = tf.reduce_sum(tf.multiply(user_vector, item_vector), 1)
    cos_scores = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="cos_scores")
    return cos_scores

def my_model(features, labels, mode, params):
    user_last_layer = build_mode(features, mode, params, params['feature_columns']['user_column'])
    item_last_layer = build_mode(features, mode, params, params['feature_columns']['item_column'])

    user_layer_norm = tf.nn.l2_normalize(tf.nn.relu(user_last_layer), axis=1, epsilon=1e-8, name="user_layer_norm")
    item_layer_norm = tf.nn.l2_normalize(tf.nn.relu(item_last_layer), axis=1, epsilon=1e-8, name="item_layer_norm")

    with tf.name_scope('Cosine_Similarity'):
        prob = tf.reduce_sum(tf.multiply(user_layer_norm, item_layer_norm), axis=1, keepdims=True, name='cos')
        # prob = tf.nn.softmax(get_cosine_score(user_last_layer, item_last_layer))
        # prob = tf.clip_by_value(prob, 1e-8, 1.0)
        output = tf.layers.dense(prob, units=1, kernel_constraint=tf.keras.constraints.NonNeg(), name="fc_loss")

        tf.summary.histogram('user_layer_norm', user_layer_norm)
        # tf.summary.scalar('item_layer_norm', tf.squeeze(item_layer_norm))
        # tf.summary.scalar('prob', prob)
        # tf.summary.scalar('output', output)

    with tf.name_scope('Loss'):
        # import pdb;pdb.set_trace()
        # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels,dtype=tf.float32), logits=prob)
        # loss = tf.reduce_sum(cross_entropy)
        print("prob:", prob)
        # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels,dtype=tf.float32), logits=prob)
        # loss = tf.reduce_sum(cross_entropy)
        # loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, dtype=tf.float32), logits=output, name="loss"))
            # loss = tf.reduce_mean(losse))
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, dtype=tf.float32), logits=output, name="scl_loss")
        loss = tf.reduce_mean(losses)
        # tf.summary.scalar('losses', losses)
        # tf.summary.scalar('loss', loss)
        # tf.summary.scalar('cross_entropy', cross_entropy)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "user_last_layer": user_layer_norm,
            "item_last_layer": item_layer_norm,
            "prob": prob,
        }
        export_outputs = {
          'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    auc = tf.metrics.auc(labels, prob)
    metrics = {'auc': auc}
    tf.summary.scalar('auc', auc[1])


    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def model_export(model):
    input_receiver = tf.estimator.export.build_parsing_serving_input_receiver_fn(Config.features)
    model.export_savedmodel(export_dir_base=Config.export_dir, serving_input_receiver_fn=input_receiver)

def input_receiver():
    user_column, item_column = build_model_columns()
    feature_spec = tf.feature_column.make_parse_example_spec(user_column + item_column)
    serialized_tf_example = tf.placeholder(dtype=tf.string,
                                           shape=[None],
                                           name='input')
    receiver_tensors = {'inputs': serialized_tf_example}
    # print(serialized_tf_example)
    print(feature_spec)
    features = tf.parse_example(serialized_tf_example, feature_spec)
    rec = tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
    print('rec:', rec)
    return rec

def read_data():
    dataset = tf.data.TFRecordDataset(tf.train.match_filenames_once('%s/part-r-*' % FLAGS.eval_data))
    dataset = dataset.map(parse_record, num_parallel_calls=FLAGS.num_parallel_readers)

    validation_iterator = dataset.make_initializable_iterator()
    data = validation_iterator.get_next()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(validation_iterator.initializer)

        with open('result/uid_'+ test_date, 'w') as f:
            try:
                while True:
                    import pdb;pdb.set_trace()
                    uid = sess.run(data[0]['uid'])
                    f.write(str(uid) + '\n')
            except tf.errors.OutOfRangeError:
                print("outOfRange")

def main():
    user_column, item_column = build_model_columns()
    feature_columns = {"user_column": user_column, "item_column": item_column}
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': feature_columns,
            'hidden_units': FLAGS.hidden_units.split(','),
            'learning_rate': FLAGS.learning_rate,
            'dropout_rate': FLAGS.dropout_rate
        },
        config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    )
    batch_size = FLAGS.batch_size
    shuffle_buffer_size = FLAGS.shuffle_buffer_size

    print("train steps:", FLAGS.train_steps, "batch_size:", batch_size)
    print("shuffle_buffer_size:", shuffle_buffer_size)

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: train_input_fn(),
        max_steps=FLAGS.train_steps
    )
    input_fn_for_eval = lambda: eval_input_fn()
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, throttle_secs=600, steps=None)

    print("before train and evaluate")
    if FLAGS.train:
        tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    print("after train and evaluate")

    # Evaluate accuracy.
    if FLAGS.evaluate:
        results = classifier.evaluate(input_fn=input_fn_for_eval)
        for key in sorted(results): print('%s: %s' % (key, results[key]))
        print("after evaluate")

    if FLAGS.predict:
        results = classifier.predict(input_fn=input_fn_for_eval)
        # import pdb;pdb.set_trace()
        with open('result/predict', 'w') as f:
            for pred in list(results):
                f.write(','.join([str(x) for x in pred['prob']]) + '\n')
        print("after predict")

    if FLAGS.export:
        print("-----------------exporting model ...------------------")


        # serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(Config.features)
        # classifier.export_savedmodel(FLAGS.output_model, serving_input_receiver_fn)
        classifier.export_savedmodel(FLAGS.output_model, input_receiver)
        # feature_spec = tf.feature_column.make_parse_example_spec(deep_columns)
        # print(feature_spec)
        # serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        # classifier.export_savedmodel(FLAGS.output_model, serving_input_receiver_fn)


    print("quit main")

if __name__ == '__main__':
    # read_data()
    main()
