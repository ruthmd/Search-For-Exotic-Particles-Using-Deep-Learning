# SCRIPT TO BUILD AN INPUT PIPELINE
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_path = "data/"


def get_data_from_csv():

    filename_queue = tf.train.string_input_producer([data_path + "higgs.csv"])
    reader = tf.TextLineReader()
    _, row = reader.read(filename_queue)

    defaults = [[0.0] for x in range(0, 29)]
    
    a = [x for x in tf.decode_csv(row, record_defaults=defaults)]

    features = tf.unstack([x for x in a[1:]])

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # grabbing 10 examples from the CSV file.
        for iteration in range(0, 10):
            example, label = sess.run([features, a[0]])

            print(example, label)
        coord.request_stop()
        coord.join(threads)


get_data_from_csv()
