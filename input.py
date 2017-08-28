import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BATCH_SIZE = 32
N_FEATURES = 28


def batch_generator(batch_size=BATCH_SIZE):
    """
    returns a (data_batch, label_batch) tuple used to generate the batch data
    """
    filename_queue = tf.train.string_input_producer(['data/higgs.csv'])
    reader = tf.TextLineReader()
    _, value = reader.read(filename_queue)

    # record_defaults are the default values in case some of our columns are empty
    # This is also to tell tensorflow the format of our data (the type of the decode result)

    record_defaults = [[0.0] for _ in range(N_FEATURES + 1)]

    content = tf.decode_csv(value, record_defaults=record_defaults)

    # pack all 28 features into a tensor
    features = tf.stack(content[1:N_FEATURES])

    # assign the first column to label
    label = content[0]

    # minimum number elements in the queue after a dequeue,
    # used to ensure that the samples are sufficiently mixed

    min_after_dequeue = 10 * batch_size

    # the maximum number of elements in the queue
    capacity = 20 * batch_size

    # shuffle the data to generate BATCH_SIZE sample pairs
    data_batch, label_batch = tf.train.shuffle_batch([features, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

    return data_batch, label_batch


def generate_batches(data_batch, label_batch):
    """
    Returns a batch of data (32 samples) (features, labels)
    """
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        features, labels = sess.run([data_batch, label_batch])
        coord.request_stop()
        coord.join(threads)

    return features, labels


def main():
    data_batch, label_batch = batch_generator(batch_size=BATCH_SIZE)
    features, labels = generate_batches(data_batch, label_batch)
    print(features, labels)
    # call these two functions every time you need 32 records(BATCH_SIZE) of data
    # print("New batch!")
    # data_batch, label_batch = batch_generator()
    # generate_batches(data_batch, label_batch)
    # print("New batch!")


if __name__ == '__main__':
    main()
