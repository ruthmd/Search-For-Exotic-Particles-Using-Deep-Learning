import os
import tensorflow as tf
from tqdm import tqdm
from input import batch_generator, generate_batches
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NUM_FEATURES = 28
NUM_OUTPUT_UNITS = 1

# hyperparameters
LEARNING_RATE = 0.001
NUM_HIDDEN_LAYERS = 3
NUM_NEURONS = 20
NUM_EPOCHS = 10
BATCH_SIZE = 32

NUM_BATCHES = int(1000000 / BATCH_SIZE)
LOG_DIR = './graphs'

x = tf.placeholder(shape=(BATCH_SIZE, NUM_FEATURES), dtype=tf.float32, name='input_x')
y = tf.placeholder(shape=(BATCH_SIZE), dtype=tf.float32, name='input_y')

def neural_network_model():
    '''
    initializing weights and bias for all layers in the model
    :return: hidden layer tensor, output layer tensor
    '''
    hidden_layers = dict()

    for i in range(NUM_HIDDEN_LAYERS):
        hidden_layers['hidden_layer_' + str(i + 1)] = dict()

        # for input layer
        if i == 0:
            hidden_layers['hidden_layer_' + str(i + 1)]['weights'] = tf.Variable(
                tf.random_normal([NUM_FEATURES, NUM_NEURONS]))

        # for  output layers
        elif i == NUM_HIDDEN_LAYERS - 1:
            hidden_layers['hidden_layer_' + str(i + 1)]['weights'] = tf.Variable(
                tf.random_normal([NUM_NEURONS, NUM_OUTPUT_UNITS]))

        # for hidden layers
        else:
            hidden_layers['hidden_layer_' + str(i + 1)]['weights'] = tf.Variable(
                tf.random_normal([NUM_NEURONS, NUM_NEURONS]))

        hidden_layers['hidden_layer_' + str(i + 1)]['biases'] = tf.Variable(tf.random_normal([NUM_NEURONS]))

    output_layer = dict()
    output_layer['weights'] = tf.Variable(tf.random_normal([NUM_NEURONS, NUM_OUTPUT_UNITS]))
    output_layer['biases'] = tf.Variable(tf.random_normal([NUM_OUTPUT_UNITS]))

    return hidden_layers, output_layer


def forward_propagate(hidden_layers, output_layer):

    weights = list()
    biases = list()

    for i in range(0, NUM_HIDDEN_LAYERS):
        weights.append(hidden_layers['hidden_layer_' + str(i + 1)]['weights'])
        biases.append(hidden_layers['hidden_layer_' + str(i + 1)]['biases'])

    weights.append(output_layer['weights'])
    biases.append(output_layer['biases'])

    for i in range(0, NUM_HIDDEN_LAYERS + 1):
        if i == 0:
            l_out = tf.add(tf.matmul(x, weights[i]), biases[i])
        else:
            l_out = tf.add(tf.matmul(l_out, weights[i]), biases[i])
        if i == NUM_HIDDEN_LAYERS:
            l_out = tf.nn.softmax(l_out)
        else:
            l_out = tf.nn.relu(l_out)

    l_out = tf.transpose(l_out)
    return l_out


def train():
    '''

    '''
    hidden_layers, output_layer = neural_network_model()
    pred = forward_propagate(hidden_layers, output_layer)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    return loss, optimizer


if __name__ == '__main__':
    loss, optimizer = train()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        data_batch, label_batch = batch_generator(batch_size=BATCH_SIZE)
        features, labels = generate_batches(data_batch, label_batch)

        writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

        cost = 0

        for i in tqdm(range(0, NUM_EPOCHS)):
            for j in range(0, NUM_BATCHES):
                data_batch, label_batch = batch_generator(batch_size=BATCH_SIZE)
                features, labels = generate_batches(data_batch, label_batch)
                l, _  = sess.run([loss, optimizer], feed_dict={x:features, y:labels})
                cost+=l
                print("Batch Loss : {}".format(l))
            printf("Loss in epoch {} : {}".format(i+1, cost))

        writer.close()
