import os
import tensorflow as tf
from tqdm import tqdm
from input import batch_generator, generate_batches
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAINING_FILE = 'data/train.csv'
TESTING_FILE = 'data/test.csv'
VALIDATION_FILE = 'data/valid.csv'

NUM_FEATURES = 28
NUM_OUTPUT_UNITS = 1

# hyperparameters
LEARNING_RATE = 0.03
NUM_HIDDEN_LAYERS = 3
NUM_NEURONS = 5
NUM_EPOCHS = 10
BATCH_SIZE = 256

NUM_BATCHES = int(660000/ BATCH_SIZE)
LOG_DIR = './graphs'

x = tf.placeholder(shape=(BATCH_SIZE, NUM_FEATURES), dtype=tf.float32, name='input_x')
y = tf.placeholder(shape=(BATCH_SIZE), dtype=tf.float32, name='input_y')

def neural_network_model():
    '''
    initializing weights and bias for all layers in the model
    return hidden layer tensor, output layer tensor
    '''
    hidden_layers = dict()

    for i in range(NUM_HIDDEN_LAYERS):
        hidden_layers['hidden_layer_' + str(i + 1)] = dict()

        # first hidden layer
        if i == 0:
            hidden_layers['hidden_layer_' + str(i + 1)]['weights'] = tf.Variable(tf.random_normal([NUM_FEATURES, NUM_NEURONS]))

        # last hidden layers
        elif i == NUM_HIDDEN_LAYERS - 1:
            hidden_layers['hidden_layer_' + str(i + 1)]['weights'] = tf.Variable(tf.random_normal([NUM_NEURONS, NUM_OUTPUT_UNITS]))

        # intermediate hidden layers
        else:
            hidden_layers['hidden_layer_' + str(i + 1)]['weights'] = tf.Variable(tf.random_normal([NUM_NEURONS, NUM_NEURONS]))

        hidden_layers['hidden_layer_' + str(i + 1)]['biases'] = tf.Variable(tf.random_normal([NUM_NEURONS]))

    output_layer = dict()
    output_layer['weights'] = tf.Variable(tf.random_normal([NUM_NEURONS, NUM_OUTPUT_UNITS]))
    output_layer['biases'] = tf.Variable(tf.random_normal([NUM_OUTPUT_UNITS]))

    return hidden_layers, output_layer


def forward_propagate(hidden_layers, output_layer):
    '''
    defines a computation to compute the output of the neural net
    using hidden_layers and output_layer returned from neural_network_model.

    return l_out which is the prediction as a tensor.
    '''
    weights = list()
    biases = list()

    for i in range(0, NUM_HIDDEN_LAYERS):
        weights.append(hidden_layers['hidden_layer_' + str(i + 1)]['weights'])
        biases.append(hidden_layers['hidden_layer_' + str(i + 1)]['biases'])

    weights.append(output_layer['weights'])
    biases.append(output_layer['biases'])

    for i in range(0, NUM_HIDDEN_LAYERS + 1):
        if i == 0:
            # first hidden layer
            l_out = tf.add(tf.matmul(x, weights[i]), biases[i])
        else:
            # all other layers
            l_out = tf.add(tf.matmul(l_out, weights[i]), biases[i])
        if i == NUM_HIDDEN_LAYERS:
            # last layer - softmax-ed
            l_out = tf.nn.softmax(l_out)
        else:
            # other layers = relu-ed
            l_out = tf.nn.relu(l_out)

    l_out = tf.transpose(l_out)
    return l_out


def train():
    '''
    defines a model using neural_network_model() and defines a prediction using
    forward_propagate() and defines a graph to compute loss using
    softmax_cross_entropy_with_logits and optimizes it using AdamOptimizer.

    returns loss,optimizer as tensors.
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

        data_batch, label_batch = batch_generator(TRAINING_FILE, batch_size=BATCH_SIZE)
        features, labels = generate_batches(data_batch, label_batch)

        writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

        cost = 0

        for i in tqdm(range(0, NUM_EPOCHS)):
            cost = 0
            for j in range(0, NUM_BATCHES):
                data_batch, label_batch = batch_generator(TRAINING_FILE, batch_size=BATCH_SIZE)
                features, labels = generate_batches(data_batch, label_batch)
                l, _  = sess.run([loss, optimizer], feed_dict={x:features, y:labels})
                cost+=l
                #print("Batch {} Loss : {}".format(j+1, l))
            print("Epoch {} Loss : {}".format(i+1, cost))

        writer.close()
