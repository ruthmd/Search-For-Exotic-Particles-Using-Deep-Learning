import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

NUM_HIDDEN_LAYERS = 3
NUM_NEURONS = 20
NUM_FEATURES = 28
NUM_OUTPUT_UNITS = 1

NUM_EPOCHS = 10
BATCH_SIZE = 128

LOG_DIR = './graphs'

# function to create a model
def neural_network_model():

    hidden_layers = dict()

    for i in range(NUM_HIDDEN_LAYERS):
        hidden_layers['hidden_layer_' + str(i+1)] = dict()

        if(i==0):
            hidden_layers['hidden_layer_'+ str(i+1)]['weights'] = tf.Variable(tf.random_normal([NUM_FEATURES, NUM_NEURONS]))
        elif(i==NUM_HIDDEN_LAYERS-1):
            hidden_layers['hidden_layer_'+ str(i+1)]['weights'] = tf.Variable(tf.random_normal([NUM_NEURONS, NUM_OUTPUT_UNITS]))
        else:
            hidden_layers['hidden_layer_'+ str(i+1)]['weights'] = tf.Variable(tf.random_normal([NUM_NEURONS, NUM_NEURONS]))

        hidden_layers['hidden_layer_'+ str(i+1)]['biases'] = tf.Variable(tf.random_normal([NUM_NEURONS]))

    output_layer = dict()
    output_layer['weights'] = tf.Variable(tf.random_normal([NUM_NEURONS, NUM_OUTPUT_UNITS]))
    output_layer['biases'] = tf.Variable(tf.random_normal([NUM_OUTPUT_UNITS]))

    return hidden_layers, output_layer


def forward_propagate(hidden_layers, output_layer):
    x = tf.placeholder(shape=(BATCH_SIZE, NUM_FEATURES), dtype=tf.float32, name='input_x')
    weights = list()
    biases = list()

    for i in range(0,NUM_HIDDEN_LAYERS):
        weights.append(hidden_layers['hidden_layer_'+str(i+1)]['weights'])
        biases.append(hidden_layers['hidden_layer_'+str(i+1)]['biases'])

    weights.append(output_layer['weights'])
    biases.append(output_layer['biases'])

    for i in range(0, NUM_HIDDEN_LAYERS+1):
        if i==0:
            l_out = tf.add(tf.matmul(x, weights[i]), biases[i])
        else:
            l_out = tf.add(tf.matmul(l_out, weights[i]), biases[i])
        if i==NUM_HIDDEN_LAYERS:
            l_out = tf.nn.softmax(l_out)
        else:
            l_out = tf.nn.relu(l_out)

    return l_out

def train():
    pass

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        hidden_layers, output_layer = neural_network_model()
        output = forward_propagate(hidden_layers, output_layer)
        writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        writer.close()
