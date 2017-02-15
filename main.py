import tensorflow as tf
import numpy as np

def getVec(word, dictionary):
    idx = dictionary[word]
    totLen = len(dictionary)
    vec = np.zeros(totLen)
    vec[idx] = 1
    return vec


def createMatrix(dictionary):
    totLen = len(dictionary)
    matrix = np.identity(totLen+1) #remove +1 if we put UNK-thing on index 0, then everything is fine :)
    matrix[0][0] = 0 #this will be kinda "default vec" if the word does not exist (or didnt come up enough times)
    return matrix

#Assumption: we will create a file where a sentence consists of numbers of the indices of the words. Same for users?

unroll_steps = 3
feature_size = 4 #this will basically be size of dictionary because its one hot vec, 50000 ish
batch_size = 1
onehotvecsize = 2 #testing purposes, look my dictionary variable below
userCount = 2 #change to 13000 later????

# sess = tf.InteractiveSession()

#Todo call function for word2vec and assign it to the variable
dictionary = {"a": '1', "c": '2', "z": '3'} #this will be my dictionary that I will somehow create by calling func from word2vec
matrix = createMatrix(dictionary)
embeddingMatrix = tf.constant(matrix)#I should be able to give it numpy object to create embedding matrix
# ids = tf.constant([1,2,0]) #testing embeding matrix
# sess.run(tf.global_variables_initializer())
#
# print(tf.nn.embedding_lookup(embeddingMatrix, ids).eval())

input = tf.placeholder(tf.int32, [1, unroll_steps]) #I want to have input as 1 row of 30 word indices so I can give it to my embedding matrix
target = tf.placeholder(tf.int32, [1, userCount]) #About 13k users, check later, just an example

lstm = tf.nn.rnn_cell.BasicLSTMCell(feature_size, forget_bias=1.0, state_is_tuple=True)#says units number in documentation. This should be right :)
# initstate = tf.zeros([batch_size, feature_size], dtype=tf.float64) #init state of lstm, might need to think here unless I am right baby
initstate = lstm.zero_state(batch_size, dtype=tf.float64)
print('my magical input is ', input.get_shape())
print('my awesome embedmatrix is ', embeddingMatrix.get_shape())
swaginputs = tf.nn.embedding_lookup(embeddingMatrix, input)
print('inputs after lookup shape', swaginputs.get_shape())
inputs = tf.unstack(swaginputs, num=unroll_steps, axis=1)
print('len of inputs after unstacking ', len(inputs))
print('inputs after unstacking ', inputs[0].get_shape())
print('object type inputs ', type(inputs))
print('fucking list fuck ', inputs)
inputs = list(inputs)

outputs, state = tf.nn.rnn(lstm, inputs, initial_state=initstate)
print('Length of outputs list', len(outputs))
print('Dimensions of one of the outputs', outputs[0].get_shape())
output = tf.reshape(tf.concat_v2(outputs, 1), [-1, feature_size])
print('Dimension of output after reshaping', output.get_shape())
weights = tf.Variable(tf.random_normal([feature_size, userCount], stddev=0.35, dtype=tf.float64), name="weights") #just example where I will have 5 neurons later
bias = tf.Variable(tf.random_normal([userCount], stddev=0.35, dtype=tf.float64), name="biases") #1 bias for the 1 layer that I will have

logits = tf.matmul(output, weights) + bias
print('shape of logits', logits.get_shape())

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=logits))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Todo just create some inputvars for lstm and run, see what input it looks and shape of output, glhf Maxim.

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    #Add a for loop later on that iterates through the data, then map indexes for users to fancy vector
    sess.run(train_step, feed_dict = {input: [1, 2, 3], target: [1, 1]})

print('not sure if works or not')
