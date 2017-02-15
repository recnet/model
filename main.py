import tensorflow as tf
import numpy as np
import word2vec

def getVec(word, dictionary):
    idx = dictionary[word]
    totLen = len(dictionary)
    vec = np.zeros(totLen)
    vec[idx] = 1
    return vec

def createMatrix(dictionary):
    totLen = len(dictionary)
    matrix = np.identity(totLen) #remove +1 if we put UNK-thing on index 0, then everything is fine :)
    matrix[0][0] = 0 #this will be kinda "default vec" if the word does not exist (or didnt come up enough times)
    return matrix

def getIndices(sentence, dictionary):
    result = []
    for word in sentence:
        result.append(dictionary[word])
    return result

unroll_steps = 3
feature_size = 4 #this will basically be size of dictionary because its one hot vec, 50000 ish
batch_size = 1
onehotvecsize = 2 #testing purposes, look my dictionary variable below
userCount = 6 #change to 13000 later????
# sess = tf.InteractiveSession()

#Todo call function for word2vec and assign it to the variable
vocabulary = ["Hej", "po", "dig", "katt", "hund"]
_, _, dictionary, _ = word2vec.build_dataset(vocabulary) #this will be my dictionary that I will somehow create by calling func from word2vec
matrix = createMatrix(dictionary)
embeddingMatrix = tf.constant(matrix)#I should be able to give it numpy object to create embedding matrix
input = tf.placeholder(tf.int32, [1, unroll_steps]) #I want to have input as 1 row of 30 word indices so I can give it to my embedding matrix
target = tf.placeholder(tf.int32, [1, userCount]) #About 13k users, check later, just an example

lstm = tf.nn.rnn_cell.BasicLSTMCell(feature_size, forget_bias=1.0, state_is_tuple=True)#says units number in documentation. This should be right :)
initstate = lstm.zero_state(batch_size, dtype=tf.float64)
swaginputs = tf.nn.embedding_lookup(embeddingMatrix, input)

inputs = tf.unstack(swaginputs, num=unroll_steps, axis=1)
inputs = list(inputs)
outputs, state = tf.nn.rnn(lstm, inputs, initial_state=initstate)
output = outputs[-1]
weights = tf.Variable(tf.random_normal([feature_size, userCount], stddev=0.35, dtype=tf.float64), name="weights") #just example where I will have 5 neurons later
bias = tf.Variable(tf.random_normal([userCount], stddev=0.35, dtype=tf.float64), name="biases") #1 bias for the 1 layer that I will have

logits = tf.matmul(output, weights) + bias
prediction = tf.nn.softmax(logits)
print('prediction ', prediction)
error = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=logits)
cross_entropy = tf.reduce_mean(error)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Todo just create some inputvars for lstm and run, see what input it looks and shape of output, glhf Maxim.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sentence = ['Hej', 'katt', 'hund']
    indexIn = getIndices(sentence, dictionary)
    sess.run(train_step, feed_dict = {input: [indexIn], target: [[1, 0, 0, 0, 0, 0]]})

print('not sure if works or not')
