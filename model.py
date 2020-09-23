import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

x = tf.placeholder(tf.float32, shape=[None, 40, 120, 3])

y_ = tf.placeholder(tf.float32, shape=[None, 1])

x_copy = x

#conv 1
W1 =tf.Variable(tf.truncated_normal([4, 4, 3, 24], stddev=0.1))
b1 =tf.Variable(tf.constant(0.1, shape=[24]))

h1 = tf.nn.relu(conv2d(x_copy, W1, 2) + b1)

#conv 2
W2 =tf.Variable(tf.truncated_normal([4, 4, 24, 36], stddev=0.1))
b2 =tf.Variable(tf.constant(0.1, shape=[36]))

h2 = tf.nn.relu(conv2d(h1, W2, 2) + b2)

#conv 3
W3 =tf.Variable(tf.truncated_normal([3, 3, 36, 48], stddev=0.1))
b3 =tf.Variable(tf.constant(0.1, shape=[48]))

h3 = tf.nn.relu(conv2d(h2, W3, 1) + b3)

#conv 4 
W4 =tf.Variable(tf.truncated_normal([3, 3, 48, 64], stddev=0.1))
b4 =tf.Variable(tf.constant(0.1, shape=[64]))

h4 = tf.nn.relu(conv2d(h3, W4, 1) + b4)

#conv 5
W5 =tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
b5 =tf.Variable(tf.constant(0.1, shape=[64]))

h5 = tf.nn.relu(conv2d(h4, W5, 1) + b5)

#conv 6
W6 =tf.Variable(tf.truncated_normal([2, 2, 64, 64], stddev=0.1))
b6 =tf.Variable(tf.constant(0.1, shape=[64]))

h6 = tf.nn.relu(conv2d(h5, W6, 1) + b6)

#FCL 1
W_fc1 =tf.Variable(tf.truncated_normal([1344, 1500], stddev=0.1))
b_fc1 =tf.Variable(tf.constant(0.1, shape=[1500]))

h6_flat = tf.reshape(h6, [-1, 1344])
h_fc1 = tf.nn.relu(tf.matmul(h6_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#FCL 2
W_fc2 =tf.Variable(tf.truncated_normal([1500, 100], stddev=0.1))
b_fc2 =tf.Variable(tf.constant(0.1, shape=[100]))

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

#FCL 3
W_fc3 =tf.Variable(tf.truncated_normal([100, 50], stddev=0.1))
b_fc3 =tf.Variable(tf.constant(0.1, shape=[50]))

h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

#FCL 3
W_fc4 =tf.Variable(tf.truncated_normal([50, 10], stddev=0.1))
b_fc4 =tf.Variable(tf.constant(0.1, shape=[10]))

h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)

h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

#Output
W_fc5 =tf.Variable(tf.truncated_normal([10, 1], stddev=0.1))
b_fc5 =tf.Variable(tf.constant(0.1, shape=[1]))

y = tf.multiply(tf.atan(tf.matmul(h_fc4_drop, W_fc5) + b_fc5), 2) #scale the atan output for angles