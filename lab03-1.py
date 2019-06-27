import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x_data = [1, 2, 3]
y_data = [1, 2, 3]

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

#W = tf.Variable(tf.random_normal([1]), name="weight")
W = tf.Variable(5.0)
#b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = X * W

cost = tf.reduce_sum(tf.square(hypothesis - Y))

learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

"""직접 미분식 작성하지 않고 자동으로 경사하강법 수행
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train = optimizer.minimize(cost)
"""

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
	sess.run(update, feed_dict={X: x_data, Y: y_data})
	print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
