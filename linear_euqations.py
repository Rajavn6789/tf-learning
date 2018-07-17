import tensorflow as tf

x = tf.constant([[1., 2., 4]], name="x")
y = tf.constant([[15., 4., 5]], name="y")
Z = tf.Variable(tf.zeros([1, 3]), name="Z")

# Solve Y = X+Z
yy = tf.add(x, Z, name="yy")

# cost or loss is square of actual value - predicted value
cost = tf.square(y - yy)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cost)

epoch = 5000
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # write graph file to view in tensorboard
    writer = tf.summary.FileWriter("linear_output", sess.graph)

    sess.run(init)
    for i in range(epoch):
        sess.run(train)
    print('Solved Z', sess.run(Z))

writer.close()
