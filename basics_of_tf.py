import tensorflow as tf

# constants
x = tf.constant(5, dtype="int32", name="x")
y = tf.constant(15, dtype="int32", name="y")
p = tf.constant(3, dtype="int32", name="p")
q = tf.constant(4, dtype="int32", name="q")

# operations
add = tf.add(x, y, name="add")
add_3 = tf.multiply(add, p, name="add_3")
div_4 = tf.divide(add, q, name="div_4")

# placeholders
p1 = tf.placeholder(tf.float32, name="p1")
p2 = tf.placeholder(tf.float32, name="p2")

# operations
A = p1 + p2
D = p1/p2

# variables
state = tf.Variable(0, name='counter')
one = tf.constant(1, name="one")
new_value = tf.add(state, one, name="addvar")
update = tf.assign(state, new_value)

# initialize variables
init = tf.global_variables_initializer()

# session that evaluates the nodes in graphs
with tf.Session() as sess:

    # init of variable
    sess.run(init)

    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

    # write graph file to view in tensorboard
    writer = tf.summary.FileWriter("basics_output", sess.graph)

    # evaluate add and div operations
    output1 = sess.run(add_3)
    output2 = sess.run(div_4)

    # feed dictionaries into the placeholders
    d1 = {p1: 10, p2: 20}
    d2 = {p1: 40, p2: 40}

    # evaluate Add and Divide operations
    output3 = sess.run(A, feed_dict=d1)
    output4 = sess.run(D, feed_dict=d2)

    # print output to console
    print(output1, output2, output3, output4)

    writer.close()
