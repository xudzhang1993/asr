import tensorflow as tf

a = tf.Variable(1)
b = tf.get_variable("b", [2, 2])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print sess.run(b[a])