import tensorflow as tf
def linear_transform(x):
    with tf.variable_scope('linear', reuse=tf.AUTO_REUSE):
        w = tf.get_variable('w', [128, 128])
    return tf.matmul(w, x)

a = tf.Variable([1.0] * 128)
a = tf.reshape(a, [128, 1])
b = linear_transform(a)
c = linear_transform(b)
c = tf.squeeze(c)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print sess.run(c)
