import tensorflow as tf
A = tf.TensorArray(tf.float32, 5)
g = tf.get_variable("g",shape=[6,2,2])
def instep(i, x):
    z = tf.get_variable("z", dtype=tf.float32, initializer=[[1.0, 0], [0, 1.0]])
    return i + 1, z * x
def step(i, x):
    global A
    v = tf.get_variable("v", dtype=tf.float32, initializer=[[1.0, 0], [0, 1.0]])
    A = A.write(i, x)
    j = tf.Variable(0, tf.int32)
    j, x = tf.while_loop(lambda j,x: j<5, instep, (j,x))
    return i + 1, g[i] * v * x
i = tf.Variable(0,dtype=tf.int32, name='i')
x = tf.Variable([[1,1],[2,2]], dtype=tf.float32, name='x')
_, y,= tf.while_loop(lambda i, x: i < 5, step, (i, x))
b = A.stack()
sess = tf.Session()
for x in tf.trainable_variables():
    print x.name
sess.run(tf.global_variables_initializer())
print sess.run([y, b])