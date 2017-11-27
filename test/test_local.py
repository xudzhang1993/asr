from ideas.local_recurrent_attention import get_local_matrix
import tensorflow as tf

p = tf.Variable([3,4])
memory = tf.Variable(tf.truncated_normal([2, 6]))
local_memory = tf.zeros([2, 2])
for i in range(2):
    local_memory[i] = memory[i][p[i] - 1: p[i] + 1]
sess = tf.Session()
sess.run(tf.global_variables_initializer())
m, lm = sess.run([memory, local_memory])
print "memory: "
print m
print "local_memory:"
print lm
tf.less()
tf.cond