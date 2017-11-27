import tensorflow as tf
import numpy as np
from tensorflow.python.ops.array_ops import shape
from tensorflow.python.ops.rnn import _rnn_step
from tensorflow.python.ops.control_flow_ops import while_loop


class RnnConvKernel(object):
    def __init__(self, length, cell):
        self.length = length
        self.cell = cell

    def step(self, input_seg, batch_size):
        cell = self.cell
        state = cell.zero_state(batch_size, tf.float32)
        output = state
        i = 0
        while i < self.length:
            output, state = cell(input_seg[i], state)
            i += 1
        return output

    # def conv(self, inputs, step):
    #     length = len(inputs)
    #     outputs = []
    #     i = 0
    #     while i < length - self.length + 1:
    #         output, _ = self.step(inputs[i: i + self.length])
    #         outputs.append(output)
    #         i += step
    #     return outputs

def dynamic_rnn_conv(inputs, seqlength, kernel, step):

    max_time = shape(inputs)[0]
    batch_size = shape(inputs)[1]
    t = 0
    outputs = tf.TensorArray(tf.float32, max_time)
    pad = (kernel.length - 1) / 2
    inputs = tf.pad(inputs, [[pad, pad], [0, 0], [0, 0]])
    def conv(t, outputs):
        input_seg = inputs[t: t + kernel.length]
        outputs = outputs.write(t, kernel.step(input_seg, batch_size))
        t += step
        return t, outputs

    _, outputs = while_loop(lambda t, _: t < max_time, conv, (t, outputs))
    return outputs.stack()


def test1():
    with tf.Session() as sess:
        cell = tf.nn.rnn_cell.BasicRNNCell(3)
        input = tf.placeholder(tf.float32, (10,3,3))
        inputs = tf.split(input,)
        kernel = RnnConvKernel(2, cell)
        outputs = kernel.conv(inputs, 1)
        sess.run(tf.global_variables_initializer())
        for v in tf.trainable_variables():
            print v.name
            print sess.run(tf.assign(v, tf.ones(shape=v.shape, dtype=tf.float32)))
        print sess.run(outputs)


def t():
    true = np.random.randn(8192, 16, 72)


    hidden = 36
    batch_size = 16
    max_time = 300
    sess = tf.Session()
    inputs = tf.placeholder(tf.float32, (None, batch_size, hidden))
    with tf.variable_scope("l1_kernel_1"):
        l1_kernel_1 = RnnConvKernel(3, tf.nn.rnn_cell.BasicRNNCell(hidden))
        l1_outputs_1 = dynamic_rnn_conv(inputs, None, l1_kernel_1, 1)
    with tf.variable_scope("l1_kernel_2"):
        l1_kernel_2 = RnnConvKernel(3, tf.nn.rnn_cell.BasicRNNCell(hidden))
        l1_outputs_2 = dynamic_rnn_conv(inputs, None, l1_kernel_2, 1)
    l2_inputs = tf.concat([l1_outputs_1, l1_outputs_2], axis=2)
    l2_kernel = RnnConvKernel(3, tf.nn.rnn_cell.BasicRNNCell(hidden * 2))
    outputs = dynamic_rnn_conv(l2_inputs, None, l2_kernel, 1)
    loss = tf.nn.l2_loss(outputs - true)
    opt = tf.train.AdamOptimizer().minimize(loss)
    print [x.name for x in tf.global_variables()]
    summary_writer = tf.summary.FileWriter("./tensorboard/",graph=tf.get_default_graph())
    summary_writer.flush()
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        _, l = sess.run([opt, loss], {inputs: np.ones((1024 * 8, batch_size, hidden))})
        print l
if __name__ == "__main__":
    pass