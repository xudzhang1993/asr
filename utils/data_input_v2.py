import tensorflow as tf
import os
from tensorflow.python.ops.data_flow_ops import RecordInput

class TimitDataSet(object):
    def __init__(self, record_dir="/home/xudong/data/timit/tensorflow/", mode="train"):
        self.record_file = os.path.join(record_dir, mode + '.record')
        pipeline = RecordInput(self.record_file, batch_size=32)
        serialized_example = pipeline.get_yield_op()
        self.features = tf.parse_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                                               'feature': tf.FixedLenSequenceFeature([], tf.string, allow_missing=True),
                                               }
                                           )
        tf.parse_example()
        #self.features['feature'] = tf.sparse_tensor_to_dense(self.features['feature'], default_value='')
        #self.features['feature'] = tf.reshape(tf.decode_raw(self.features['feature'], tf.float32), [-1, 123])

    def get_batch(self):
        return self.features['label'], self.features['feature']



if __name__ == "__main__":
    with tf.Session() as sess:
        dataset = TimitDataSet()
        print sess.run(dataset.get_batch())
