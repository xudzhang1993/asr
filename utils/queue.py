import os
from experiment.timit.config import hparams
from experiment import create_none_input, ModelFeed
import tensorflow as tf


class TimitDataSet(object):
    def __init__(self, record_dir="/home/xudong/data/timit/tensorflow/", mode="train"):
        self.record_file = os.path.join(record_dir, mode + '.record')
        filename_queue = tf.train.string_input_producer([self.record_file])
        reader = tf.TFRecordReader()
        _, example = reader.read(filename_queue)
        #examples = tf.train.shuffle_batch([example], hparams.batch_size, 1024, 256, 4)
        content = tf.parse_single_example(example,
                                           features={
                                               'feature': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                                               'target_input': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                                               'target_output': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                                               'feature_length': tf.FixedLenFeature([], tf.int64),
                                               'target_length': tf.FixedLenFeature([], tf.int64),
                                               }
                                           )
        feature = tf.reshape(content['feature'], (-1, hparams.feature_dimension))

        self.batched = tf.train.batch([feature, content['target_input'], content['target_output'], content['feature_length'], content['target_length']], hparams.batch_size, dynamic_pad=True)
        #self.features['feature'] = tf.sparse_tensor_to_dense(self.features['feature'], default_value='')
        #self.features['feature'] = tf.reshape(tf.decode_raw(self.features['feature'], tf.float32), [-1, 123])

    def get_next(self):
        return self.batched


if __name__ == "__main__":
    with tf.Session() as sess:
        dataset = TimitDataSet()
        tf.train.start_queue_runners(sess=sess)
        a = sess.run(dataset.get_next())
        print a
