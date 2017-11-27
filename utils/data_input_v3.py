import os
from experiment.timit.config import hparams
from experiment import create_none_input, ModelFeed
import tensorflow as tf


def _parse_example_proto(example_proto):
    pattern = {
        'feature': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'target_input': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'target_output': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'feature_length': tf.FixedLenFeature([], tf.int64),
        'target_length': tf.FixedLenFeature([], tf.int64),
    }
    example = tf.parse_single_example(example_proto, features=pattern)
    return example['feature'], example['target_input'], example['target_output'], example['feature_length'], example['target_length']


def get_dataset():
    record_file = tf.placeholder(tf.string, [])
    ds = tf.data.TFRecordDataset(record_file)
    ds = ds.map(lambda x: _parse_example_proto(x))
    ds = ds.map(lambda x1, x2, x3, x4, x5: (tf.reshape(x1, [-1, hparams.feature_dimension]), x2, x3, x4, x5))

    ds = ds.padded_batch(hparams.batch_size,
                        padded_shapes=(tf.TensorShape([None, hparams.feature_dimension]),  # src
                                       tf.TensorShape([None]),  # tgt_input
                                       tf.TensorShape([None]),  # tgt_output
                                       tf.TensorShape([]),  # src_len
                                       tf.TensorShape([])),  # tgt_len
                        padding_values=(0.0,  # src
                                        tf.cast(hparams.eos_id, tf.int64),  # tgt_input
                                        tf.cast(hparams.eos_id, tf.int64),  # tgt_output
                                        tf.cast(0, tf.int64),  # tgt_output
                                        tf.cast(0, tf.int64),)  # tgt_output
                        )
    ds = ds.filter(lambda x, *_: tf.equal(tf.shape(x)[0], hparams.batch_size))
    ds = ds.shuffle(buffer_size=256)
    return ds, record_file


if __name__ == "__main__":
    ds, rf = get_dataset()
    it = ds.make_initializable_iterator()
    with tf.Session() as sess:
        sess.run(it.initializer, feed_dict={rf: hparams.data_dir + 'train.record'})
        while True:
            print sess.run(it.get_next())[4].shape[0]
       #print '--------------------'

       #sess.run(it.initializer, feed_dict={rf: hparams.data_dir + 'dev.record'})
       #print sess.run(it.get_next())[4]
       #print '--------------------'

       #sess.run(it.initializer, feed_dict={rf: hparams.data_dir + 'train.record'})
       #print sess.run(it.get_next())[4]
       #print '--------------------'

       #sess.run(it.initializer, feed_dict={rf: hparams.data_dir + 'dev.record'})
       #print sess.run(it.get_next())[4]
       #print '--------------------'
