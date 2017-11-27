import tensorflow as tf
import os

import time


def shuffle_record_files(hparams):

    ds = tf.contrib.data.TFRecordDataset("/home/xudong/data/tensorASR/timit/TFRecord/phn_fbank/fbank")
    label = tf.contrib.data.TFRecordDataset("/home/xudong/data/tensorASR/timit/TFRecord/phn_fbank/label_39")
    ds = tf.contrib.data.Dataset.zip((ds, label))
    ds = ds.shuffle(buffer_size=1024)
    iterator = ds.make_initializable_iterator()
    sess = tf.Session()
    sess.run(iterator.initializer)
    i = 0
    while True:
        t = time.time()
        sess.run(iterator.get_next())
        print "%d : %f" % (i, time.time() - t)
        i += 1
    # with tf.python_io.TFRecordWriter(
    #         "/home/xudong/data/tensorASR/timit/TFRecord/phn_fbank/fbank_temp") as feature_writer:
    #     with tf.python_io.TFRecordWriter(
    #             "/home/xudong/data/tensorASR/timit/TFRecord/phn_fbank/label_39_temp") as label_writer:
    #         while True:
    #             try:
    #                 feature, label = sess.run(iterator.get_next())
    #                 # feature_writer.write(feature)
    #                 # label_writer.write(label)
    #                 print i
    #                 i += 1
    #             except tf.errors.OutOfRangeError:
    #                 break
    # os.remove("/home/xudong/data/tensorASR/timit/TFRecord/phn_fbank/fbank")
    # os.remove("/home/xudong/data/tensorASR/timit/TFRecord/phn_fbank/label_39")
    # os.rename("/home/xudong/data/tensorASR/timit/TFRecord/phn_fbank/fbank_temp",
    #           "/home/xudong/data/tensorASR/timit/TFRecord/phn_fbank/fbank")
    # os.rename("/home/xudong/data/tensorASR/timit/TFRecord/phn_fbank/label_39_temp",
    #           "/home/xudong/data/tensorASR/timit/TFRecord/phn_fbank/label_39")


def get_dataset(hparams):
    # get params from hparams
    batch_size = hparams.batch_size
    sos_id = hparams.sos_id
    eos_id = hparams.eos_id
    make_buckets = hparams.make_buckets
    num_buckets = hparams.num_buckets
    dev_size = hparams.dev_size
    feature_file = hparams.feature_file
    label_file = hparams.label_file

    # create dataset from TFRecord files
    # combine the feature and label dataset
    # reform them into numpy array structure(array tensor)
    # shuffle the data set
    ds = tf.contrib.data.TFRecordDataset("/home/xudong/data/tensorASR/timit/TFRecord/phn_fbank/fbank")
    label = tf.contrib.data.TFRecordDataset("/home/xudong/data/tensorASR/timit/TFRecord/phn_fbank/label_61")
    ds = tf.contrib.data.Dataset.zip((ds, label))
    ds = ds.map(lambda x, y: (tf.reshape(tf.decode_raw(x, tf.float32), shape=(-1, 39)), tf.cast(tf.decode_raw(y, tf.int16), tf.int64)))
    ds = ds.map(lambda x, y: (x, tf.concat(([sos_id], y), 0), tf.concat((y, [eos_id]), 0)))
    ds = ds.map(lambda x, y, z: (x, y, z, tf.shape(x)[0], tf.size(y)))
    # ds = ds.shuffle(buffer_size=128)

    # bucket and batch the dataset

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([None, 39]),  # src
                           tf.TensorShape([None]),  # tgt_input
                           tf.TensorShape([None]),  # tgt_output
                           tf.TensorShape([]),  # src_len
                           tf.TensorShape([])),  # tgt_len
            padding_values=(0.0,  # src
                            tf.cast(eos_id, tf.int64),  # tgt_input
                            tf.cast(eos_id, tf.int64),  # tgt_output
                            0,  # src_len -- unused
                            0))  # tgt_len -- unused

    if make_buckets:
        def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
            # Calculate bucket_width by maximum source sequence length.
            # Pairs with length [0, bucket_width) go to bucket 0, length
            # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
            # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
            bucket_width = 10

            # Bucket sentence pairs by the length of their source sentence and target
            # sentence.
            bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        batched_dataset = ds.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size)
    else:
        batched_dataset = batching_func(ds)

    # spilt dataset into train and dev dataset according
    dev_ds = batched_dataset.take(dev_size)
    train_ds = batched_dataset.skip(dev_size)
    train_ds = train_ds.shuffle(buffer_size=128)

    return dev_ds,  train_ds
