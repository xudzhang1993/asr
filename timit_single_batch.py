import numpy as np
import tensorflow as tf

from model.attention import AttentionModel
from model.attention_related_model import ConvModel
from utils.general import generate_voc_table, edit_distance, print_variables, truncate
from utils.iterator import get_dataset

hparams = tf.contrib.training.HParams(feature_dimension=39, num_alpha=41, encoder_num_hidden=128, encoder_num_layers=2,
                                      decoder_num_hidden=128,filter_width=3,filter_channels=128,filter_stride=1,
                                      batch_size=32, make_buckets=False, num_buckets=0, feature_file='', label_file='',
                                      sos_id=39, eos_id=40, dev_size=14, save_path="./save_monotonic/", epochs=300,
                                      restore=False)
with tf.Session() as sess:
    dev_ds, train_ds = get_dataset(hparams)
    iterator = train_ds.make_initializable_iterator()
    attention_model = AttentionModel(hparams, iterator, generate_voc_table(),single_batch=True)
    sess.run(tf.global_variables_initializer())
    print_variables()
    sess.run(tf.tables_initializer())
    sess.run(iterator.make_initializer(train_ds))
    input_value = sess.run(iterator.get_next())
    for epoch in range(hparams.epochs):
        _, loss = sess.run([attention_model.update, attention_model.loss], feed_dict={attention_model.inputs: input_value})
        print loss
