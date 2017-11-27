import tensorflow as tf
from utils.iterator import get_dataset
def check(l1, l2):
    for i in range(len(l1)):
        try:
            l2.index(l1[i])
            print "check not pass"
        except ValueError, e:
            print e
            pass

hparams = tf.contrib.training.HParams(feature_dimension=39, num_alpha=41, encoder_num_hidden=128, encoder_num_layers=2,
                                      decoder_num_hidden=128,
                                      batch_size=32, make_buckets=False, num_buckets=0, feature_file='', label_file='',
                                      sos_id=39, eos_id=40, dev_size=14, save_path="./save/", epochs=300,
                                      restore=True)
d,t = get_dataset(hparams)
iter = d.make_initializable_iterator()
sess = tf.Session()
for i in range(5):
    l = [[], []]
    sess.run(iter.make_initializer(d))
    print "_____________________________________"
    while True:
        try:
            l[0].append(sess.run(iter.get_next())[0].shape[1])
        except tf.errors.OutOfRangeError:
            break
    sess.run(iter.make_initializer(t))
    while True:
        try:
            l[1].append(sess.run(iter.get_next())[0].shape[1])
        except tf.errors.OutOfRangeError:
            break
    check(l[0], l[1])
