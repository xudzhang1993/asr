from utils.iterator import get_dataset
from timit import hparams
import tensorflow as tf
from tensorflow.contrib.data.python.ops.dataset_ops import Dataset, TextLineDataset
#dev_ds, train_ds = get_dataset(hparams)
#iterator = train_ds.make_initializable_iterator()
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
#sess.run(iterator.initializer)
#for i in range(2):
#    while True:
#        print "-------------------"
#        try:
#            print sess.run(iterator.get_next())[0].shape[0]
#        except tf.errors.OutOfRangeError:
#            sess.run(iterator.initializer)
#            break

data = TextLineDataset('./a.txt')
data = data.batch(4)
iterator = data.make_initializable_iterator()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(iterator.initializer)
print iterator.get_next().shape
for i in range(2):
    while True:
        try:
            print sess.run(iterator.get_next()).shape[0]
        except Exception:
            sess.run(iterator.initializer)
            break



