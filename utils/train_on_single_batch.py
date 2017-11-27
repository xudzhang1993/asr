import tensorflow as tf
import numpy as np
from model.attention import Input

def train_on_single_batch(iterator, model, sess):
    model.inputs = Input(sess.run(*iterator.get_next()))
    while True:
        _, loss = sess.run([model.update, model.loss])
        print loss

