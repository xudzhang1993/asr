import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from experiment.timit.config import hparams
from model.attention import AttentionModel
from utils.general import generate_voc_table_61, edit_distance, print_variables, truncate, map_phoneme
from utils.iterator import get_dataset
from utils.queue import TimitDataSet

if __name__ == "__main__":
    best_eval_score = 100
    with tf.Session() as sess:
        attention_model = AttentionModel(hparams, iterator, generate_voc_table_61())
        save_path = os.path.join(hparams.save_path, attention_model.name)
        log_path = os.path.join(hparams.log_path, attention_model.name)
        #if not os.path.isdir(save_path):
        #    os.makedirs(save_path)
        #if not os.path.isdir(log_path):
        #    os.makedirs(log_path)
        ckpt = tf.train.get_checkpoint_state(hparams.save_path)
        print ckpt
        if hparams.restore and ckpt and ckpt.model_checkpoint_path:
            attention_model.saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model restored from:' + hparams.save_path)
        else:
            sess.run(tf.global_variables_initializer())
        print_variables()
        sess.run(tf.tables_initializer())
        summary_writer = tf.summary.FileWriter(log_path, graph=sess.graph)
        dev_set = TimitDataSet(mode="dev")
        train_set = TimitDataSet(mode="train")
        for epoch in range(hparams.epochs):
            batch = 0
            batch_loss_list = []
            batch_per_list = []
            while True:
                _, batch_loss, summary = sess.run([attention_model.update, attention_model.loss, attention_model.summary])
                batch_loss_list.append(batch_loss)
                summary_writer.add_summary(summary)
                print("epoch: %d batch: %d loss: %f" % (epoch, batch, batch_loss))
                batch += 1
                target, predict = sess.run([attention_model.target_string, attention_model.predict_string])
                map_phoneme(predict)
                map_phoneme(target)
                per = sess.run(edit_distance(predict, target))
                aver_per = np.mean(per)
                batch_per_list.append(aver_per)
                index = np.random.randint(0, target.shape[0])
                print "epoch"
                print "average per : %f" % aver_per
                print "target : "
                print truncate(target[index])
                print "predict: "
                print truncate(predict[index])
                break
            epoch_per = np.mean(batch_per_list)
            print("epoch_per: %f" % epoch_per)
            print("best_per: %f") % best_eval_score
            if best_eval_score > epoch_per:
                print "we has a better model"
                best_eval_score = epoch_per
                attention_model.saver.save(sess, hparams.save_path, global_step=epoch)
                print "best model has been saved"
