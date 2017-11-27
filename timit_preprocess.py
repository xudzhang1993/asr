import os
from python_speech_features import fbank
from scikits.audiolab import Format, Sndfile
import scipy
import numpy as np
from utils.feature import extract_feature
import tensorflow as tf
from experiment.timit.config import hparams

timit_dir = '/home/xudong/data/timit'
import random

phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el',
       'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l',
       'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux',
       'v', 'w', 'y', 'z', 'zh']


def split_speaker(timit_dir=timit_dir, mode='train'):
    root_dir = os.path.join(timit_dir, mode)
    all_speakers = set()
    for subdir, dirs, files in os.walk(root_dir):
        # print subdir, dirs, files
        file_dir = os.path.join(root_dir, subdir)
        for file_name in files:
            if (file_name.endswith('.wav') or file_name.endswith('.WAV')) and not (
                file_name.startswith('sa') or file_name.startswith('SA')):
                spk_id = subdir.split('/')[-1]
                utt_id = file_name.split('.')[0][2:]
                all_speakers.add(spk_id)
    print "all speaker num: ", len(all_speakers)
    dev_speakers = set(random.sample(all_speakers, 20))
    train_speakers = all_speakers - dev_speakers
    print "dev speaker num: ", len(dev_speakers)
    print "train speaker num: ", len(train_speakers)
    return dev_speakers, train_speakers


def timit(timit_dir, record_dir="/home/xudong/data/timit/tensorflow", mode='train'):
    print "mode has been setting to %s", mode
    root_dir = os.path.join(timit_dir, mode)
    dev_speakers = []
    train_speakers = []
    if mode == 'train':
        dev_speakers, train_speakers = split_speaker(timit_dir)
        record_files = [os.path.join(record_dir, x + '.record') for x in ['train', 'dev']]
    else:
        record_files = [os.path.join(record_dir, 'test.record')]
    for record_file in record_files:
        if os.path.isfile(record_file):
            os.remove(record_file)
    writers = [tf.python_io.TFRecordWriter(x) for x in record_files]
    i = 0
    for subdir, dirs, files in os.walk(root_dir):
        # print subdir, dirs, files
        file_dir = os.path.join(root_dir, subdir)
        for file_name in files:
            if (file_name.endswith('.wav') or file_name.endswith('.WAV')) and not (
                file_name.startswith('sa') or file_name.startswith('SA')):
                spk_id = subdir.split('/')[-1]
                utt_id = file_name.split('.')[0][2:]
                if mode == 'train':
                    if spk_id in train_speakers:
                        writer = writers[0]
                    elif spk_id in dev_speakers:
                        writer = writers[1]
                    else:
                        raise ValueError("Unknown speaker id %s" % spk_id)
                else:
                    writer = writers[0]
                # print 'process wav file (%s, %s)' % (spk_id, utt_id)
                phenome = [hparams.sos_id]
                # read phonome
                with open(os.path.join(file_dir, file_name.replace('.wav', '.phn')), 'r') as f:
                    for line in f.read().splitlines():
                        s = line.split(' ')[2]
                        p_index = phn.index(s)
                        phenome.append(p_index)
                    phenome.append(hparams.eos_id)
                        # print(phenome)
                # read wav

                # read NIST format
                sf = Sndfile(os.path.join(file_dir, file_name), 'r')
                nframes = sf.nframes
                sig = sf.read_frames(nframes)
                rate = sf.samplerate
                # extract feature
                feature = extract_feature(sig, rate)
                target_input = phenome[:-1]
                target_output = phenome[1:]
                feature_length = feature.shape[0]
                target_length = len(target_input)
                # print "feature dims:", feature.shape
                record = {"feature": tf.train.Feature(float_list=tf.train.FloatList(value=feature.reshape([-1]))),
                          "target_input": tf.train.Feature(int64_list=tf.train.Int64List(value=target_input)),
                          "target_output": tf.train.Feature(int64_list=tf.train.Int64List(value=target_output)),
                          "feature_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[feature_length])),
                          "target_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[target_length]))}
                example = tf.train.Example(features=tf.train.Features(feature=record))
                writer.write(example.SerializeToString())
                i += 1
                if i == 1:
                    print "first feature:"
                    print "shape:", feature.shape
                    print "feature content: "
                    print feature
                    print "target_input_output:"
                    print target_input
                    print target_output
                    print "feature_target_length"
                    print feature_length, target_length

                if i % 20 == 0:
                    print "have written:", i, "record"
                    print "feature shape", feature.shape
    for writer in writers:
        writer.close()


if __name__ == "__main__":
    print tf.VERSION
    timit(timit_dir, mode='train')
