import tensorflow as tf

hparams = tf.contrib.training.HParams(feature_dimension=123, num_alpha=63, encoder_num_hidden=200, encoder_num_layers=3,
                                      decoder_num_hidden=200, filter_width=3,filter_channels=128, filter_stride=1,
                                      batch_size=32, make_buckets=False, num_buckets=0, feature_file='', label_file='',
                                      sos_id=61, eos_id=62, dev_size=14, data_dir='../data/', save_path="../model_save/", epochs=300,
                                      restore=False, log_path="../model_log/", step_per_eval=200)
