import librosa
from python_speech_features import fbank, mfcc
import numpy as np


def extract_feature(sig, rate, filter_num=40, delta1=True, delta2=True, type='fbank', use_energy=True):
    if type == 'fbank':
        fb, energy = fbank(sig, rate, nfilt=filter_num)
    else:
        raise ValueError("unknown feature type")
    if use_energy:
        energy = energy[:, np.newaxis]
        feature = np.concatenate((fb, energy), -1)
    else:
        feature = fb
    if delta2:
        delta1_feature = _delta(feature, N=2)
        delta2_feature = _delta(delta1_feature, N=2)
        feature = np.c_[feature, delta1_feature, delta2_feature]
    elif delta1:
        delta1_feature = _delta(feature, N=2)
        feature = np.c_[feature, delta1_feature]

    feature = (feature - np.mean(feature)) / np.std(feature)

    return feature


def _delta(feat, N):
    """Compute delta features from a feature vector sequence.
    Args:
        feat: A numpy array of size (NUMFRAMES by number of features)
            containing features. Each row holds 1 feature vector.
        N: For each frame, calculate delta features based on preceding and
            following N frames
    Returns:
        A numpy array of size (NUMFRAMES by number of features) containing
            delta features. Each row holds 1 delta feature vector.
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N + 1)])
    delta_feat = np.empty_like(feat)
    # padded version of feat
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')
    for t in range(NUMFRAMES):
        # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
        delta_feat[t] = np.dot(np.arange(-N, N + 1),
                               padded[t: t + 2 * N + 1]) / denominator
    return delta_feat