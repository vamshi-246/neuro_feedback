import numpy as np
import mne
import pandas as pd
# import matplotlib
# matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import scipy.io as sio


# epochs = mne.io.read_epochs_eeglab("datasets/ds005285-download/derivatives/mark_ica/sub-001_29ByANT.set")
# times = epochs.times
# data = epochs.get_data()

# first_trial = data[0]

# plt.plot(times, first_trial[0])
# plt.savefig("temp.png")
band_hi = np.load("outputs/pipeline_dev/pooled_features/band_hi.npy")
band_lo = np.load("outputs/pipeline_dev/pooled_features/band_lo.npy")
features = np.load("outputs/pipeline_dev/pooled_features/features_uint8.npy")
dataset_id = np.load("outputs/pipeline_dev/pooled_features/dataset_id.npy")
labels = np.load("outputs/pipeline_dev/pooled_features/labels.npy")
ratings = np.load("outputs/pipeline_dev/pooled_features/ratings.npy")
subject_id = np.load("outputs/pipeline_dev/pooled_features/subject_id.npy")
print(features.shape)
# print(dataset_id)
print(labels)
# print(ratings)
# print(subject_id)
# print(band_hi)
# print(band_lo)