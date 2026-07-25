import mne

raw = mne.io.read_raw_eeglab(
    "/home/vamshi/IIT Mandi Academic Folder/HARDWARE_PROJECTS/neuro_feedback/datasets/ds005285-download/sub-009/ses-1/eeg/sub-009_ses-1_task-29ByANT_eeg.set",
    preload=True
)

print(raw)
print(raw.ch_names)
print(raw.get_data().shape)