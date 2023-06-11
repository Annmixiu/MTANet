"""

This file contains all constants, hyperparameters, and settings for the model

"""

exp_name = "MTAnet"
# file path
model_type = "MTAnet" # MCDNN, FTANet, MSNet, MLDRNet
data_path = "data"
train_file = "data/train_data.txt"
test_file = [
    "data/test_adc.txt",
    "data/test_mirex.txt",
    "data/test_melody.txt"
]
    
save_path = "model_backup"
resume_checkpoint = None
# "model_backup/TO-FTANet_adc_best.ckpt" # the model checkpoint

# train config
batch_size = 6
lr = 1e-4
epochs = 1000
n_workers = 4  # 4 processes are used for data loading; the default is 0, which loads all data
save_period = 1
tone_class = 12  # 60
octave_class = 8  # 6
random_seed = 19980627
max_epoch = 500  # set max epoch
freq_bin = 360
input_channel = 32

ablation_mode = "single"
startfreq = 32
stopfreq = 2050
cfp_dir = "cfp_360_new"  # cfp_360_new cfp_360_noise

# feature config
fs = 8000.0
hop = 80.0
octave_res = 60  # num_fre_bin of each octave
seg_dur = 1.28  # the audio segment is 1.28 seconds
seg_frame = int(seg_dur * fs // hop)
shift_dur = 1.28  # sec
shift_frame = int(shift_dur * fs // hop)
