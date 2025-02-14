import math
import uuid
from datetime import datetime

#########################################################################
# MODEL PARAMETERS														#
#########################################################################
# number of rows of input images
shape_r = 480
# number of cols of input images
shape_c = 640
# number of rows of predicted maps
shape_r_gt = int(math.ceil(shape_r / 8))
# number of cols of predicted maps
shape_c_gt = int(math.ceil(shape_c / 8))

#########################################################################
# OTHER STUFF                                                           #
#########################################################################
total_frames_each_run = 7500
batchsize = 10
dreyeve_train_seq = range(1, 37+1)
dreyeve_test_seq = range(38, 74+1)

train_frame_range = list(range(0, 3500)) + list(range(4000, total_frames_each_run))
val_frame_range = range(3500, 4000)
test_frame_range = range(0, total_frames_each_run)
DREYEVE_DIR = 'C:/Users/SCTW54265/Olivia/DReyeVE/code/DREYEVE_DATA'

#########################################################################
# TRAINING SETTINGS										            	#
#########################################################################
# number of validation images
nb_imgs_val = 64 * batchsize
# number of epochs
nb_epoch = 2
# samples per epoch
nb_samples_per_epoch = 2 * batchsize


#########################################################################
# EXPERIMENT ID															#
#########################################################################
# use current datetime to generate a unique experiment id
datetime_now = datetime.now()
# convert datetime to string
datetime_now_str = datetime_now.strftime("%Y-%m-%d_%H-%M-%S")
each_experiment_id = datetime_now_str


# Append the experiment_id to the file
with open("all_experiments.txt", "a") as f:
    # Add a newline character before appending the new experiment_id
    f.write(f"\n{each_experiment_id}")


# experiment_id = "e3cfa94f-cc94-4098-88ce-8b1656ce43e7"
