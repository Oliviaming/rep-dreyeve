import math
import uuid

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
nb_epoch = 50
# samples per epoch
nb_samples_per_epoch = 256 * batchsize

each_experiment_id = uuid.uuid4()

# Save the experiment_id to a file
with open("last_experiment.txt", "w") as f:
    f.write(str(each_experiment_id))


# experiment_id = "e3cfa94f-cc94-4098-88ce-8b1656ce43e7"
