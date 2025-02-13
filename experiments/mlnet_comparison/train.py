from __future__ import division
from keras.optimizers import SGD

from callbacks import get_callbacks

from model import ml_net_model, loss
from batch_generators import generate_batch

from config import shape_c, shape_r, batchsize
from config import nb_samples_per_epoch, nb_epoch, nb_imgs_val


if __name__ == '__main__':

    model = ml_net_model(img_cols=shape_c, img_rows=shape_r, downsampling_factor_product=10)
    sgd = SGD(learning_rate=1e-3, decay=0.0005, momentum=0.9, nesterov=True)
    print("Compile ML-Net Model")
    model.compile(sgd, loss)
    model.summary()

    print("Training ML-Net")
    model.fit(x=generate_batch(batchsize=batchsize, mode='train', gt_type='fix'),
                        validation_data=generate_batch(batchsize=batchsize, mode='val', gt_type='fix'),
                        validation_steps=nb_imgs_val // batchsize,  # Adjust the validation steps according to your dataset
                        epochs=nb_epoch,
                        steps_per_epoch=nb_samples_per_epoch // batchsize,  # Adjust the steps per epoch according to your dataset
                        callbacks=get_callbacks())
