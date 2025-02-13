from keras.layers import Layer, InputSpec
from keras import constraints, regularizers, initializers, activations  # Updated import
import tensorflow as tf  # Replace Theano with TensorFlow

class EltWiseProduct(Layer):
    def __init__(self, downsampling_factor=10, init='glorot_uniform', activation='linear',
                 weights=None, W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, input_dim=None, **kwargs):

        self.downsampling_factor = downsampling_factor
        self.init = initializers.get(init)  # Updated to initializers
        self.activation = activations.get(activation)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        self.initial_weights = weights

        # self.input_dim = input_dim
        # if self.input_dim:
        #     kwargs['input_shape'] = (self.input_dim,)

        # self.input_spec = [InputSpec(ndim=4)] # Expecting 4D input, batch_size x height x width x channels
        super(EltWiseProduct, self).__init__(**kwargs)


    def build(self, input_shape):
        # Create weight tensor with shape derived from input shape and downsampling factor
        # print("input_shape", input_shape)
        height = input_shape[1]
        width = input_shape[2]
        self.W = self.add_weight(
            name='kernel',
            # shape=[s // self.downsampling_factor for s in input_shape[2:]],  # Calculate weight shape based on input shape
            shape=[height // self.downsampling_factor, width // self.downsampling_factor], 
            initializer=self.init,
            regularizer=self.W_regularizer,  # Use regularizer here
            constraint=self.W_constraint
        )

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)

        # If initial weights are provided, set them
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output_shape_for(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        # Replace Theano's bilinear_upsampling with TensorFlow's resize_bilinear
        W_expanded = tf.expand_dims(tf.expand_dims(1 + self.W, 0), 0)
        W_upsampled = tf.image.resize(W_expanded, [x.shape[1], x.shape[2]], method='bilinear')
        output = x * W_upsampled
        return output

    def get_config(self):
        config = {
            'name': self.__class__.__name__,
            'init': initializers.serialize(self.init),  # Serialize initializer
            'activation': activations.serialize(self.activation),  # Serialize activation
            'W_regularizer': regularizers.serialize(self.W_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'W_constraint': constraints.serialize(self.W_constraint),
            # 'input_dim': self.input_dim,
            'downsampling_factor': self.downsampling_factor
        }
        base_config = super(EltWiseProduct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))