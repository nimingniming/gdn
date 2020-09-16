import tensorflow as tf

class DiffuseFeatures(layers.Layer):
    """Utility layer calculating a single channel of the
    diffusional convolution.
    """

    def __init__(
        self,
        num_diffusion_steps: int,
        kernel_initializer,
        kernel_regularizer,
        kernel_constraint,
        **kwargs
    ):
        super(DiffuseFeatures, self).__init__()

        # number of diffusino steps (K in paper)
        self.K = num_diffusion_steps

        # get regularizer, initializer and constraint for kernel
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint

    def build(self, input_shape):

        self.kernel = self.add_weight(
            shape=(self.K,),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

    def call(self, inputs):

        # Get signal X and adjacency A
        X, A = inputs
        diffusion_matrix = tf.math.polyval(tf.unstack(self.kernel), A)
        diffused_features = tf.matmul(diffusion_matrix, X)
        H = tf.math.reduce_sum(diffused_features, axis=-1)
        return tf.expand_dims(H, -1)