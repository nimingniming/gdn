import tensorflow as tf

class DiffusionConv(GraphConv):
    """
    **Input**
    - Node features of shape `([batch], N, F)`;
    - Normalized adjacency 
    **Output**
    - Node features with the same shape as the input, but with the last
    dimension changed to `channels`.
    """

    def __init__(
        self,
        channels: int,
        num_diffusion_steps: int = 6,
        kernel_initializer='glorot_uniform',
        kernel_regularizer=None,
        kernel_constraint=None,
        activation='relu',
        ** kwargs
    ):
        super().__init__(channels,
                         activation=activation,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer,
                         kernel_constraint=kernel_constraint,
                         **kwargs)
        assert channels > 0
        self.Q = channels
        # number of diffusion steps 
        self.K = num_diffusion_steps + 1

    def build(self, input_shape):
    
        X_shape, A_shape = input_shape
        self.filters = []

        for _ in range(self.Q):
            layer = DiffuseFeatures(
                num_diffusion_steps=self.K,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
            )
            self.filters.append(layer)

    def apply_filters(self, X, A):
    
        diffused_features = []
        for diffusion in self.filters:
            diffused_feature = diffusion((X, A))
            diffused_features.append(diffused_feature)
        H = tf.concat(diffused_features, -1)

        return H

    def call(self, inputs):

        # Get graph signal X and adjacency tensor A
        X, A = inputs
        H = self.apply_filters(X, A)
        H = self.activation(H)

        return H
