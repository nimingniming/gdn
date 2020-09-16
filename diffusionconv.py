import tensorflow as tf

class DiffusionConv(GraphConv):
    """
    **Input**
    - Node features of shape `([batch], N, F)`;
    - Normalized adjacency or attention coef. matrix \(\hat \A \) of shape
    `([batch], N, N)`
    **Output**
    - Node features with the same shape as the input, but with the last
    dimension changed to `channels`.
    **Arguments**
    - `channels`: number of output channels;
    - `num_diffusion_steps`: How many diffusion steps to consider. \(K\) in paper.
    - `activation`: activation function
    - `kernel_initializer`: initializer for the weights;
    - `kernel_regularizer`: regularization applied to the weights;
    - `kernel_constraint`: constraint applied to the weights;
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

        # number of features to generate (Q)
        assert channels > 0
        self.Q = channels

        # number of diffusion steps for each output feature
        self.K = num_diffusion_steps + 1

    def build(self, input_shape):

        # We expect to receive (X, A)
        # A - Adjacency ([batch], N, N)
        # X - graph signal ([batch], N, F)
        X_shape, A_shape = input_shape

        # initialise Q diffusion convolution filters
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
        """Applies diffusion convolution self.Q times to get a
        ([batch], N, Q) diffused graph signal
        """

        # This will be a list of Q diffused features.
        # Each diffused feature is a (batch, N, 1) tensor.
        # Later we will concat all the features to get one
        # (batch, N, Q) diffused graph signal
        diffused_features = []

        # Iterating over all Q diffusion filters
        for diffusion in self.filters:
            diffused_feature = diffusion((X, A))
            diffused_features.append(diffused_feature)

        # Concat them into ([batch], N, Q) diffused graph signal
        H = tf.concat(diffused_features, -1)

        return H

    def call(self, inputs):

        # Get graph signal X and adjacency tensor A
        X, A = inputs

        H = self.apply_filters(X, A)
        H = self.activation(H)

        return H