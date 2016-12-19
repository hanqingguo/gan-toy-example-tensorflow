import tensorflow as tf


def linear(x, dim, initializer=tf.truncated_normal_initializer(), scope=None):
    """
    Args:
        x: tensor of shape (N, D)
        dim: scalar
        initializer: initializer for weight
        scope: str
    Returns:
        tensor representing transformation of this layer
    """
    scope = scope or "fc"
    with tf.variable_scope(scope):
        N, D = x.get_shape().as_list()

        # weight
        w = tf.get_variable(name="%s_w" % scope, initializer=initializer, dtype=tf.float32, shape=(D, dim))

        # bias
        b = tf.get_variable(name="%s_b" % scope, initializer=tf.constant_initializer(value=0.1), shape=dim)

        # start calculation
        x = tf.matmul(x, w)
        x = tf.nn.bias_add(x, b)
        return x