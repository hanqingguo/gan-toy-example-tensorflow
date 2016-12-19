from ops import *
import tensorflow as tf
import numpy as np


class GAN(object):
    def __init__(self, sess, data, discrim_hidden_dim, gen_hidden_dim, k=1, x_dim=1, lr=0.001, z_dim=1,
                 initializer=tf.truncated_normal_initializer(), batch_size=64, std=1.0, mean=0.0):
        """
        Args:
            sess: tf.Session()
            data: a function generates x and y
            discrim_hidden_dim: iterable
                the dimension of discriminator network
            gen_hidden_dim: iterable
                the dimension of generator network
            y_dim: scalar
                the dimension of input data of both
                generator and discriminator networks.
            initializer: tensorflow initializer
            batch_size: scalar
        """
        self.sess = sess
        self.lr = lr
        self.data = data
        self.k = k
        self.std = std
        self.mean = mean

        self.discrim_hidden_dim = discrim_hidden_dim
        self.gen_hidden_dim = gen_hidden_dim

        # x_dim is the input dim for discriminator, z_dim is for generator
        self.y_dim = x_dim
        self.z_dim = z_dim

        self.initializer = initializer

        self.batch_size = batch_size

        # input to the discriminator
        self.x = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.y_dim))

        # input to the generator
        self.z = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.z_dim))

        # g: generator, d1: discriminator for real data, d2: discriminator for fake data
        self.loss_d, self.loss_g, self.g, self.d1, self.d2, self.summary_d, self.summary_d1, self.summary_d2, \
        self.summary_g = self.build()

        # parameters for d and g
        self.d_params = [v for v in tf.trainable_variables() if "discriminator" in v.name]
        self.g_params = [v for v in tf.trainable_variables() if "generator" in v.name]

        self.d_opt, self.g_opt = self.optimizer()


        # initialize all variables
        self.sess.run(tf.global_variables_initializer())

        self.writer = tf.train.SummaryWriter("log")
        self.writer.add_graph(tf.get_default_graph())

    def build(self):
        """
        Build the model
        """
        # generate fake data from noise prior pg(z)
        g = self.generator(self.z)

        # the probability of data coming from the real data
        d1 = self.discriminator(self.x, False)
        # the probability of data coming from the fake data
        d2 = self.discriminator(g, True)

        # loss for discriminator
        loss_d1 = - tf.log(d1, name="loss_d1")
        loss_d2 = - tf.log(1 - d2, name="loss_d2")
        loss_d = tf.reduce_mean(loss_d1 + loss_d2)

        # loss for generator
        loss_g = tf.reduce_mean(-tf.log(d2, name="loss_g"))

        # add summary
        summary_d = tf.summary.scalar("discriminator_loss", loss_d)
        summary_d1 = tf.summary.scalar("real_data_loss", tf.reduce_mean(loss_d1))
        summary_d2 = tf.summary.scalar("fake_data_loss", tf.reduce_mean(loss_d2))
        summary_g = tf.summary.scalar("generator_loss", loss_g)

        return loss_d, loss_g, g, d1, d2, summary_d, summary_d1, summary_d2, summary_g

    def discriminator(self, x, reuse):
        """
        Build the discriminator network.
        Args:
            x: tf.placeholder
                A placeholder representing input data to the discriminator network
            reuse: bool
                Indicating if reuse this model. The reuse situation should happen when discriminating
                the generator inputs
        """
        with tf.variable_scope("discriminator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            # build discriminator network
            x = linear(x, self.discrim_hidden_dim[0], initializer=self.initializer, scope="fc1")
            x = tf.nn.relu(x)

            if len(self.discrim_hidden_dim) > 1:   # if there is more than 1 hidden layer

                # from the second dimension to the last hidden dimension
                for index, dim in enumerate(self.discrim_hidden_dim[1:]):
                    x = linear(x, dim, initializer=self.initializer, scope="fc%s" % (index+2))
                    x = tf.nn.relu(x)

            # output layer
            x = tf.nn.sigmoid(linear(x, dim=self.y_dim, initializer=self.initializer,
                                     scope="fc%s" % (len(self.discrim_hidden_dim)+1)))
        return x

    def generator(self, x):
        with tf.variable_scope("generator"):
            # build generator network
            x = linear(x, self.gen_hidden_dim[0], initializer=self.initializer, scope="fc1")
            x = tf.nn.relu(x)

            if len(self.gen_hidden_dim) > 1:   # if there is more than one hidden layer

                # from the second dimension to the last hidden dimension
                for index, dim in enumerate(self.gen_hidden_dim[1:]):
                    x = linear(x, dim, initializer=self.initializer, scope="fc%s" % (index+2))
                    x = tf.nn.relu(x)

            # output layer
            x = linear(x, dim=self.y_dim, initializer=self.initializer, scope="fc%s" % (len(self.gen_hidden_dim)+1))
        return x

    def optimizer(self):
        d_opt = tf.train.AdamOptimizer(self.lr, name="d_opt")
        g_opt = tf.train.AdamOptimizer(self.lr, name="g_opt")

        d_optimizer = d_opt.minimize(self.loss_d, var_list=self.d_params)
        g_optimizer = g_opt.minimize(self.loss_g, var_list=self.g_params)
        return d_optimizer, g_optimizer

    def train(self, max_epoch, print_every):
        for i in range(max_epoch):
            # sample data z from noise prior (uniform distribution in this case)
            z = np.random.uniform(size=(self.batch_size, self.z_dim))
            x = self.data(self.std, self.mean, self.batch_size, self.y_dim)

            # train discriminator for self.k steps
            for _ in range(self.k):
                loss_d, _, summary_d, summary_d1, summary_d2 = self.sess.run(
                    [self.loss_d, self.d_opt, self.summary_d, self.summary_d1, self.summary_d2],
                    feed_dict={self.x: x, self.z: z})
                self.writer.add_summary(summary_d, i)
                self.writer.add_summary(summary_d1, i)
                self.writer.add_summary(summary_d2, i)

            # train generator
            z = np.random.uniform(size=(self.batch_size, self.z_dim))
            loss_g, _, summary_g = self.sess.run([self.loss_g, self.g_opt, self.summary_g], feed_dict={self.z: z})
            self.writer.add_summary(summary_g, i)

            if i % print_every == 0:
                print("At iteration %s, discriminator loss is: %s" % (i, loss_d))
                print("At iteration %s, generator loss is %s" % (i, loss_g))

    def sample(self, num_points=64000, bin_num=50):
        """
        Return decision boundary, generator data distribution, real data distribution
        """
        # real data
        x = self.data(self.std, self.mean, num_points, self.y_dim)

        # decision boundary
        decision = np.zeros(num_points)

        # calculate decision boundary for real data
        for i in range(num_points/self.batch_size):
            decision[i*self.batch_size:(i+1)*self.batch_size] = self.sess.run(
                self.d1, feed_dict={self.x: x[i*self.batch_size: (i+1)*self.batch_size]}
            ).reshape(self.batch_size)

        # calculate generated data
        z = np.random.uniform(size=(num_points, self.z_dim))
        generator = np.zeros((num_points, 1))
        for i in range(num_points/self.batch_size):
            generator[i*self.batch_size:(i+1)*self.batch_size, :] = self.sess.run(
                self.g, feed_dict={self.z: z[i*self.batch_size:(i+1)*self.batch_size]}
            )

        # number of bins
        bins = np.linspace(np.min(x), np.max(x), bin_num)

        # real data distribution
        dd, _ = np.histogram(x, bins=bins, density=True)

        # generator data distribution
        gd, _ = np.histogram(generator, bins=bins, density=True)

        return decision, gd, dd, x