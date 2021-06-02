"""
Construct the VaDE model using tensorflow==1.15.0
2021/5/27
"""

import tensorflow as tf
import numpy as np
from sklearn import mixture
import os
import warnings
import matplotlib.pylab as plt
import shutil

warnings.filterwarnings("ignore")

pretrainmodel = r'.\pretrain\pretrainmodel'
fullmodel = r'.\trained_model\fullmodel'


def weight_init(size):
    in_dim = size[0]
    limit = tf.sqrt(1/in_dim)
    return tf.random_uniform(shape=size, minval=-limit, maxval=limit)


def bias_init(size):
    in_dim = size[0]
    limit = tf.sqrt(1/in_dim)
    return tf.random_uniform(shape=[size[1]], minval=-limit, maxval=limit)


def reparameterization(mu, logvar):
    epsilon = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(logvar / 2) * epsilon


def reparameterization_value(mu, logvar):
    epsilon = np.random.normal(0, 1, mu.shape)
    return mu + np.exp(logvar / 2) * epsilon


def cluster_acc(Y_pred, Y):
    from scipy.optimize import linear_sum_assignment as linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    temp_sum = 0
    for i in range(D):
        temp_sum += w[ind[0][i], ind[1][i]]
    return temp_sum * 1.0 / Y_pred.size, w


def repeat(data, t, axis=0):
    multiples = np.ones(len(np.shape(data)) + 1)
    multiples[axis] = t
    return tf.tile(tf.expand_dims(data, axis=axis), multiples=multiples)


def get_class(data, axis=1):
    return np.argmax(data, axis=axis)


class VaDE:
    def __init__(self, encoder_layerinfo, decoder_layerinfo, lr, epochs, batch_size, x_dim, z_dim, n_centroid, trainset,
                 testset, trainy, testy):
        self.encoder_layerinfo = encoder_layerinfo
        self.decoder_layerinfo = decoder_layerinfo
        self.epochs = epochs
        self.batch_size = batch_size
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.n_centroid = n_centroid
        self.trainset = trainset
        self.testset = testset
        self.trainy = trainy
        self.testy = testy

        # initializing model
        self.x = tf.placeholder(tf.float32, shape=[None, self.x_dim])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.z_mu, self.z_logvar, self.encodew, self.encodeb = self.build_encoder(self.x)
        self.z_sample = reparameterization(self.z_mu, self.z_logvar)
        self.x_decoded_logit, self.x_decoded_prob, self.decodew, self.decodeb = self.build_decoder(self.z_sample)
        self.x_generate = self.generate(self.z)
        self.gmm = mixture.GaussianMixture(n_components=n_centroid, covariance_type='diag')
        self.gmm_variable = [tf.Variable(np.zeros([self.n_centroid, self.z_dim]), dtype=tf.float32, name="gmm_mean"),
                             tf.Variable(np.zeros([self.n_centroid, self.z_dim]), dtype=tf.float32, name="gmm_logcov"),
                             tf.Variable(np.ones([self.n_centroid]) / self.n_centroid, dtype=tf.float32, name="gmm_w")]
        self.gamma, _, _, _ = self.get_gamma(self.z_sample)
        self.vadeloss, self.reconstructionerror, self.kl = self.vade_loss()

        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(lr, self.global_step, 10 * int(self.trainset.shape[0] / self.batch_size),
                                             0.9, staircase=True)

        self.global_step_gmm = tf.Variable(0, trainable=False)
        self.lr_gmm = tf.train.exponential_decay(
            lr, self.global_step_gmm, 10 * int(self.trainset.shape[0] / self.batch_size), 0.9, staircase=True)
        if not os.path.exists(r'.\pretrain'):
            os.makedirs(r'.\pretrain')
        if not os.path.exists(r'.\trained_model'):
            os.makedirs(r'.\trained_model')
        if not os.path.exists(r".\Output"):
            os.makedirs(r".\Output")

    def build_encoder(self, x):
        last_dim = self.x_dim
        w = [tf.Variable(tf.zeros([self.x_dim]), dtype=tf.float32) for i in range(len(self.encoder_layerinfo))]
        b = [tf.Variable(tf.zeros([self.x_dim]), dtype=tf.float32) for i in range(len(self.encoder_layerinfo))]
        for i in range(len(self.encoder_layerinfo)):
            dim = self.encoder_layerinfo[i]
            w[i] = tf.Variable(weight_init([last_dim, dim]), dtype=tf.float32, name="encoder_w_{0}".format(i))
            b[i] = tf.Variable(bias_init([last_dim, dim]), dtype=tf.float32, name="encoder_b_{0}".format(i))
            x = tf.nn.relu(tf.matmul(x, w[i]) + b[i])
            last_dim = dim
        w1 = tf.Variable(weight_init([last_dim, self.z_dim]), dtype=tf.float32, name="encoder_w_mu")
        b1 = tf.Variable(bias_init([last_dim, self.z_dim]), dtype=tf.float32, name="encoder_b_mu")
        mu = tf.matmul(x, w1) + b1
        w.append(w1)
        b.append(b1)
        w2 = tf.Variable(weight_init([last_dim, self.z_dim]), dtype=tf.float32, name="encoder_w_logvar")
        b2 = tf.Variable(bias_init([last_dim, self.z_dim]), dtype=tf.float32, name="encoder_b_logvar")
        logvar = tf.matmul(x, w2) + b2
        w.append(w2)
        b.append(b2)
        return mu, logvar, w, b

    def build_decoder(self, z):
        last_dim = self.z_dim
        w = [tf.Variable(tf.zeros([self.x_dim]), dtype=tf.float32) for i in range(len(self.decoder_layerinfo))]
        b = [tf.Variable(tf.zeros([self.x_dim]), dtype=tf.float32) for i in range(len(self.decoder_layerinfo))]
        for i in range(len(self.decoder_layerinfo)):
            dim = self.decoder_layerinfo[i]
            w[i] = tf.Variable(weight_init([last_dim, dim]), dtype=tf.float32, name="decoder_w_{0}".format(i))
            b[i] = tf.Variable(bias_init([last_dim, dim]), dtype=tf.float32, name="decoder_b_{0}".format(i))
            z = tf.nn.relu(tf.matmul(z, w[i]) + b[i])
            last_dim = dim
        w1 = tf.Variable(weight_init([last_dim, self.x_dim]), dtype=tf.float32, name="decoder_w_out")
        b1 = tf.Variable(bias_init([last_dim, self.x_dim]), dtype=tf.float32, name="decoder_w_out")
        x_decoded_logit = tf.matmul(z, w1) + b1
        x_decoded_prob = tf.nn.sigmoid(x_decoded_logit)
        w.append(w1)
        b.append(b1)
        return x_decoded_logit, x_decoded_prob, w, b

    def get_gamma(self, z):
        temp_z = repeat(z, self.n_centroid, axis=1)
        gmm_mean = self.gmm_variable[0]
        gmm_logcov = self.gmm_variable[1]
        gmm_weights = self.gmm_variable[2]  # /tf.reduce_sum(self.gmm_variable[2])

        p_z_c = tf.exp(tf.log(gmm_weights) +
                       tf.reduce_sum(-0.5 * (tf.log(2 * np.pi)+gmm_logcov) -
                                     tf.square(temp_z - gmm_mean) / (2 * tf.exp(gmm_logcov)), axis=2)
                       ) + 1e-10

        gamma = p_z_c / tf.reduce_sum(p_z_c, axis=1, keepdims=True)
        return gamma, gmm_mean, gmm_logcov, gmm_weights

    def vade_loss(self):
        gamma, gmm_mean, gmm_logcov, gmm_weights = self.get_gamma(self.z_sample)

        temp_z_var = repeat(tf.exp(self.z_logvar), self.n_centroid, axis=1)
        temp_z_mu = repeat(self.z_mu, self.n_centroid, axis=1)

        reconstructionerror = self.x_dim * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.x_decoded_logit, labels=self.x), axis=1
        )
        kl = 0.5 * tf.reduce_sum(
            gamma * tf.reduce_sum(
                gmm_logcov+temp_z_var/tf.exp(gmm_logcov)+tf.square(temp_z_mu-gmm_mean)/tf.exp(gmm_logcov), axis=-1),
            axis=-1
        ) - tf.reduce_sum(gamma*tf.log(gmm_weights/gamma), axis=-1) - 0.5*tf.reduce_sum(1+self.z_logvar, axis=-1)
        vaeloss = tf.reduce_mean(reconstructionerror + kl)
        return vaeloss, tf.reduce_mean(reconstructionerror), tf.reduce_mean(kl)

    def get_posterior(self, z, u, cov, theta):
        z_m = np.repeat(z, self.n_centroid, axis=0)
        posterior = np.exp(
            np.sum((np.log(theta) - 0.5 * np.log(2 * np.pi * cov) - np.square(z_m - u) / (2 * cov)), axis=1)
        )
        return posterior / np.sum(posterior, axis=-1, keepdims=True)

    def generate(self, z):
        last_dim = self.z_dim
        w = self.decodew
        b = self.decodeb
        for i in range(len(self.decoder_layerinfo)):
            z = tf.nn.relu(tf.matmul(z, w[i]) + b[i])
        x_decoded_logit = tf.matmul(z, w[-1]) + b[-1]
        x_decoded_prob = tf.nn.sigmoid(x_decoded_logit)
        return x_decoded_prob

    def test_generateplot(self, sess, epoch, acc, acc_init):
        n = 15  # figure with 15x15 digits
        digit_size = 28
        figure = np.zeros((digit_size * self.n_centroid, digit_size * n))
        g_u = sess.run(self.gmm_variable[0])
        g_cov = sess.run(self.gmm_variable[1])
        g_weight = sess.run(self.gmm_variable[2])
        for i in range(self.n_centroid):
            u = g_u[i]
            cov = g_cov[i]
            count = 0
            while count < n:
                z_generate_i = np.random.multivariate_normal(u, np.diag(cov), (1,))
                posterior = self.get_posterior(z_generate_i, g_u, g_cov, g_weight)[i]
                if posterior > 0.999:
                    x_generate_i = sess.run(self.x_generate, feed_dict={self.z: z_generate_i})
                    figure[i * digit_size:(i + 1) * digit_size, count * digit_size:(count + 1) * digit_size] = \
                        np.reshape(x_generate_i, newshape=[digit_size, digit_size])
                    count += 1
        plt.figure(figsize=(10, 10))
        plt.title("Generation {0}D".format(self.z_dim))
        plt.imshow(figure, cmap='Greys_r')
        plt.savefig(r".\Output\Acc_init{:.2f}_Acc{:.2f}%\Generatioin_Epoch{:}.png".format(
            acc_init * 100, acc * 100, epoch))

    def get_gamma_value(self, z, sess):
        temp_z = np.expand_dims(z, axis=1).repeat(self.n_centroid, axis=1)
        gmm_mean = sess.run(self.gmm_variable[0])
        gmm_cov = sess.run(self.gmm_variable[1])
        gmm_weights = sess.run(self.gmm_variable[2])  # /tf.reduce_sum(self.gmm_variable[2])

        p_z_c = np.exp(np.log(gmm_weights) +
                       np.sum(-0.5 * np.log(2 * np.pi * gmm_cov) -
                                     np.square(temp_z - gmm_mean) / (2 * gmm_cov), axis=2)
                       ) + 1e-10

        gamma = p_z_c / np.sum(p_z_c, axis=1, keepdims=True)
        return gamma

    def train(self):

        # pretrain using AE
        # encoder
        h = tf.nn.relu(tf.matmul(self.x, self.encodew[0]) + self.encodeb[0])
        for i in range(1, len(self.encodew) - 2):
            h = tf.nn.relu(tf.matmul(h, self.encodew[i]) + self.encodeb[i])
        mu = tf.matmul(h, self.encodew[-2]) + self.encodeb[-2]
        h = mu
        # decoder
        for i in range(0, len(self.decodew) - 1):
            h = tf.nn.relu(tf.matmul(h, self.decodew[i]) + self.decodeb[i])
        x_decoded_logit = tf.matmul(h, self.decodew[-1]) + self.decodeb[-1]
        x_decoded_prob = tf.nn.sigmoid(x_decoded_logit)

        vae_reconstruction_loss = tf.losses.mean_squared_error(predictions=x_decoded_prob, labels=self.x)
        loss = tf.reduce_mean(vae_reconstruction_loss)

        solver = tf.train.AdamOptimizer(0.0005).minimize(loss)
        solver_nn = tf.train.AdamOptimizer(self.lr).minimize(
            self.vadeloss, global_step=self.global_step,
            var_list=self.encodew+self.encodeb+self.decodew+self.decodeb
        )
        solver_gmm = tf.train.AdamOptimizer(self.lr_gmm).minimize(
            self.vadeloss, global_step=self.global_step_gmm, var_list=self.gmm_variable)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        acc_init = 0
        mode = 'pretrain'
        if mode == 'pretrain':
            print("Befor load pretrain model: ")
            print(sess.run(loss, feed_dict={self.x: self.testset}))
            if os.path.exists(pretrainmodel + ".meta"):
                saver.restore(sess, pretrainmodel)
                print("After load pretrain model: ")
                print(sess.run(loss, feed_dict={self.x: self.testset}))
            else:
                total_count = self.trainset.shape[0]
                batch_size = self.batch_size
                batch_count = int(total_count / batch_size - 1) + 1
                for epoch in range(20):
                    for it in range(batch_count):
                        x_batch = self.trainset[it * batch_size: min((it + 1) * batch_size, total_count)]
                        _, _ = sess.run([solver, loss], feed_dict={self.x: x_batch})
                    temp_z, temp_loss = sess.run([self.z_mu, loss], feed_dict={self.x: self.trainset})
                    print("Pretraining Epoch {0}\ttest_loss:{1}".format(
                        epoch + 1, temp_loss))
                print("After load pretrain model: ")
                print(sess.run(loss, feed_dict={self.x: self.testset}))
                saver.save(sess, pretrainmodel)
            op = tf.assign(self.encodew[-1], self.encodew[-2])
            sess.run(op)
            op = tf.assign(self.encodeb[-1], self.encodeb[-2])
            sess.run(op)
            temp_z, temp_loss = sess.run([self.z_mu, loss], feed_dict={self.x: self.trainset})
            pre = self.gmm.fit_predict(temp_z)
            acc_init = cluster_acc(pre, self.trainy)[0]
            print("Pretraining acc_gmm: {0}, TrainSet Loss:{1}".format(acc_init, temp_loss))
            op1 = tf.assign(self.gmm_variable[0], self.gmm.means_)
            op2 = tf.assign(self.gmm_variable[1], self.gmm.covariances_)
            op3 = tf.assign(self.gmm_variable[2], self.gmm.weights_)
            sess.run([op1, op2, op3])

        total_count = self.trainset.shape[0]
        batch_size = self.batch_size
        batch_count = int(total_count / batch_size - 1) + 1

        acc = 0
        for epoch in range(self.epochs):
            # batch start
            for it in range(batch_count):
                x_batch = self.trainset[it * batch_size: min((it + 1) * batch_size, total_count)]
                sess.run([solver_nn, solver_gmm], feed_dict={self.x: x_batch})
            # Epoch Begin
            gamma, temp_z, trainloss = sess.run(
                [self.gamma, self.z_sample, self.vadeloss], feed_dict={self.x: self.trainset}
            )
            temp_c = get_class(gamma)
            acc = cluster_acc(temp_c, self.trainy)[0]
            print("Training Epoch {0}\tacc_pcz:{1}\ttrainloss:{2}\tLR:{3}".format(
                epoch + 1, acc, trainloss, sess.run(self.lr)))

        x_test_sub = np.reshape(self.testset[0:15 * 15], newshape=(15, 15, 28, 28))
        x_test_vae = np.reshape(
            sess.run(self.x_decoded_prob, feed_dict={self.x: self.testset[0:15 * 15]}), newshape=(15, 15, 28, 28))

        n = 15  # figure with 15x15 digits
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n * 2))
        for i in range(0, n):
            for j in range(0, n * 2, 2):
                figure[i * digit_size:(i + 1) * digit_size, j * digit_size:(j + 1) * digit_size] = x_test_sub[
                    i, int(j / 2)]
                figure[i * digit_size:(i + 1) * digit_size, (j + 1) * digit_size:(j + 2) * digit_size] = x_test_vae[
                    i, int(j / 2)]
        plt.figure(figsize=(10, 10))
        plt.title("Latent Variable {0}D".format(self.z_dim))
        plt.imshow(figure, cmap='Greys_r')
        if not os.path.exists(r".\Output\Acc_init{:.2f}_Acc{:.2f}%".format(acc_init*100, acc*100)):
            os.makedirs(r".\Output\Acc_init{:.2f}_Acc{:.2f}%".format(acc_init*100, acc*100))
        plt.savefig(r".\Output\Acc_init{:.2f}_Acc{:.2f}%\Reconstruct.png".format(acc_init*100, acc*100))

        self.test_generateplot(sess, epoch + 1, acc, acc_init)

        shutil.move(r".\pretrain", r".\Output\Acc_init{:.2f}_Acc{:.2f}%\pretrain".format(acc_init*100, acc*100))
