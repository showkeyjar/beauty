import os
import time
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from matplotlib.ticker import FormatStrFormatter
# from tensorflow.keras.utils import multi_gpu_model

# import data loader
from data import load

# import computational graphs
from graphs import IICGraph, KERNEL_INIT, BIAS_INIT

# import utility functions
from utils import unsupervised_labels, save_performance

# plot settings
DPI = 600
# gpu_list = '0,1,2,3'
# os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
# gpu_num = len(gpu_list.split(','))


class ClusterIIC(object):
    def __init__(self, num_classes, learning_rate, num_repeats, save_dir=None):
        """
        :param num_classes: number of classes
        :param learning_rate: gradient step size
        :param num_repeats: number of data repeats for x and g(x), used to up-sample
        """
        # save configuration
        self.k_A = 5 * num_classes
        self.num_A_sub_heads = 1
        self.k_B = num_classes
        self.num_B_sub_heads = 5
        self.num_repeats = num_repeats

        # initialize losses
        self.loss_A = None
        self.loss_B = None
        self.losses = []

        # initialize outputs
        self.y_hats = None

        # initialize optimizer
        self.is_training = tf.compat.v1.placeholder(tf.bool)
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.opt = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        self.train_ops = []

        # initialize performance dictionary
        self.perf = None
        self.save_dir = save_dir

        # configure performance plotting
        self.fig_learn, self.ax_learn = plt.subplots(1, 2)

    def __iic_loss(self, pi_x, pi_gx):

        # up-sample non-perturbed to match the number of repeat samples
        pi_x = tf.tile(pi_x, [self.num_repeats] + [1] * len(pi_x.shape.as_list()[1:]))

        # get K
        k = pi_x.shape.as_list()[1]

        # compute P
        p = tf.transpose(pi_x) @ pi_gx

        # enforce symmetry
        p = (p + tf.transpose(p)) / 2

        # enforce minimum value
        p = tf.clip_by_value(p, clip_value_min=1e-6, clip_value_max=tf.float32.max)

        # normalize
        p /= tf.reduce_sum(p)

        # get marginals
        pi = tf.broadcast_to(tf.reshape(tf.reduce_sum(p, axis=0), (k, 1)), (k, k))
        pj = tf.broadcast_to(tf.reshape(tf.reduce_sum(p, axis=1), (1, k)), (k, k))

        # complete the loss
        loss = -tf.reduce_sum(p * (tf.math.log(p) - tf.math.log(pi) - tf.math.log(pj)))

        return loss

    @staticmethod
    def __head_out(z, k, name):

        # construct a new head that operates on the model's output for x
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            phi = tf.layers.dense(
                inputs=z,
                units=k,
                activation=tf.nn.softmax,
                use_bias=True,
                kernel_initializer=KERNEL_INIT,
                bias_initializer=BIAS_INIT)

        return phi

    def __head_loss(self, z_x, z_gx, k, num_sub_heads, head):

        # loop over the number of sub-heads
        loss = tf.constant(0, dtype=tf.float32)
        for i in range(num_sub_heads):

            # run the model
            pi_x = self.__head_out(z_x, k, name=head + str(i + 1))
            num_vars = len(tf.compat.v1.global_variables())
            pi_gx = self.__head_out(z_gx, k, name=head + str(i + 1))
            assert num_vars == len(tf.compat.v1.global_variables())

            # accumulate the clustering loss
            loss += self.__iic_loss(pi_x, pi_gx)

        # take the average
        if num_sub_heads > 0:
            loss /= num_sub_heads

        return loss

    def __build(self, x, gx, graph):

        # run the graph
        z_x = graph.evaluate(x, is_training=self.is_training)
        num_vars = len(tf.compat.v1.global_variables())
        z_gx = graph.evaluate(gx, is_training=self.is_training)
        assert num_vars == len(tf.compat.v1.global_variables())

        # construct losses
        self.loss_A = self.__head_loss(z_x, z_gx, self.k_A, self.num_A_sub_heads, 'A')
        self.loss_B = self.__head_loss(z_x, z_gx, self.k_B, self.num_B_sub_heads, 'B')
        self.losses = [self.loss_A, self.loss_B]

        # set alternating training operations
        self.train_ops.append(tf.contrib.layers.optimize_loss(loss=self.loss_A,
                                                              global_step=self.global_step,
                                                              learning_rate=self.learning_rate,
                                                              optimizer=self.opt,
                                                              summaries=['loss', 'gradients']))
        self.train_ops.append(tf.contrib.layers.optimize_loss(loss=self.loss_B,
                                                              global_step=self.global_step,
                                                              learning_rate=self.learning_rate,
                                                              optimizer=self.opt,
                                                              summaries=['loss', 'gradients']))

        # initialize outputs outputs
        self.y_hats = [tf.argmax(self.__head_out(z_x, self.k_B, 'B' + str(i + 1)), axis=1)
                       for i in range(self.num_B_sub_heads)]

    def __performance_dictionary_init(self, num_epochs):
        """
        :param num_epochs: maximum number of epochs (used to size buffers)
        :return: None
        """
        # initialize performance dictionary
        self.perf = dict()

        # loss terms
        self.perf.update({'loss_A': np.zeros(num_epochs)})
        self.perf.update({'loss_B': np.zeros(num_epochs)})

        # classification error
        self.perf.update({'class_err_min': np.zeros(num_epochs)})
        self.perf.update({'class_err_avg': np.zeros(num_epochs)})
        self.perf.update({'class_err_max': np.zeros(num_epochs)})

    def __classification_accuracy(self, sess, iter_init, idx, y_ph=None):
        """
        :param sess: TensorFlow session
        :param iter_init: TensorFlow data iterator initializer associated
        :param idx: insertion index (i.e. epoch - 1)
        :param y_ph: TensorFlow placeholder for unseen labels
        :return: None
        """
        if self.perf is None or y_ph is None:
            return

        # initialize results
        y = np.zeros([0, 1])
        y_hats = [np.zeros([0, 1])] * self.num_B_sub_heads

        # initialize unsupervised data iterator
        sess.run(iter_init)

        # loop over the batches within the unsupervised data iterator
        print('Evaluating classification accuracy... ')
        while True:
            try:
                # grab the results
                results = sess.run([self.y_hats, y_ph], feed_dict={self.is_training: False})

                # load metrics
                for i in range(self.num_B_sub_heads):
                    y_hats[i] = np.concatenate((y_hats[i], np.expand_dims(results[0][i], axis=1)))
                if y_ph is not None:
                    y = np.concatenate((y, np.expand_dims(results[1], axis=1)))

                # _, ax = plt.subplots(2, 10)
                # i_rand = np.random.choice(results[3].shape[0], 10)
                # for i in range(10):
                #     ax[0, i].imshow(results[3][i_rand[i]][:, :, 0], origin='upper', vmin=0, vmax=1)
                #     ax[0, i].set_xticks([])
                #     ax[0, i].set_yticks([])
                #     ax[1, i].imshow(results[4][i_rand[i]][:, :, 0], origin='upper', vmin=0, vmax=1)
                #     ax[1, i].set_xticks([])
                #     ax[1, i].set_yticks([])
                # plt.show()

            # iterator will throw this error when its out of data
            except tf.errors.OutOfRangeError:
                break

        # compute classification accuracy
        if y_ph is not None:
            class_errors = [unsupervised_labels(y, y_hats[i], self.k_B, self.k_B)
                            for i in range(self.num_B_sub_heads)]
            self.perf['class_err_min'][idx] = np.min(class_errors)
            self.perf['class_err_avg'][idx] = np.mean(class_errors)
            self.perf['class_err_max'][idx] = np.max(class_errors)

        # metrics are done
        print('Done')

    def plot_learning_curve(self, epoch):
        """
        :param epoch: epoch number
        :return: None
        """
        # generate epoch numbers
        t = np.arange(1, epoch + 1)

        # colors
        c = {'Head A': '#1f77b4', 'Head B': '#ff7f0e'}

        # plot the loss
        self.ax_learn[0].clear()
        self.ax_learn[0].set_title('Loss')
        self.ax_learn[0].plot(t, self.perf['loss_A'][:epoch], label='Head A', color=c['Head A'])
        self.ax_learn[0].plot(t, self.perf['loss_B'][:epoch], label='Head B', color=c['Head B'])
        self.ax_learn[0].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        self.ax_learn[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # plot the classification error
        self.ax_learn[1].clear()
        self.ax_learn[1].set_title('Class. Error (Min, Avg, Max)')
        self.ax_learn[1].plot(t, self.perf['class_err_avg'][:epoch], color=c['Head B'])
        self.ax_learn[1].fill_between(t,
                                      self.perf['class_err_min'][:epoch],
                                      self.perf['class_err_max'][:epoch],
                                      facecolor=c['Head B'], alpha=0.5)
        self.ax_learn[1].plot(t, self.perf['class_err_avg'][:epoch], color=c['Head B'])
        self.ax_learn[1].fill_between(t,
                                      self.perf['class_err_min'][:epoch],
                                      self.perf['class_err_max'][:epoch],
                                      facecolor=c['Head B'], alpha=0.5)
        self.ax_learn[1].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        self.ax_learn[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # make the legend
        self.ax_learn[1].legend(handles=[patches.Patch(color=val, label=key) for key, val in c.items()],
                                ncol=len(c),
                                bbox_to_anchor=(0.35, -0.06))

        # eliminate those pesky margins
        self.fig_learn.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.95, wspace=0.25, hspace=0.3)

    def train(self, graph, train_set, test_set, num_epochs, early_stop_buffer=15):
        """
        :param graph: the computational graph
        :param train_set: TensorFlow Dataset object that corresponds to training data
        :param test_set: TensorFlow Dataset object that corresponds to validation data
        :param num_epochs: number of epochs
        :param early_stop_buffer: early stop look-ahead distance (in epochs)
        :return: None
        """
        # construct iterator
        iterator = tf.compat.v1.data.make_initializable_iterator(train_set)
        x, gx, y = iterator.get_next().values()

        # construct initialization operations
        train_iter_init = iterator.make_initializer(train_set)
        test_iter_init = iterator.make_initializer(test_set)

        # build the model using the supplied computational graph
        self.__build(x, gx, graph)

        # initialize performance dictionary
        self.__performance_dictionary_init(num_epochs)

        # todo 未验证多GPU这样写是否可行
        # 参考: https://blog.csdn.net/bcfd_yundou/article/details/112600549
        # self = multi_gpu_model(self, gpus=gpu_num)

        # start a monitored session
        cfg = tf.compat.v1.ConfigProto()
        cfg.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=cfg) as sess:
            # initialize model variables
            sess.run(tf.global_variables_initializer())
            # loop over the number of epochs
            for i in range(num_epochs):

                # start timer
                start = time.time()

                # get epoch number
                epoch = i + 1

                # get training operation
                i_train = i % len(self.train_ops)

                # initialize epoch iterator
                sess.run(train_iter_init)

                # loop over the batches
                loss_A = []
                loss_B = []
                while True:
                    try:
                        # run training and losses
                        losses = sess.run([self.train_ops[i_train]] + [self.losses],
                                          feed_dict={self.is_training: True})[-1]

                        # load metrics
                        loss_A.append(losses[0])
                        loss_B.append(losses[1])

                        if np.isnan(losses).any():
                            print('\n NaN whelp!')
                            return

                        # print update
                        print('\rEpoch {:d}, Loss = {:.4f}'.format(epoch, losses[i_train]), end='')

                    # iterator will throw this error when its out of data
                    except tf.errors.OutOfRangeError:
                        break

                # new line
                print('')

                # save averaged training performance
                self.perf['loss_A'][i] = np.mean(loss_A)
                self.perf['loss_B'][i] = np.mean(loss_B)

                # get classification performance
                self.__classification_accuracy(sess, test_iter_init, i, y)

                # plot learning curve
                self.plot_learning_curve(epoch)

                # pause for plot drawing if we aren't saving
                if self.save_dir is None:
                    plt.pause(0.05)

                # print time for epoch
                stop = time.time()
                print('Time for Epoch = {:f}'.format(stop - start))

                # early stop check
                # i_best_elbo = np.argmin(self.perf['loss']['test'][:epoch])
                # i_best_class = np.argmin(self.perf['class_err']['test'][:epoch])
                # epochs_since_improvement = min(i - i_best_elbo, i - i_best_class)
                # print('Early stop checks: {:d} / {:d}\n'.format(epochs_since_improvement, early_stop_buffer))
                # if epochs_since_improvement >= early_stop_buffer:
                #     break

        # save the performance
        save_performance(self.perf, epoch, self.save_dir)


if __name__ == '__main__':
    # pick a data set
    DATA_SET = 'beauty'

    # define splits
    DS_CONFIG = {
        # mnist data set parameters
        'beauty': {
            'batch_size': 32,
            'num_repeats': 5,
            'mdl_input_dims': [80, 96, 3]}
    }

    # load the data set
    TRAIN_SET, TEST_SET = load(**DS_CONFIG[DATA_SET])

    # configure the common model elements
    MDL_CONFIG = {
        # mist hyper-parameters
        'beauty': {
            # 这里先拍脑袋决定12种不同的皮肤
            'num_classes': 12,
            'learning_rate': 1e-4,
            'num_repeats': DS_CONFIG[DATA_SET]['num_repeats'],
            'save_dir': None},
    }

    # declare the model
    mdl = ClusterIIC(**MDL_CONFIG[DATA_SET])

    # train the model
    mdl.train(IICGraph(config='B', batch_norm=True, fan_out_init=64), TRAIN_SET, TEST_SET, num_epochs=600)

    print('All done!')
    plt.show()
