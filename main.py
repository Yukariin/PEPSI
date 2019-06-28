import os
import random
import time

import cv2
import numpy as np
import tensorflow as tf
import scipy.misc as sci

import module as mm
import ops as op
import Read_Image_List as ri


HEIGHT = 256
WIDTH = 256
BATCH_SIZE = 8

name_f, num_f = ri.read_labeled_image_list('/content/drive/My Drive/DEC/data/train.flist')
total_batch = int(num_f / BATCH_SIZE)

model_path = './model/v1'

restore = True
restore_point = 10000
Checkpoint = model_path + '/cVG iter ' + str(restore_point) + '/'
WeightName = Checkpoint + 'Train_' + str(restore_point) + '.meta'

if not restore:
    restore_point = 0

saving_iter = 10000
Max_iter = 1000000

# ------- variables

X = tf.placeholder(tf.float32, [BATCH_SIZE, HEIGHT, WIDTH, 3])
Y = tf.placeholder(tf.float32, [BATCH_SIZE, HEIGHT, WIDTH, 3])

MASK = tf.placeholder(tf.float32, [BATCH_SIZE, HEIGHT, WIDTH, 3])
IT = tf.placeholder(tf.float32)

# ------- structure

input = tf.concat([X, MASK], 3)

vec_en = mm.encoder(input, reuse=False, name='G_en')

vec_con = mm.contextual_block(vec_en, vec_en, MASK, 3, 50.0, 'CB1', stride=1)

I_co = mm.decoder(vec_en, reuse=False, name='G_de')
I_ge = mm.decoder(vec_con, reuse=True, name='G_de')

image_result = I_ge * (1-MASK) + Y*MASK

D_real_red = mm.discriminator_red(Y, reuse=False, name='disc_red')
D_fake_red = mm.discriminator_red(image_result, reuse=True, name='disc_red')

# ------- Loss

Loss_D_red = tf.reduce_mean(tf.nn.relu(1+D_fake_red)) + tf.reduce_mean(tf.nn.relu(1-D_real_red))

Loss_D = Loss_D_red

Loss_gan_red = -tf.reduce_mean(D_fake_red)

Loss_gan = Loss_gan_red

Loss_s_re = tf.reduce_mean(tf.abs(I_ge - Y))
Loss_hat = tf.reduce_mean(tf.abs(I_co - Y))

A = tf.image.rgb_to_yuv((image_result+1)/2.0)
A_Y = tf.to_int32(A[:, :, :, 0:1]*255.0)

B = tf.image.rgb_to_yuv((Y+1)/2.0)
B_Y = tf.to_int32(B[:, :, :, 0:1]*255.0)

ssim = tf.reduce_mean(tf.image.ssim(A_Y, B_Y, 255.0))

alpha = IT/Max_iter

Loss_G = 0.1*Loss_gan + 10*Loss_s_re + 5*(1-alpha) * Loss_hat

# --------------------- variable & optimizer

var_D = [v for v in tf.global_variables() if v.name.startswith('disc_red')]
var_G = [v for v in tf.global_variables() if v.name.startswith('G_en') or v.name.startswith('G_de') or v.name.startswith('CB1')]

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
    optimize_D = tf.train.AdamOptimizer(learning_rate=0.0004, beta1=0.5, beta2=0.9).minimize(Loss_D, var_list=var_D)
    optimize_G = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(Loss_G, var_list=var_G)

# --------- Run

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = False

sess = tf.Session(config=config)

init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()

if restore:
    print('Weight Restoring.....')
    Restore = tf.train.import_meta_graph(WeightName)
    Restore.restore(sess, tf.train.latest_checkpoint(Checkpoint))
    print('Weight Restoring Finish!')

start_time = time.time()
for iter_count in range(restore_point, Max_iter + 1):

    i = iter_count % total_batch
    e = iter_count // total_batch

    if i == 0:
        np.random.shuffle(name_f)

    data_g = ri.MakeImageBlock(name_f, HEIGHT, WIDTH, i, BATCH_SIZE)

    data_temp = 255.0 * ((data_g + 1) / 2.0)

    mask = op.ff_mask_batch(HEIGHT, BATCH_SIZE, 50, 20, 3.14, 5, 15)

    data_m = data_temp * mask

    data_m = (data_m / 255.0) * 2.0 - 1

    _, Loss1 = sess.run([optimize_D, Loss_D], feed_dict={X: data_m, Y: data_g, MASK: mask})
    _, Loss2, Loss3 = sess.run([optimize_G, Loss_G, Loss_s_re], feed_dict={X: data_m, Y: data_g, MASK: mask, IT: iter_count})

    if iter_count % 100 == 0:
        consume_time = time.time() - start_time
        print('%d     Epoch : %d       D Loss = %.5f    G Loss = %.5f    Recon Loss = %.5f     time = %.4f' % (iter_count, e, Loss1, Loss2, Loss3, consume_time))
        start_time = time.time()

    if iter_count % saving_iter == 0:

        print('SAVING MODEL')
        Temp = model_path + '/cVG iter %s/' % iter_count

        if not os.path.exists(Temp):
            os.makedirs(Temp)

        SaveName = (Temp + 'Train_%s' % (iter_count))
        saver.save(sess, SaveName)
        print('SAVING MODEL Finish')
