# -*- coding:utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt

file_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("data1/test.gif")
)
reader = tf.WholeFileReader()
key,value = reader.read(file_queue)
image = tf.image.decode_gif(value)

kernel = tf.constant([
    [[[-1.,]],[[-1.]],[[-1.]]],
    [[[-1.,]],[[8.]],[[-1.]]],
    [[[-1.,]],[[-1.]],[[-1.]]],
])

coord = tf.train.Coordinator()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    print(sess.run(kernel).shape)
    sess.run(init)
    threads = tf.train.start_queue_runners(coord=coord)
    image_tensor = tf.image.rgb_to_grayscale(sess.run([image])[0])
    print(sess.run(image_tensor).shape)

    image_convoluted_tensor = tf.nn.conv2d(tf.cast(image_tensor,tf.float32)
                                           ,kernel,[1,1,1,1],'SAME')



