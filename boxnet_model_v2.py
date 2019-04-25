
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training):
  inputs = tf.layers.batch_normalization(inputs=inputs, 
                                         axis=1,
                                         momentum=_BATCH_NORM_DECAY, 
                                         epsilon=_BATCH_NORM_EPSILON, 
                                         center=True, 
                                         scale=True, 
                                         training=is_training, 
                                         fused=True)
  #print(is_training)
  inputs = tf.nn.leaky_relu(inputs, alpha=0.4)
  return inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, dilation):

  return tf.layers.conv2d(inputs=inputs, 
                          filters=filters, 
                          kernel_size=kernel_size, 
                          strides=strides,
                          dilation_rate=dilation,
                          padding='SAME', 
                          use_bias=False,
                          data_format='channels_first',
                          kernel_initializer=tf.variance_scaling_initializer())


def building_block_residual(inputs, filters, is_training, projection_shortcut, strides, dilation):

  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides, dilation=dilation)

  inputs = batch_norm_relu(inputs, is_training)
  inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=1, dilation=1)

  return inputs + shortcut


def building_block(inputs, filters, is_training, projection_shortcut, strides, dilation):

  inputs = batch_norm_relu(inputs, is_training)
  inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides, dilation=dilation)

  inputs = batch_norm_relu(inputs, is_training)
  inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=1, dilation=1)

  return inputs


def block_layer(inputs, filters, block_fn, blocks, strides, dilation, is_training, name):

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1, strides=strides, dilation=1)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides, dilation)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, is_training, None, 1, dilation)

  return tf.identity(inputs, name)
  
def upsample(inputs):

  return tf.keras.layers.UpSampling2D(size=[2, 2], data_format='channels_first')(inputs)


def boxnet_v2_generator(num_classes):

  def model(inputs, is_training):
  
    training_batchnorm = is_training
  
    # input is channels_last, but we need channels_first
    inputs = tf.transpose(inputs, [0, 3, 1, 2])
    
    n_blocks = 3

    in0 = conv2d_fixed_padding(inputs=inputs, filters=32, kernel_size=5, strides=1, dilation=1)                                                                                  # 256
    
    in1 = block_layer(inputs=in0, filters=32, block_fn=building_block_residual, blocks=n_blocks, strides=1, dilation=1, is_training=training_batchnorm, name='in_layer1')        # 256
    in2 = block_layer(inputs=in1, filters=32, block_fn=building_block_residual, blocks=n_blocks, strides=2, dilation=1, is_training=training_batchnorm, name='in_layer2')        # 128
    in3 = block_layer(inputs=in2, filters=64, block_fn=building_block_residual, blocks=n_blocks, strides=2, dilation=1, is_training=training_batchnorm, name='in_layer3')        # 64
    in4 = block_layer(inputs=in3, filters=128, block_fn=building_block_residual, blocks=n_blocks, strides=2, dilation=1, is_training=training_batchnorm, name='in_layer4')       # 32
    in5 = block_layer(inputs=in4, filters=256, block_fn=building_block_residual, blocks=n_blocks, strides=2, dilation=1, is_training=training_batchnorm, name='in_layer5')       # 16
    in6 = block_layer(inputs=in5, filters=256, block_fn=building_block_residual, blocks=1, strides=2, dilation=1, is_training=training_batchnorm, name='in_layer6')       # 8
    in7 = block_layer(inputs=in6, filters=256, block_fn=building_block_residual, blocks=1, strides=2, dilation=1, is_training=training_batchnorm, name='in_layer7')       # 4
    
    out6 = upsample(in7)
    out6 = tf.concat([in6, out6], 1)
    out6 = block_layer(inputs=out6, filters=128, block_fn=building_block_residual, blocks=1, strides=1, dilation=1, is_training=training_batchnorm, name='out_layer4')        # 8
    out5 = upsample(out6)
    out5 = tf.concat([in5, out5], 1)
    out5 = block_layer(inputs=out5, filters=128, block_fn=building_block_residual, blocks=1, strides=1, dilation=1, is_training=training_batchnorm, name='out_layer4')        # 16
    out4 = upsample(in5)
    out4 = tf.concat([in4, out4], 1)
    out4 = block_layer(inputs=out4, filters=128, block_fn=building_block_residual, blocks=3, strides=1, dilation=1, is_training=training_batchnorm, name='out_layer4')        # 32
    out3 = upsample(out4)
    out3 = tf.concat([in3, out3], 1)
    out3 = block_layer(inputs=out3, filters=64, block_fn=building_block_residual, blocks=3, strides=1, dilation=1, is_training=training_batchnorm, name='out_layer3')  # 64
    out2 = upsample(out3)
    out2 = tf.concat([in2, out2], 1)
    out2 = block_layer(inputs=out2, filters=32, block_fn=building_block_residual, blocks=3, strides=1, dilation=1, is_training=training_batchnorm, name='out_layer2')  # 128
    out1 = upsample(out2)
    out1 = tf.concat([in1, out1], 1)
    out1 = block_layer(inputs=out1, filters=16, block_fn=building_block_residual, blocks=4, strides=1, dilation=1, is_training=training_batchnorm, name='out_layer1')  # 256


    #inputs = batch_norm_relu(combined0, training_batchnorm)
    inputs = tf.layers.conv2d(inputs=out1, 
                              filters=num_classes, 
                              kernel_size=1, 
                              strides=1,
                              dilation_rate=1,
                              padding='SAME', 
                              use_bias=True,
                              data_format='channels_first',
                              kernel_initializer=tf.variance_scaling_initializer())
    
    # and back to channels_last
    inputs = tf.transpose(inputs, [0, 2, 3, 1])
    
    return inputs

  return model