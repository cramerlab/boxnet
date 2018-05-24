
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

import boxnet_model_v2
from estimator_v2 import EstimatorV2

parser = argparse.ArgumentParser()


_HEIGHT = 256
_WIDTH = 256
_HEIGHT_TRAIN = 256
_WIDTH_TRAIN = 256
_DEPTH = 1
_NUM_CLASSES = 3
_BATCHSIZE = 1

# We use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
_WEIGHT_DECAY = 5e-4
_MOMENTUM = 0.9

def input_fn(batch_size):
  
  images = tf.random_normal([batch_size, _WIDTH_TRAIN, _HEIGHT_TRAIN, 1])
  image_classes = tf.random_normal([batch_size, _WIDTH_TRAIN, _HEIGHT_TRAIN, _NUM_CLASSES])
  image_ignore = tf.random_normal([batch_size, _WIDTH_TRAIN, _HEIGHT_TRAIN, 1])

  return {'images' : images, 
          'image_classes' : image_classes, 
          'image_weights' : image_ignore}, images


def boxnet_model_fn(features, labels, mode, params):

  network = boxnet_model_v2.boxnet_v2_generator(_NUM_CLASSES)

  inputs = features["images"] if mode == tf.estimator.ModeKeys.TRAIN else features["images_predict"]
  
  logits = network(inputs, mode == tf.estimator.ModeKeys.TRAIN)
  logits = tf.reshape(logits, [logits.shape[0] * logits.shape[1] * logits.shape[2], _NUM_CLASSES])
  print(logits.shape)

  predictions = {
      'classes': tf.argmax(logits, axis=1, name='argmax_tensor'),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }
  
  export_outputs = {
      'prediction': tf.estimator.export.PredictOutput(predictions)
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, export_outputs=export_outputs, predictions=predictions)
  
  input_labels = tf.reshape(features["image_classes"], [logits.shape[0], _NUM_CLASSES]);
  input_weights = tf.reshape(features["image_weights"], [logits.shape[0]]);
  
  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=input_labels, weights=input_weights, reduction=tf.losses.Reduction.MEAN)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # Add weight decay to the loss.
  loss = cross_entropy + _WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
  global_step = tf.train.get_or_create_global_step()
  
  learning_rate = tf.placeholder(tf.float32, [], name="training_learning_rate") # replace with something small for pseudo-training

  optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=_MOMENTUM)
  #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

  # Batch norm requires update ops to be added as a dependency to the train_op
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss, global_step, name='train_momentum')

  accuracy = tf.metrics.accuracy(tf.argmax(input_labels, axis=1), predictions['classes'])
  metrics = {'accuracy': accuracy}

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  trainings = {
      'classes': predictions['classes'],
      'probabilities': predictions['probabilities'],
      'loss': loss
  }
  
  export_outputs = {
      'training': tf.estimator.export.PredictOutput(trainings)
  }

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics,
      export_outputs=export_outputs)


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # Set up a RunConfig to only save checkpoints once per training cycle.
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=999999).replace(session_config=config)
  
  boxnet = EstimatorV2(model_fn=boxnet_model_fn, model_dir='boxnet_model', config=run_config,
                       params=
                       {
                           'batch_size': _BATCHSIZE,
                       })

  #boxnet.train(input_fn=lambda: input_fn(_BATCHSIZE))
	  
  feature_spec = {'images': tf.placeholder(tf.float32, [None, _WIDTH_TRAIN, _HEIGHT_TRAIN, 1], name="images"),
                  'image_classes': tf.placeholder(tf.float32, [None, _WIDTH_TRAIN, _HEIGHT_TRAIN, _NUM_CLASSES], name="image_classes"),
                  'image_weights': tf.placeholder(tf.float32, [None, _WIDTH_TRAIN, _HEIGHT_TRAIN, 1], name="image_weights"),
                  'images_predict': tf.placeholder(tf.float32, [None, _WIDTH, _HEIGHT, 1], name="images_predict")}
	  
  boxnet.export_savedmodel(export_dir_base='boxnet_model_export', 
                           serving_input_receiver_fn=tf.estimator.export.build_raw_serving_input_receiver_fn(features=feature_spec, 
                                                                                                             default_batch_size=_BATCHSIZE),
                           export_name='BoxNet_256',
                           as_text=False)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
