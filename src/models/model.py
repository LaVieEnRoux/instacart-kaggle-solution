from __future__ import print_function


import tensorflow as tf
import numpy as np

from Instacart.src.dataUtils.loading import FEATURE_SIZE


def mlp_model(hidden_size, positive_ratio):

    model = {}

    model["input"] = tf.placeholder(tf.float32, [None, FEATURE_SIZE])
    model["labels"] = tf.placeholder(tf.float32, [None, 2])  # binary classification

    model["W"] = tf.Variable(tf.random_normal(
        [FEATURE_SIZE, hidden_size], stddev=0.5), name="weights"
    )
    model["b"] = tf.Variable(tf.zeros([hidden_size]), name="biases")

    model["hidden"] = tf.nn.relu(tf.matmul(model["input"], 
                                           model["W"]) + model["b"])

    model["W_2"] = tf.Variable(tf.random_normal(
        [hidden_size, 2], stddev=0.5), name="output_weights"
    )
    model["b_2"] = tf.Variable(tf.zeros([2]), name="output_biases")

    model["logits"] = tf.matmul(model["hidden"], model["W_2"]) \
                      + model["b_2"]

    model["loss"] = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
        model["labels"], model["logits"], 1.0 / positive_ratio
    ))

    model["reg"] = tf.nn.l2_loss(model["W"]) + tf.nn.l2_loss(model["W_2"])

    return model


def log_reg_model():

    model = {}

    model["input"] = tf.placeholder(tf.float32, [None, FEATURE_SIZE])
    model["labels"] = tf.placeholder(tf.float32, [None, 2])  # binary classification

    model["W"] = tf.Variable(tf.random_normal(
        [FEATURE_SIZE, 2], stddev=0.5), name="weights"
    )
    model["b"] = tf.Variable(tf.zeros([2]), name="biases")

    model["logits"] = tf.matmul(model["input"], model["W"]) \
                      + model["b"]

    model["loss"] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=model["labels"], logits=model["logits"]
    ))

    model["reg"] = tf.nn.l2_loss(model["W"])

    return model
