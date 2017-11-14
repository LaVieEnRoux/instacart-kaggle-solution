from __future__ import print_function

import sys
import argparse
from datetime import datetime
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
import sklearn.metrics

from model import mlp_model, log_reg_model
from Instacart.src.dataUtils.loading import DataLoader


def main(_):

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info("Setting up model...")

    # set up model graph
    # model = log_reg_model()
    model = mlp_model(FLAGS.hidden_size)

    all_loss = tf.reduce_mean(model["loss"] 
                              + FLAGS.regularization_rate * model["reg"])
    # all_loss = model["loss"]

    optimizer = \
        tf.train.RMSPropOptimizer(FLAGS.learning_rate).minimize(all_loss)

    # add evaluation operations
    prediction = tf.argmax(model["logits"], 1)
    correct_prediction = \
        tf.equal(prediction, tf.argmax(model["labels"], 1))
    evaluation_step = tf.reduce_mean(
        tf.cast(correct_prediction, tf.float32)
    )

    tf.logging.info("Setting up data pipeline...")

    # set up data pipeline
    loader = DataLoader()

    tf.logging.info("Initializing variables...")
 
    # initialize and begin training
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)
        sess.run(init_local)

        for epoch in np.arange(FLAGS.training_epochs):

            # run training iterations
            for train_iter in np.arange(FLAGS.train_iter):

                batch_x, batch_y = \
                    loader.load_batch(batch_size=FLAGS.batch_size)

               
                start = timer()
                _, loss_value, logit_vals, W = sess.run(
                    [optimizer, all_loss, model["logits"], model["W"]],
                    feed_dict={model["input"]: batch_x,
                               model["labels"]: batch_y}
                )
                end = timer()
                print("Time: {}".format((end - start) / float(batch_y.shape[0])))

                if train_iter % FLAGS.train_update == 0:
                    tf.logging.info(
                        "{}: Iter {} -- Epoch {} -- Loss: {}".format(
                            datetime.now(), train_iter, epoch, loss_value
                        )
                    )

                    '''
                    print()
                    print("Logits: {}".format(logit_vals))
                    print()
                    print("Batch x: {}".format(batch_x))
                    print()
                    print("Weights: {}".format(W))
                    print()
                    '''

                if True in np.isnan(batch_x):
                    print("NaN detected in input!")
                    print("NaN input: {}".format(batch_x[np.where(np.isnan(batch_x))[0]]))
                    sys.exit(1)

                if True in np.isnan(W):
                    print("NaN detected in weights!")
                    sys.exit(1)



            # evaluate on a training batch
            batch_x, batch_y = loader.load_batch(batch_size=FLAGS.eval_batch_size, all_samples=True)
            accuracy, preds, W = \
                sess.run([evaluation_step, prediction, model["W"]],
                         feed_dict={model["input"]: batch_x,
                                    model["labels"]: batch_y}
                        )

            true_labels = np.argmax(batch_y, axis=1)
            pred_labels = preds

            precision = sklearn.metrics.precision_score(true_labels, pred_labels)
            recall = sklearn.metrics.recall_score(true_labels, pred_labels)

            try:
                F1_score = 2 * (precision * recall) \
                    / (precision + recall)
            except:
                F1_score = 0

            tf.logging.info("{}: Epoch {} -- Accuracy: {}".format(
                datetime.now(), epoch, accuracy
            ))
            tf.logging.info("Precision: {} -- Recall: {} -- F1: {}".format(
                precision, recall, F1_score
            ))
            print()
            tf.logging.info("Weights: {}".format(W))

        preds = open("submission.csv", 'w')
        preds.write("order_id,products\n")

        # run inference on final test set
        for order, order_feat in loader.load_test():

            preds.write(str(order.order_id))
            preds.write(",")

            dummy_labels = [[1, 0]] * order_feat.shape[0]

            pred = sess.run([prediction],
                            feed_dict={model["input"]: order_feat,
                                       model["labels"]: dummy_labels})

            pred_products = np.where(pred[0] > 0)[0] + 1
            pred_str = " ".join(list(map(str, pred_products)))

            if len(pred_products) > 0:
                preds.write("{}\n".format(pred_str))
            else:
                preds.write("None\n")

        preds.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=20
    )
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=200
    )
    parser.add_argument(
        '--train_iter',
        type=int,
        default=300
    )
    parser.add_argument(
        '--valid_iter',
        type=int,
        default=100
    )
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=10
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.1
    )
    parser.add_argument(
        '--regularization_rate',
        type=float,
        default=0.01
    )
    parser.add_argument(
        '--training_epochs',
        type=int,
        default=7
    )
    parser.add_argument(
        '--train_update',
        type=int,
        default=50
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
