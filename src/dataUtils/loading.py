from __future__ import print_function

import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import progressbar
import cPickle as pickle

from setup import setup_data

FEATURE_SIZE = 21

NEG_MULTIPLIER = 50
MIN_NEGS = 5


class DataLoader(object):

    def __init__(self, dataDir="../../data", shuffle=True,
                 load_histories=False):

        # grab data and basic prior features
        print("Setting up data...")

        self.data, self.dept_feat, self.prod_feat, self.aisle_feat = \
            setup_data(dataDir)

        print("Setting up user histories...")
        history_path = os.path.join(dataDir, "history.csv")
        last_order_path = os.path.join(dataDir, "last_order.csv")
        if not load_histories:
            self.user_histories, self.last_order = self.get_user_histories()
            self.user_histories.to_csv(history_path)
            self.last_order.to_csv(last_order_path)
        else:
            self.user_histories = pd.Series.from_csv(history_path)
            self.last_order = pd.Series.from_csv(last_order_path)

        print("User histories are set up...")

        # variables for sample loading
        self.train_orders = np.unique(self.data["train"].order_id.values)
        self.order_indices = np.arange(len(self.train_orders))
        self.order_idx = 0
        self.product_len = self.data["products"].shape[0]

        if shuffle:
            np.random.shuffle(self.order_indices)


    def get_feature(self, user_id, prod_ids, order):
        ''' Given the user and product, return a set of features thought
        to identify whether or not the user is likely to purchase the
        product again. '''

        output_feature = np.zeros((len(prod_ids), 8))  # product stats

        # grab statistics
        output_feature[:, 0] = self.prod_feat["mean"][prod_ids].values
        output_feature[:, 1] = self.prod_feat["std"][prod_ids].values

        dept_id = \
            self.data["products"].iloc()[prod_ids - 1].department_id.values
        aisle_id = \
            self.data["products"].iloc()[prod_ids - 1].aisle_id.values

        output_feature[:, 2] = self.dept_feat["mean"][dept_id].values
        output_feature[:, 3] = self.dept_feat["std"][dept_id].values

        output_feature[:, 4] = self.aisle_feat["mean"][aisle_id].values
        output_feature[:, 5] = self.aisle_feat["std"][aisle_id].values

        output_feature[:, 6] = self.prod_feat["cart_mean"][prod_ids].values
        output_feature[:, 7] = self.prod_feat["cart_std"][prod_ids].values

        # grab features
        product_feature = self.make_feature(user_id, prod_ids, order)

        output_feature = np.concatenate((output_feature, product_feature),
                                        axis=1)

        return output_feature


    def make_feature(self, user_id, prod_ids, order):
        ''' Given a set of information about the user's order and product,
        return a descriptive feature vector '''

        more_features = np.zeros((len(prod_ids), FEATURE_SIZE - 8))

        # how many times has the product been ordered?
        try:
            times_ordered = self.user_histories[user_id][prod_ids].values
        except KeyError:
            times_ordered = 0
        times_ordered[np.where(np.isnan(times_ordered))[0]] = 0

        # set (sorta) bucketed categorical variable for times_ordered
        never_ordered = np.array((times_ordered == 0), dtype=np.float32)
        ordered_once = np.array((times_ordered == 1), dtype=np.float32)
        ordered_up_to_ten = \
            np.array(((times_ordered > 1) & (times_ordered < 11)),
                     dtype=np.float32)
        ordered_more = np.array((times_ordered >= 11), dtype=np.float32)

        more_features[:, 0] = never_ordered
        more_features[:, 1] = ordered_once
        more_features[:, 2] = ordered_up_to_ten
        more_features[:, 3] = ordered_more

        user_last_order = self.last_order[user_id]
        last_order_num = np.max(user_last_order)

        # what percentage of the user's orders include the product?
        # is it more than prob_thresh?
        order_prob = times_ordered / float(last_order_num)
        prob_thresh = 0.10
        above_thresh = np.array((order_prob > prob_thresh), dtype=np.float32)
        below_thresh = np.array((order_prob <= prob_thresh), dtype=np.float32)
        more_features[:, 4] = above_thresh
        more_features[:, 5] = below_thresh

        # grab order features
        try:
            days_since_last = order.days_since_prior_order.values[0]
            order_dow = order.order_dow.values[0]
            order_hour_of_day = order.order_hour_of_day.values[0]
        except AttributeError:
            days_since_last = order.days_since_prior_order
            order_dow = order.order_dow
            order_hour_of_day = order.order_hour_of_day

        # how many days since last order?
        if days_since_last > 7:
            more_features[:, 6] = 1
        else:
            more_features[:, 7] = 1

        # how recently was the product ordered?
        more_features[:, 8] = self.last_order[user_id][prod_ids].values
        more_features[:, 8] /= last_order_num

        # is it the weekend?
        if order_dow in [0, 1]:
            more_features[:, 9] = 1
        else:
            more_features[:, 10] = 1

        # is it between 4AM and 12PM?
        if (order_hour_of_day <= 12) and (order_hour_of_day >= 4):
            more_features[:, 11] = 1
        else:
            more_features[:, 12] = 1

        # return [times_ordered, prod_order_prob, order_prob, in_last_order,
        #         days_since_last]
        return more_features


    def get_user_histories(self):
        ''' Given the dataframes, create the purchase histories for each
        user. '''

        # since the user -> product mapping is sparse, we can enumerate
        # all of them without worrying about space

        prior_orders = self.data["orders"].ix[
            self.data["orders"].eval_set == "prior"
        ]

        # get a mapping from user id to the products AND 
        # their highest order number and number of orders
        merged = pd.merge(prior_orders[["user_id", "order_id",
                                        "order_number"]],
                          self.data["prior"][["order_id", "product_id"]],
                          how="left",
                          on="order_id")

        recent_orders = \
            merged.groupby(
                ["user_id", "product_id"]
            )["order_number"].aggregate(np.max)

        order_nums = \
            merged.groupby(
                ["user_id", "product_id"]
            )["order_number"].aggregate(lambda x: len(x))

        return order_nums, recent_orders


    def load_test(self):
        ''' return a generator over the test users '''

        test_orders = self.data["orders"].ix[
            self.data["orders"].eval_set == "test"
        ]

        num_test_orders = test_orders.shape[0]
        
        for order_idx in np.arange(num_test_orders):

            order = test_orders.iloc()[order_idx]
            user_id = order.user_id

            # get features for ALL products
            order_feat = self.get_feature(user_id,
                                          np.arange(self.product_len) + 1,
                                          order)

            order_feat[np.where(np.isnan(order_feat))[0]] = 0

            yield (order, order_feat)


    def load_sample(self, all_samples=False):
        ''' load in a sample of data.
        
        Returns two values, a feature matrix and a label.
        Feature:  [numSamples X sizeOfFeature]
        Label:    [numSamples X 1] 
        '''

        # grab index from possibly shuffled index set
        real_idx = self.order_indices[self.order_idx]
        order_id = self.train_orders[real_idx]

        # wrap back around for an epoch and reshuffle
        self.order_idx += 1
        if self.order_idx == len(self.order_indices):
            self.order_idx = 0
            np.random.shuffle(self.order_indices)

        # load actual sample
        order = self.data["orders"].ix[
            self.data["orders"].order_id == order_id
        ]
        user_id = order.user_id.values[0]

        # get label i.e. which products were ordered
        query_str = 'order_id == {} & reordered == 1'.format(order_id)
        products = self.data["train"].query(query_str).product_id.values - 1
        one_hot_label = np.array([[1, 0]] * self.product_len)
        one_hot_label[products] = [0, 1]

        # which products are we loading for samples????
        if all_samples:
            products_to_sample = np.arange(self.product_len)
        else:
            # take all the positive examples and only some of the
            # negative examples
            #
            # We do this because negative examples dramatically outnumber
            # the positive examples
            #
            # Also: hard negative mining. Add all the products
            # that have been ordered but not reordered
            negatives = np.where(one_hot_label == 0)[0]
            num_negatives = max(int(len(products) * NEG_MULTIPLIER),
                                MIN_NEGS)
            added_negatives = np.random.choice(negatives, num_negatives,
                                               replace=False)

            # ordered, but not reordered!
            all_ordered = self.user_histories[user_id].keys() - 1
            not_reordered = np.setdiff1d(all_ordered, products)

            products_to_sample = np.concatenate((products,
                                                 added_negatives))
            products_to_sample = np.concatenate((products_to_sample,
                                                 not_reordered))

            # also, need a different label vector now
            one_hot_label = np.array([[0, 1]] * len(products) +
                                     [[1, 0]] * len(added_negatives) +
                                     [[1, 0]] * len(not_reordered))

        # load feature vector for all product ids
        feature_matrix = self.get_feature(user_id, products_to_sample + 1, order)

        return feature_matrix, one_hot_label


    def load_batch(self, batch_size=1):
        ''' load in a single batch of order data '''

        features, labels = np.empty((0, FEATURE_SIZE)), np.empty((0, 2))

        for ii in np.arange(batch_size):

            feature, label = self.load_sample()
            features = np.concatenate((features, feature))
            labels = np.concatenate((labels, label))

        # gotta normalize, too!
        # Be careful -- sometimes a feature column can be all zeros
        #print("BEFORE: {}".format(features[0, :]))
        #features -= np.mean(features, axis=0)
        #feature_std = np.std(features)
        #good_std_idx = np.where(feature_std > 1E-10)[0]
        #features[:, good_std_idx] \
        #    /= np.std(features[:, good_std_idx], axis=0)
        #print("AFTER: {}".format(features[0, :]))

        # change NaNs to 0
        features[np.where(np.isnan(features))[0]] = 0

        return features, labels



if __name__ == "__main__":

    dl = DataLoader(load_histories=True)
    dl.load_batch()
