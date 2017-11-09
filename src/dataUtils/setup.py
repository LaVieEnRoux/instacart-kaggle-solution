from __future__ import print_function

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_all_df(dataDir):

    df = {}

    df["aisles"] = pd.read_csv(
        os.path.join(dataDir, "aisles.csv"))
    df["train"] = pd.read_csv(
        os.path.join(dataDir, "order_products__train.csv"))
    df["prior"] = pd.read_csv(
        os.path.join(dataDir, "order_products__prior.csv"))
    df["orders"] = pd.read_csv(
        os.path.join(dataDir, "orders.csv"))
    df["products"] = pd.read_csv(
        os.path.join(dataDir, "products.csv"))
    df["departments"] = pd.read_csv(
        os.path.join(dataDir, "departments.csv"))

    return df


def setup_data(dataDir):
    ''' Do the preliminary combinations and light feature processing on the data
    before being setup for the data loader '''

    df = get_all_df(dataDir)

    # merge prior
    df["prior"] = pd.merge(df["prior"], df["products"], on="product_id",
                           how='left')
    df["prior"] = pd.merge(df["prior"], df["aisles"], on="aisle_id",
                           how="left")
    df["prior"] = pd.merge(df["prior"], df["departments"], on="department_id",
                           how="left")

    '''
    # merge training
    df["train"] = pd.merge(df["train"], df["products"], on="order_id",
                           how='left')
    df["train"] = pd.merge(df["train"], df["aisles"], on="aisle_id",
                           how="left")
    df["train"] = pd.merge(df["train"], df["departments"], on="department_id",
                           how="left")
    '''

    dept_feat, prod_feat, aisle_feat = {}, {}, {}

    # set up probability prior features (mean and std)
    dept_group =         df["prior"].groupby("department_id")["reordered"]
    dept_feat["mean"] =  dept_group.aggregate(np.mean)
    dept_feat["std"] =   dept_group.aggregate(np.std)
    prod_group =         df["prior"].groupby("product_id")["reordered"]
    prod_feat["mean"] =  prod_group.aggregate(np.mean)
    prod_feat["std"] =   prod_group.aggregate(np.std)
    aisle_group =        df["prior"].groupby("aisle_id")["reordered"]
    aisle_feat["mean"] = aisle_group.aggregate(np.mean)
    aisle_feat["std"] =  aisle_group.aggregate(np.std)
    prod_group_2 = \
        df["prior"].groupby("product_id")["add_to_cart_order"]
    prod_feat["cart_mean"] = prod_group_2.aggregate(np.mean)
    prod_feat["cart_std"] = prod_group_2.aggregate(np.std)

    # normalize these ones
    prod_feat["cart_mean"] -= np.mean(prod_feat["cart_mean"].values)
    prod_feat["cart_mean"] /= np.std(prod_feat["cart_mean"].values)
    prod_feat["cart_std"] -= np.nanmean(prod_feat["cart_std"].values)
    prod_feat["cart_std"] /= np.nanstd(prod_feat["cart_std"].values)

    return df, dept_feat, prod_feat, aisle_feat
