# ============================================================== #
#  SECTION: Imports                                              #
# ============================================================== #

# standard library
import sys
import statistics
from math import sqrt

# third party library
import pandas as pd

# k means
from sklearn.cluster import KMeans

# local


# ============================================================== #
#  SECTION: Globals                                              #
# ============================================================== #

# name of csv containing shopping cart records
SHOPPING_CART_RECORDS_CSV = 'HW_PCA_SHOPPING_CART_v896.csv'

# ============================================================== #
#  SECTION: Main                                                 #
# ============================================================== #


def show_kmeans(data, number_of_clusters=6):
    """Run kmeans."""
    print('==================K-Means==============')
    # run kmeans to fit data into number_of_clusters
    kmeans = KMeans(n_clusters=number_of_clusters).fit(data)
    # print the prototype for each cluster
    print(kmeans.cluster_centers_)

    labels = data.columns
    centers = kmeans.cluster_centers_
    for cluster_id, center in enumerate(centers):
        # sort the attributes of the clusters center from least to most common item
        sorted_items = sorted(zip(labels, list(center)), key=lambda item: item[1])
        # grab the 5 most common items
        min_list = sorted_items[:5]
        # grab the 5 least common items
        max_list = sorted_items[-5:]
        print("K_Cluster", cluster_id + 1)
        # print list of items by name
        print('The 5 most common items:', list(map(lambda item: item[0], max_list)))
        print('The 5 least common items:', list(map(lambda item: item[0], min_list)))


if __name__ == '__main__':
    # read shopping records into data frame
    shopping_records = pd.read_csv(SHOPPING_CART_RECORDS_CSV)
    # remove unique attribute ID from data frame
    del shopping_records['ID']

    show_kmeans(shopping_records)