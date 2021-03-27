# ============================================================== #
#  SECTION: Imports                                              #
# ============================================================== #

# standard library
import sys
from math import sqrt

# third party libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# k means
from sklearn.cluster import KMeans

# dendogram
from scipy.cluster.hierarchy import dendrogram

# local


# ============================================================== #
#  SECTION: Globals                                              #
# ============================================================== #

# name of csv containing shopping cart records
SHOPPING_CART_RECORDS_CSV = 'HW_PCA_SHOPPING_CART_v896.csv'


# ============================================================== #
#  SECTION: Class Definitions                                   #
# ============================================================== #

class Cluster:
    # id of the next cluster
    ID = 0

    """Class representing a cluster."""
    def __init__(self, data_points, center=None):
        # handle non-list object
        if not isinstance(data_points, list):
            data_points = [data_points]
        # list object containing points within the cluster of type Series
        self.data_points = data_points
        # Series representing center of cluster
        self.center = center
        # gives cluster an id and sets id for next cluster
        self.id = Cluster.ID
        Cluster.ID += 1

    def compute_center(self, replace_center=True):
        """Compute and return the center of the cluster. If replace_center
           is true set clusters center to newly computed one."""
        # dictionary containing the average value for each column in records
        center_dict = {}

        # for each column in records
        for column in self.data_points[0].keys():
            # add the value at [row][column] to column_values
            column_values = []
            for record in self.data_points:
                column_values.append(record[column])

            # compute average
            average = sum(column_values) / len(column_values)

            # add mode to dictionary
            center_dict[column] = average

        # create a Series object as the computed center
        center = pd.Series(center_dict)

        # if replace center is true make the computed center the clusters center
        if replace_center:
            self.center = center
        return center

# ============================================================== #
#  SECTION: Helper Definitions                                   #
# ============================================================== #


def find_strongest_correlation(cross_correlation_matrix, prime_attribute):
    """Find the attribute with the most strongest correlation to prime_attribute within the cross_correlation_matrix."""
    # get a mapping of the correlation among the prime_attribute and other attributes in the cross_correlation_matrix
    prime_attribute_to_all_attributes = cross_correlation_matrix[prime_attribute]
    # remove prime_attribute from mapping
    del prime_attribute_to_all_attributes[prime_attribute]

    # default values
    highest_correlating_attribute = None
    highest_correlation = 0
    # loop through all attributes
    for attribute in prime_attribute_to_all_attributes.index:
        # get the correlation of primary_attribute and attribute
        attribute_correlation = abs(prime_attribute_to_all_attributes[attribute])
        # if correlation exceeds known highest correlation a new highest correlation has been found
        if attribute_correlation > highest_correlation:
            highest_correlation = attribute_correlation
            highest_correlating_attribute = attribute

    return highest_correlating_attribute, highest_correlation


def find_weakest_correlation(cross_correlation_matrix, prime_attribute):
    """Find the attribute with the least correlation to prime_attribute within the cross_correlation_matrix."""
    # get a mapping of the correlation among the prime_attribute and other attributes in the cross_correlation_matrix
    prime_attribute_to_all_attributes = cross_correlation_matrix[prime_attribute]
    # remove prime_attribute from mapping
    del prime_attribute_to_all_attributes[prime_attribute]

    # default values
    lowest_correlating_attribute = None
    lowest_correlation = sys.maxsize
    # loop through all attributes
    for attribute in prime_attribute_to_all_attributes.index:
        # get the correlation of primary_attribute and attribute
        attribute_correlation = abs(prime_attribute_to_all_attributes[attribute])
        # if correlation is less than the known highest correlation a new highest correlation has been found
        if attribute_correlation < lowest_correlation:
            lowest_correlation = attribute_correlation
            lowest_correlating_attribute = attribute

    return lowest_correlating_attribute, lowest_correlation


def find_total_correlation(cross_correlation_matrix, prime_attribute):
    """Find the total correlation among prime_attribute and all other attributes."""
    # get a mapping of the correlation among the prime_attribute and other attributes in the cross_correlation_matrix
    prime_attribute_to_all_attributes = cross_correlation_matrix[prime_attribute]
    # remove prime_attribute from mapping
    del prime_attribute_to_all_attributes[prime_attribute]

    total_correlation = 0
    for attribute in prime_attribute_to_all_attributes.index:
        # get the correlation of primary_attribute and attribute, add it to the total
        total_correlation += abs(prime_attribute_to_all_attributes[attribute])
    return total_correlation


def euclidean_distance(cluster1, cluster2, column_names):
    """Return the euclidean_distance of two clusters."""
    summation = 0
    for column_name in column_names:
        summation += ((cluster1.center[column_name] - cluster2.center[column_name]) ** 2)

    return sqrt(summation)


def compute_distances(clusters, column_names):
    """Return a dictionary containing all cluster pairs mapped to their distances."""
    # initial values
    index_start = 0
    index_seeker = 1

    # a list of distances
    distances = dict()

    # while indexes are in range of of clusters
    while index_start < len(clusters) - 1:
        while index_seeker < len(clusters):
            # compute the euclidian distance two clusters
            distance = euclidean_distance(clusters[index_start], clusters[index_seeker], column_names)
            distances[(index_start, index_seeker)] = distance

            # if distance is less than the known lowest distance reassess
            index_seeker += 1
        index_start += 1
        index_seeker = index_start + 1

    return distances


def combine_clusters(cluster1, cluster2):
    """Combine two clusters in clusters."""
    # add cluster1 and cluster2's records together
    records = cluster1.data_points + cluster2.data_points
    # create a new cluster with their merged records
    new_cluster = Cluster(records)
    # compute the new clusters center
    new_cluster.compute_center()

    return new_cluster


def show_kmeans(data, number_of_clusters=6):
    """Run kmeans."""
    print('==================K-Means==============')
    # run kmeans to fit data into number_of_clusters, 10 times (default)
    kmeans = KMeans(n_clusters=number_of_clusters).fit(data)
    # print the prototype for each cluster
    print(kmeans.cluster_centers_)

    labels = data.columns
    centers = kmeans.cluster_centers_
    for cluster_id, center in enumerate(centers):
        # sort the attributes of the clusters center from least to most common item
        sorted_items = sorted(zip(labels, list(center)), key=lambda item: item[1])
        # grab the 5 least common items
        min_list = sorted_items[:5]
        # grab the 5 most common items
        max_list = sorted_items[-5:]
        print("K_Cluster", cluster_id + 1)
        # print list of items by name
        print('The 5 most common items:', list(map(lambda item: item[0], max_list)))
        print('The 5 least common items:', list(map(lambda item: item[0], min_list)))

# ============================================================== #
#  SECTION: Main                                                 #
# ============================================================== #


if __name__ == '__main__':
    # read shopping records into data frame
    shopping_records = pd.read_csv(SHOPPING_CART_RECORDS_CSV)
    # remove unique attribute ID from data frame
    del shopping_records['ID']

    # compute the pearson's cross correlation coefficient
    cross_correlation_matrix = shopping_records.corr()
    # pandas config for debug
    pd.set_option('max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_rows', cross_correlation_matrix.shape[0] + 1)
    print(cross_correlation_matrix)

    # for each attribute find the attribute which correlates best with it, and it's correlation value
    for attribute in cross_correlation_matrix.columns:
        best_correlating_attribute, correlation = find_strongest_correlation(cross_correlation_matrix.copy(), attribute)
        print('{} is most strongly correlated with {} and has a correlation value of {}'. format(attribute,
                                                                                                 best_correlating_attribute,
                                                                                                 correlation))

    # for each attribute find the attribute which correlates least with it, and it's correlation value
    for attribute in cross_correlation_matrix.columns:
        worst_correlating_attribute, correlation = find_weakest_correlation(cross_correlation_matrix.copy(), attribute)
        print('{} is least correlated with {} and has a correlation value of {}'. format(attribute,
                                                                                         worst_correlating_attribute,
                                                                                         correlation))

    # for each attribute get its total correlation with all other attributes.
    for attribute in cross_correlation_matrix.columns:
        total_correlation = find_total_correlation(cross_correlation_matrix.copy(), attribute)
        print('{}\'s total correlation {}'. format(attribute, total_correlation))

    # PART B OF HW

    # create a cluster out of each data point
    clusters = [Cluster(row, row) for _, row in shopping_records.iterrows()]
    # compute cluster distances
    distances = compute_distances(clusters, shopping_records.columns)
    # create dictionary sorted by values (distances), mapping tuples of two indexes within clusters to distance between
    sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1]))

    linkage_matrix = []
    # stop when there is only one cluster
    while len(clusters) != 1:
        # retrieve the indexes of the closest clusters
        cluster1_index, cluster2_index = list(sorted_distances.keys())[0]
        # remove the closest two clusters from the list of clusters
        cluster2 = clusters.pop(cluster2_index)
        cluster1 = clusters.pop(cluster1_index)
        # merge those clusters
        new_cluster = combine_clusters(cluster1, cluster2)
        # add merge to linkage matrix
        linkage_matrix.append([cluster1.id,
                               cluster2.id,
                               sorted_distances[(cluster1_index, cluster2_index)],
                               len(new_cluster.data_points)])

        # create a new sorted dictionary of pairs of cluster indexes to distances
        new_distances = dict()
        for cluster_indexes, distance in sorted_distances.items():
            # do not include entries that utilized either one of the two closest clusters
            if cluster1_index in cluster_indexes or cluster2_index in cluster_indexes:
                continue
            # due to the removal of two clusters references to indexes must be fixed/shifted
            cluster1_shift = 0
            cluster2_shift = 0
            if cluster_indexes[0] > cluster1_index:
                cluster1_shift += 1
            if cluster_indexes[0] > cluster2_index:
                cluster1_shift += 1

            if cluster_indexes[1] > cluster1_index:
                cluster2_shift += 1
            if cluster_indexes[1] > cluster2_index:
                cluster2_shift += 1

            # add the old entry and possibly shifted entry to the new dictionary
            new_distances[(cluster_indexes[0] - cluster1_shift, cluster_indexes[1] - cluster2_shift)] = distance

        # create an entries of every cluster to the newly formed cluster and distance between them
        for cluster_index, cluster in enumerate(clusters):
            new_distances[(cluster_index, len(clusters))] = euclidean_distance(cluster, new_cluster, shopping_records.columns)
        # add merged cluster
        clusters.append(new_cluster)
        # resort the dictionary
        sorted_distances = dict(sorted(new_distances.items(), key=lambda item: item[1]))

        # code used to output hw answers
        debug = True
        if debug:
            print('=========NEWMERGE=========')
            print('{} into {}'.format(min(len(cluster1.data_points), len(cluster2.data_points)),
                                      max(len(cluster1.data_points), len(cluster2.data_points))))
            print('CLUSTERS LEFT: {}'.format(len(clusters)))
            size_list = [len(cluster.data_points) for cluster in clusters]
            size_list.sort()
            print(size_list)

            if len(size_list) == 6:
                for i, cluster in enumerate(clusters):
                    print("CLUSTER", i+1, '---------------------------------------------------')
                    print(pd.DataFrame(cluster.data_points).sum())
                    print('CENTER')
                    print(cluster.center)
                print('++++++++AVERAGE PROTOTYPE+++++++++')
                avg = pd.DataFrame(columns=shopping_records.columns)
                for cluster in clusters:
                    avg = avg.append([cluster.center], ignore_index=True)
                print(avg)
                print(avg.mean())

    # perform k-means
    show_kmeans(shopping_records)

    # create and show dendrogram of the linkage matrix
    dendrogram(np.array(linkage_matrix))
    plt.show()

