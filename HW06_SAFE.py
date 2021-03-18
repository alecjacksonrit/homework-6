# ============================================================== #
#  SECTION: Imports                                              #
# ============================================================== #

# standard library
import sys
import statistics
from math import sqrt

# third party library
import pandas as pd

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
    """Class representing a cluster."""
    def __init__(self, data_points, center=None):
        # handle non-list object
        if not isinstance(data_points, list):
            data_points = [data_points]
        # list object containing points within the cluster of type Series
        self.data_points = data_points
        # Series representing center of cluster
        self.center = center

    def compute_center(self, replace_center=True):
        """Compute and return the center of the cluster. If replace_center
           is true set clusters center to newly computed one."""
        # dictionary containing the median value for each column in records
        center_dict = {}

        # for each column in records
        for column in self.data_points[0].keys():
            # add the value at [row][column] to column_values
            column_values = []
            for record in self.data_points:
                column_values.append(record[column])

            # compute median

            median = statistics.median(column_values)

            # add mode to dictionary
            center_dict[column] = median

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


def find_closest_pairs(clusters, column_names):
    """Return the indexes associated with the two closest pairs in clusters."""
    # initial values
    index_start = 0
    index_seeker = 1
    closest_pairs = (index_start, index_seeker)
    closest_pair_distance = sys.maxsize

    # while indexes are in range of of clusters
    while index_start < len(clusters) - 1:
        while index_seeker < len(clusters):

            def euclidean_distance(cluster1, cluster2, column_names):
                """Return the jaccard coefficient of two clusters."""
                summation = 0
                for column_name in column_names:
                    summation += ((cluster1.center[column_name] - cluster2.center[column_name])**2)

                return sqrt(summation)

            # compute the jaccard coefficient of two clusters
            distance = euclidean_distance(clusters[index_start], clusters[index_seeker], column_names)
            # if distance is less than the known lowest distance reassess
            if distance < closest_pair_distance:
                closest_pairs = (index_start, index_seeker)
                closest_pair_distance = distance
            index_seeker += 1
        index_start += 1
        index_seeker = index_start + 1

    # return indexes associated with two closest pairs in clusters
    return closest_pairs


def combine_clusters(clusters, cluster_index1, cluster_index2, debug=True):
    """Combine two clusters in clusters."""
    # pop the cluster at cluster_index2 from clusters
    cluster2 = clusters.pop(cluster_index2)
    # pop the cluster at cluster_index1 from clusters
    cluster1 = clusters.pop(cluster_index1)

    # add cluster1 and cluster2's records together
    records = cluster1.data_points + cluster2.data_points
    # create a new cluster with their merged records
    new_cluster = Cluster(records)
    # compute the new clusters center
    new_cluster.compute_center()
    # append the new cluster to the existing set of clusters
    clusters.append(new_cluster)

    # code used from homework and debugging
    if debug:
        print('=========NEWMERGE=========')
        print('{} into {}'.format(min(len(cluster1.data_points), len(cluster2.data_points)),
                                  max(len(cluster1.data_points), len(cluster2.data_points))))
        print('CLUSTERS LEFT: {}'.format(len(clusters) + 1))
        size_list = [len(cluster.data_points) for cluster in clusters]
        size_list.sort()
        print(size_list)

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
    # clusters = [Cluster(row, row) for _, row in shopping_records.iterrows()]
#
    # # stop when there is only one cluster
    # while len(clusters) != 1:
    #     cluster_index1, cluster_index2 = find_closest_pairs(clusters, shopping_records.columns)
    #     combine_clusters(clusters, cluster_index1, cluster_index1)