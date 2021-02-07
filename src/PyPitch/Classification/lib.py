from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import pandas_profiling
from scipy.stats import zscore
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA

from PyPitch.db import query_data_dictionary, query_raw_data, query_pitcher_data


class EDA():

    def correlation_analysis(data, num_cols, univariate = False):
        corr = data[num_cols].corr()

        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=(11, 9))

        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False)

        plt.show()


    def correlation_rank(data, num_cols):
        corr = data[num_cols].corr()

        d = {}
        for var in corr:
            var_name = corr[var].name
            ix = corr[var].index
            corr_value = corr[var]

            for i, element in enumerate(ix):
                l = [var_name, element]
                l.sort()

                if var_name != element:
                    var_pair = l[0] + '__' + l[1]
                    d[var_pair] = corr_value[i]

        # Sort var pairs by correlation values
        d_sort_values = sorted(d.items(), key=lambda x: x[1], reverse=False)
        d_sort_values_rev = sorted(d.items(), key=lambda x: x[1], reverse=True)

        # Define highest positive & negative correlations
        top10_neg = list(d_sort_values)[:10]
        top10_pos = list(d_sort_values_rev)[:10]

        return top10_neg, top10_pos


    def describe(data, out_file):
        df_describe = data.describe()
        df_describe.to_csv(out_file)

        return 0


    def feature_density_plot(data):
        data.plot(kind='density', subplots=True, layout=(5,5), sharex=False, figsize=(15,10))
        plt.show()

        return 0


    def null_check(data):
        d_null_cols = {}
        null_check = data.isnull().sum()

        for i in range(len( null_check)):
            null_ct = None
            null_key = None

            if null_check[i] != 0:
                null_ct = null_check[i]
                null_key = null_check.keys()[i]

                d_null_cols[null_key] = null_ct

                print('||WARN', datetime.now(), '||', null_ct, 'NULL VALUES EXIST FOR', null_key)

        return d_null_cols


    def profile(data, out_file):
        profile = pandas_profiling.ProfileReport(data)
        profile.to_file(out_file)

        return 0





class Load():

    def import_all_data(search_text):
        try:
            retd = {}

            # IMPORT DATA DICTIONARY #
            ret1, df1 = import_data_dictionary()

            retd['ret1'] = ret1

            if ret1 == 0:
                print('||MSG', datetime.now(), '|| SHAPE:', df1.shape)

            # IMPORT ALL RAW DATA #
            ret2, df2 = import_raw_data()

            retd['ret2'] = ret2

            if ret2 == 0:
                print('||MSG', datetime.now(), '|| SHAPE:', df2.shape)

            # IMPORT PITCHER DATA #
            ret3, df3 = import_pitcher_data(search_text)

            retd['ret3'] = ret3

            if ret3 == 0:
                print('||MSG', datetime.now(), '|| SHAPE:', df3.shape)

            s = retd.values()

            if 1 in s:
                raise 'ERROR IMPORTING MODEL DATA'

            response = {'df_data_dictionary': df1, 'df_data': df2, 'df_data_pitcher': df3}

            return response

        except Exception as e:
            print('||ERR', datetime.now(), '|| ERROR MESSAGE:', e)

            return e





class KMeansModel():

    def __init__(self, n_clust):
        self.pca = PCA()
        self.kmeans = KMeans(n_clust, random_state=True)


    def run_kmeans(self, data):
        _kmeans = self.kmeans

        _kmeans.fit(data)
        labels = _kmeans.predict(data)
        centers = _kmeans.cluster_centers_

        return labels, centers


    def kmeans_predict(self, data, features):
        _kmeans = self.kmeans

        labels = _kmeans.predict(data[features])

        return labels


    def pca_fit(self, data, features):
        '''
        TO DO: Create options to fit, transform, fit_transform, untransform, etc...
        '''

        _pca = self.pca
        data_pca_fit = _pca.fit_transform(data[features])

        return data_pca_fit


    def elbow_plot(self, data, cluster_range):
        cluster_results = []
        cluster_range_list = []
        
        for c in range(cluster_range[0],cluster_range[1] + 1):
            kmeans = KMeans(n_clusters=c, random_state=True)
            kmeans.fit(data)
            distance = kmeans.inertia_
            cluster_range_list.append(c)
            cluster_results.append(distance)

        plt.figure(figsize = (20,10))
        plt.plot(cluster_range_list, cluster_results)
        plt.xlabel("Number of Clusters")
        plt.ylabel("Sum of Squared Distances")
        plt.title("Elbow Plot for K-means Clustering")
        plt.show()


    def num_pca_components(self, cum_sum, alpha):
        threshold = 1 - alpha
        n = 1

        for i in cum_sum:
            if i >= threshold:
                return n
            else:
                n += 1


    def pca_var_coverage(self, explained_var_cumsum):
        plt.figure(figsize = (20,10))
        x = np.arange(1, len(explained_var_cumsum) + 1, 1)
        plt.plot(x, explained_var_cumsum)

        plt.title("Variance Coverage by Principal Components")
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Variance Coverage Percentage")
        plt.show()


    def detect_outliers(self, data, features, std_thresh = 6):

        d_outliers = {}

        for column in features:

            # Create z_score proxy for each column
            data['z_score'] = np.absolute(zscore(data[column]))

            # Check if there are NaNs in z_score
            if data['z_score'].isnull().sum() > 0:
                print('WARNING: NaNs found in data column: {}. More analysis may be necessary to ensure there are not outliers'.format(column))

            # Determine if there are outliers, as defined by z_score threshold
            outliers = data.loc[data.z_score > std_thresh, [column, 'z_score']]

            # If there are no outliers
            if outliers.shape[0] == 0:
                print('No outliers for column {} at threshold of {} stdevs'.format(column, std_thresh))

                d_outliers[column] = []

            # If there are outliers
            else:
                print('{} outlier(s) found for column {} at threshold of {} stdevs'.format(outliers.shape[0], column, std_thresh))

                l_column_outliers = []
                for i,r in outliers.iterrows():
                    l_column_outliers.append({'index': i, 'value': r[column], 'z_score': r['z_score']})

                d_outliers[column] = l_column_outliers

            # Drop z_score from data
            data.drop('z_score', axis = 1, inplace = True)

        return d_outliers


    def silhouette_viz(self, train_data, model, num_clusters, x_feature_index = 1, y_feature_index = 2):
        # Create a subplot with 1 row and 2 columns
        #Set the size of the plot
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from [-1, 1]
        # However, typically the scores will be positive
        ax1.set_xlim([-0.2, 1])

        #Isolate cluster labels
        cluster_labels = model.labels_

        # The (cluster_num+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(train_data) + (num_clusters + 1) * 10])

        #Generate silhouette average
        silhouette_avg = silhouette_score(train_data, cluster_labels)

        #Generate silhouette_scores
        sample_silhouette_values = silhouette_samples(train_data, cluster_labels)

        y_lower = 10

        #For each cluster number
        for i in range(num_clusters):

            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            # Set the size of the cluster, and adjust viz size
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            # Set colors and fill in the distances shape for
            # each data point silhouette score
            color = cm.nipy_spectral(float(i) / num_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        #Set the title and axis labels
        ax1.set_title("The silhouette plot for the various clusters")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / num_clusters)

        ax2.scatter(train_data[:, x_feature_index],
                    train_data[:, y_feature_index],
                    marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = model.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, x_feature_index], centers[:, y_feature_index], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        # Draw white circles at cluster centers
        ax2.scatter(centers[:, x_feature_index], centers[:, y_feature_index], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        #Plot the clusters with the centroids
        for i, c in enumerate(centers):
            ax2.scatter(c[x_feature_index], c[y_feature_index], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        #Set axis plot and labels
        ax2.set_title("The visualization of the clustered data")
        ax2.set_xlabel('PCA Component 1')
        ax2.set_ylabel('PCA Component 2')

        #Set overall title for both plots
        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                    "with n_clusters = {}".format(num_clusters)),
                    fontsize=14, fontweight='bold')

        #Show the plot
        plt.show()


    def split(self, data, test_ratio):
        train_size = int(len(data) * test_ratio)
        df_train, df_test = data.loc[1:train_size], data.loc[train_size:]

        df_train = df_train.reset_index()
        df_test = df_test.reset_index()

        df_train = df_train.drop(labels=['index'], axis=1)
        df_test = df_test.drop(labels=['index'], axis=1)

        return df_train, df_test


    def remove_outliers(self, data, features, std_thresh):
        outliers = KMeansModel.detect_outliers(self, data, features, std_thresh)

        delete = []
        w_outliers = len(data)
        for i in outliers:
            if outliers[i]:
                for i in outliers[i]:
                    delete.append(i['index'])

        delete = list(set(delete))
        data = data.drop(data.index[delete])
        wo_outliers = len(data)
        outliers_removed = w_outliers - wo_outliers

        df = data.reset_index()
        df = df.drop(labels=['index'], axis=1)

        print('||MSG', datetime.now(), '||', outliers_removed, 'OUTLIERS REMOVED')

        return df





def import_data_dictionary():
    ret, df = query_data_dictionary()

    return 0, df





def import_pitcher_data(search_text):
    '''
    Import raw pitcher data from sql instance
    '''

    try:
        print('||MSG', datetime.now(), '|| IMPORTING RAW DATASET')

        ret, df = query_pitcher_data(search_text)

        if ret == 1:
            raise Exception()

        print('||MSG', datetime.now(), '|| IMPORTED RAW DATASET SUCCESSFULLY WITH SHAPE:', df.shape)

        return 0, df

    except Exception as e:
        print('||ERR', datetime.now(), '|| ERROR MESSAGE:', e)

        df = pd.DataFrame(None)

        return 1, df





def import_raw_data():
    '''
    Import raw dataset from sql instance
    '''

    try:
        print('||MSG', datetime.now(), '|| IMPORTING RAW DATASET')

        ret, df = query_raw_data()

        if ret == 1:
            raise Exception()

        print('||MSG', datetime.now(), '|| IMPORTED RAW DATASET SUCCESSFULLY WITH SHAPE:', df.shape)

        return 0, df

    except Exception as e:
        print('||ERR', datetime.now(), '|| ERROR MESSAGE:', e)

        df = pd.DataFrame(None)

        return 1, df

