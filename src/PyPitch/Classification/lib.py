from datetime import datetime
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import pandas_profiling
from scipy.stats import zscore
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler

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

        d_sort_values = sorted(d.items(), key=lambda x: x[1], reverse=False)
        d_sort_values_rev = sorted(d.items(), key=lambda x: x[1], reverse=True)

        top10_neg = list(d_sort_values)[:10]
        top10_pos = list(d_sort_values_rev)[:10]

        return top10_neg, top10_pos


    def describe(data, out_file):
        df_describe = data.describe()
        df_describe.to_csv(out_file)


    def feature_density_plot(data):
        data.plot(kind='density', subplots=True, layout=(5,5), sharex=False, figsize=(15,10))
        plt.show()


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


    def detect_outliers(self, data, features, std_thresh = 6):

        d_outliers = {}

        for column in features:

            # Create z_score proxy for each column
            data['z_score'] = np.absolute(zscore(data[column]))

            # Check if there are NaNs in z_score
            if data['z_score'].isnull().sum() > 0:
                print('||WARN', datetime.now(), '|| NaNs found in data column: {}. More analysis may be necessary to ensure there are not outliers'.format(column))

            # Determine if there are outliers, as defined by z_score threshold
            outliers = data.loc[data.z_score > std_thresh, [column, 'z_score']]

            # If there are no outliers
            if outliers.shape[0] == 0:
                print('||MSG', datetime.now(), '|| No outliers for column {} at threshold of {} stdevs'.format(column, std_thresh))

                d_outliers[column] = []

            # If there are outliers
            else:
                print('||MSG', datetime.now(), '|| {} outlier(s) found for column {} at threshold of {} stdevs'.format(outliers.shape[0], column, std_thresh))

                l_column_outliers = []
                for i,r in outliers.iterrows():
                    l_column_outliers.append({'index': i, 'value': r[column], 'z_score': r['z_score']})

                d_outliers[column] = l_column_outliers

            # Drop z_score from data
            data.drop('z_score', axis = 1, inplace = True)

        return d_outliers


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


    def kmeans_fit_predict(self, data):
        '''
        '''

        _kmeans = self.kmeans

        _kmeans.fit(data)
        labels = _kmeans.predict(data)
        centers = _kmeans.cluster_centers_

        return labels, centers


    def kmeans_predict(self, data):
        '''
        '''

        _kmeans = self.kmeans

        labels = _kmeans.predict(data)

        return labels


    def num_pca_components(self, cum_sum, alpha):
        threshold = 1 - alpha
        n = 1

        for i in cum_sum:
            if i >= threshold:
                return n
            else:
                n += 1


    def pca_fit_transform(self, data, features):
        '''
        This method applies a PCA transformation fit to training data
        '''

        _pca = self.pca
        data_pca_fit_transform = _pca.fit_transform(data[features])

        return data_pca_fit_transform


    def pca_transform(self, data, features):
        '''
        This method applies pca transformation that was previously fit to training data
        '''

        _pca = self.pca
        data_pca_fit = _pca.transform(data[features])

        return data_pca_fit


    def pca_var_coverage(self, explained_var_cumsum):
        plt.figure(figsize = (20,10))
        x = np.arange(1, len(explained_var_cumsum) + 1, 1)
        plt.plot(x, explained_var_cumsum)

        plt.title("Variance Coverage by Principal Components")
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Variance Coverage Percentage")
        plt.show()


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


class Pipeline():

    def __init__(self):
        '''
        Define Pipeline() instance transformers/scalers
        '''

        self.pt = PowerTransformer(method='yeo-johnson', standardize=False,)
        self.ss = StandardScaler()
        self.mm = MinMaxScaler()


    def fit_transform_data(self, data, features):
        '''
        This method applies an optimal power transformation via Yeo-Johnson method, then standardizes and normalizes
        '''

        _pt = self.pt
        _ss = self.ss
        _mm = self.mm

        # YEO-JOHNSON POWER TRANSFORMATION #
        yeojt = _pt.fit(data[features])
        yeojt = _pt.transform(data[features])
        df_yeojt = pd.DataFrame(yeojt, columns=features)

        # STANDARDIZE & NORMALIZE #
        df_std = pd.DataFrame(_ss.fit_transform(df_yeojt[features]), columns=features)
        df_std_norm = pd.DataFrame(_mm.fit_transform(df_std[features]), columns=features)

        # FORMAT RETURN DATA #
        df = data.drop(labels=features, axis=1)
        df = df.join(df_std_norm)

        return df


    def load_model_data(repo_dir, model_version):
        '''
        TO DO:
        - Check if pkl files exist, don't load if they already exist
        - Parameterize player data import, make optional to import one player
        '''

        repo = Path(repo_dir)
        data_output = repo / 'src/PyPitch/output' / model_version / 'data'

        # LOAD RAW DATA #
        data = Load.import_all_data('Lester')

        df_data_dictionary = data['df_data_dictionary']
        df_data = data['df_data']

        df_data_R = df_data.loc[df_data['PitcherThrows'] == 'R']
        df_data_R = df_data_R.reset_index()
        df_data_R = df_data_R.drop(labels=['index'], axis=1)

        df_data_L = df_data.loc[df_data['PitcherThrows'] == 'L']
        df_data_L = df_data_L.reset_index()
        df_data_L = df_data_L.drop(labels=['index'], axis=1)

        # DEFINE FEATURES #
        l_cols = []
        for i in df_data.columns:
            l_cols.append(i)

        l_keep_cols = ['ID', 'PitcherThrows']
        for i,r in df_data_dictionary['ColumnName'].iteritems():
            l_keep_cols.append(r)

        l_drop_cols = list(set(l_cols) - set(l_keep_cols))

        l_drop_cols.append('PitcherThrows')
        l_drop_cols.append('sz_bot')
        l_drop_cols.append('sz_top')
        l_drop_cols.append('Break_y')
        l_drop_cols.append('BreakAngle')
        l_drop_cols.append('BreakLength')
        l_drop_cols.append('x0')
        l_drop_cols.append('y0')
        l_drop_cols.append('z0')
        l_drop_cols.append('vx0')
        l_drop_cols.append('vy0')
        l_drop_cols.append('vz0')
        l_drop_cols.append('ax0')
        l_drop_cols.append('ay0')
        l_drop_cols.append('az0')
        l_drop_cols.append('px')
        l_drop_cols.append('pz')
        l_drop_cols.append('StartSpeed')
        l_drop_cols.append('EndSpeed')
        # l_drop_cols.append('pfx_x')
        # l_drop_cols.append('pfx_z')

        df = df_data.drop(labels=l_drop_cols, axis=1)
        df_R = df_data_R.drop(labels=l_drop_cols, axis=1)
        df_L = df_data_L.drop(labels=l_drop_cols, axis=1)

        features = []
        cols = df.columns

        for i in cols:
            if i != 'ID':
                features.append(i)

        df_features = pd.DataFrame(features, columns=['feature'])

        df_out_file = data_output / 'df.pkl'
        df_R_out_file = data_output / 'df_R.pkl'
        df_L_out_file = data_output / 'df_L.pkl'
        df_features_out_file = data_output / 'features.pkl'

        df.to_pickle(df_out_file)
        df_R.to_pickle(df_R_out_file)
        df_L.to_pickle(df_L_out_file)
        df_features.to_pickle(df_features_out_file)


    def transform_data(self, data, features):
        '''
        This method applies data transformations that were previously fit to training data
        '''

        _pt = self.pt
        _ss = self.ss
        _mm = self.mm

        # YEO-JOHNSON POWER TRANSFORMATION #
        yeojt = _pt.transform(data[features])
        df_yeojt = pd.DataFrame(yeojt, columns=features)

        # STANDARDIZE & NORMALIZE #
        df_std = pd.DataFrame(_ss.transform(df_yeojt[features]), columns=features)
        df_std_norm = pd.DataFrame(_mm.transform(df_std[features]), columns=features)

        # FORMAT RETURN DATA #
        df = data.drop(labels=features, axis=1)
        df = df.join(df_std_norm)

        return df


class Result():

    def __init__(self, data, label_col_name):
        self.data = data
        self.labels = list(data[label_col_name].drop_duplicates())
        self.colors = ['red', 'green', 'blue', 'purple']

    def pfx_scatter_alllabels(self, data, label_col_name):
        _data = self.data
        _labels = self.labels
        _colors = self.colors

        fig, a = plt.subplots()

        for color, label in zip(_colors, _labels):
            pfx_x = np.array(_data['pfx_x'].loc[_data[label_col_name]==label])
            pfx_z = np.array(_data['pfx_z'].loc[_data[label_col_name]==label])
            a.scatter(pfx_x, pfx_z, c=color, label=label)

        a.legend()
        a.grid(True)
        a.set(xlabel='Horizontal Movement', ylabel='Vertical Movement')

        plt.show()


    def pfx_scatter_bylabel(self, data, label_col_name):
        _data = self.data
        _labels = self.labels
        _colors = self.colors

        fig, a = plt.subplots(4)

        pfx_x_0 = np.array(_data['pfx_x'].loc[_data[label_col_name]==_labels[0]])
        pfx_z_0 = np.array(_data['pfx_z'].loc[_data[label_col_name]==_labels[0]])

        pfx_x_1 = np.array(_data['pfx_x'].loc[_data[label_col_name]==_labels[1]])
        pfx_z_1 = np.array(_data['pfx_z'].loc[_data[label_col_name]==_labels[1]])

        pfx_x_2 = np.array(_data['pfx_x'].loc[_data[label_col_name]==_labels[2]])
        pfx_z_2 = np.array(_data['pfx_z'].loc[_data[label_col_name]==_labels[2]])

        pfx_x_3 = np.array(_data['pfx_x'].loc[_data[label_col_name]==_labels[3]])
        pfx_z_3 = np.array(_data['pfx_z'].loc[_data[label_col_name]==_labels[3]])

        a[0].scatter(pfx_x_0, pfx_z_0, c=_colors[0])
        a[0].set_title('Label: {}'.format(_labels[0]))

        a[1].scatter(pfx_x_1, pfx_z_1, c=_colors[1])
        a[1].set_title('Label: {}'.format(_labels[1]))

        a[2].scatter(pfx_x_2, pfx_z_2, c=_colors[2])
        a[2].set_title('Label: {}'.format(_labels[2]))

        a[3].scatter(pfx_x_3, pfx_z_3, c=_colors[3])
        a[3].set_title('Label: {}'.format(_labels[3]))

        for i in a.flat:
            i.set(xlabel='Horizontal Movement (in)', ylabel='Vertical Movement (in)')

        fig.tight_layout()

        plt.show()


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


def save_pickle(out_file, pyobj):
    pkl_file = open(out_file, 'wb')
    pickle.dump(pyobj, pkl_file)


def show_pitch_location(x, z, color):
    plt.scatter(x, z, c=color)
    plt.show()


def show_release_point(x, y, z, color):
    fig_release_point = plt.figure()
    ax = Axes3D(fig_release_point)
    ax.scatter(x, y, z, color)
    plt.show()


