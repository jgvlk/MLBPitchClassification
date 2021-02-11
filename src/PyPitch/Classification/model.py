from datetime import datetime
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import event
from sqlalchemy.dialects import mssql

from PyPitch.classification.lib import KMeansModel, Pipeline, save_pickle
from PyPitch.db import SessionManager


def run_model():

    try:

        # SET PARAMETERS #
        if os.name == 'posix':
            repo = Path('/Users/jonathanvlk/dev/MLBPitchClassification')
        else:
            repo = Path('C:/Users/jonat/source/repos/MLBPitchClassification')

        version = 'v1'
        test_ratio = .7
        outlier_std_thresh = 6
        pca_alpha = .01
        clusters = 4 #Aiming to identify top level pitch classes -> FB, CH, SL (sideways break), CV (vertical break)


        # INSTANTIATE PIPELINES AND CLUSTERING MODELS #
        pipeline = Pipeline()
        pipelineR = Pipeline()
        pipelineL = Pipeline()

        model = KMeansModel(clusters)
        modelR = KMeansModel(clusters)
        modelL = KMeansModel(clusters)


        # LOAD RAW DATA AND SUMMARIZE #
        print('||MSG', datetime.now(), '|| IMPORTING MODEL DATA')

        # Pipeline.load_model_data(repo, version)

        if os.name == 'posix':
            df = pd.read_pickle(r'/Users/jonathanvlk/dev/MLBPitchClassification/src/PyPitch/output/v1/data/df.pkl')
            df_R = pd.read_pickle(r'/Users/jonathanvlk/dev/MLBPitchClassification/src/PyPitch/output/v1/data/df_R.pkl')
            df_L = pd.read_pickle(r'/Users/jonathanvlk/dev/MLBPitchClassification/src/PyPitch/output/v1/data/df_L.pkl')
            features = pd.read_pickle(r'/Users/jonathanvlk/dev/MLBPitchClassification/src/PyPitch/output/v1/data/features.pkl')
        else:
            df = pd.read_pickle(r'C:\Users\jonat\source\repos\MLBPitchClassification\src\PyPitch\output\v1\data\df.pkl')
            df_R = pd.read_pickle(r'C:\Users\jonat\source\repos\MLBPitchClassification\src\PyPitch\output\v1\data\df_R.pkl')
            df_L = pd.read_pickle(r'C:\Users\jonat\source\repos\MLBPitchClassification\src\PyPitch\output\v1\data\df_L.pkl')
            features = pd.read_pickle(r'C:\Users\jonat\source\repos\MLBPitchClassification\src\PyPitch\output\v1\\data\features.pkl')

        features = list(features['feature'])

        # TRAIN/TEST SPLIT AND SUMMARIZE #
        print('||MSG', datetime.now(), '|| CREATING TRAIN & TEST DATASETS')

        df_train, df_test = model.split(df, test_ratio)
        df_R_train, df_R_test = modelR.split(df_R, test_ratio)
        df_L_train, df_L_test = modelL.split(df_L, test_ratio)


        # TRANSFORM #
        print('||MSG', datetime.now(), '|| FITTING & TRANSFORMING TRAINING DATA')

        df_train = pipeline.fit_transform_data(df_train, features)
        df_R_train = pipelineR.fit_transform_data(df_R_train, features)
        df_L_train = pipelineL.fit_transform_data(df_L_train, features)


        # CHECK FOR & HANDLE OUTLIERS #
        print('||MSG', datetime.now(), '|| REMOVING OUTLIERS AT STD THRESHOLD =', outlier_std_thresh)

        df_train = model.remove_outliers(df_train, features, outlier_std_thresh)
        df_R_train = modelR.remove_outliers(df_R_train, features, outlier_std_thresh)
        df_L_train = modelL.remove_outliers(df_L_train, features, outlier_std_thresh)


        # PCA #
        print('||MSG', datetime.now(), '|| PERFORMING PCA ANALYSIS')

        df_model = model.pca_fit_transform(df_train, features)
        df_R_model = modelR.pca_fit_transform(df_R_train, features)
        df_L_model = modelL.pca_fit_transform(df_L_train, features)

        # vars = model.pca.explained_variance_ratio_
        # varsR = modelR.pca.explained_variance_ratio_
        # varsL = modelL.pca.explained_variance_ratio_

        # cumsum = np.cumsum(vars)
        # cumsumR = np.cumsum(varsR)
        # cumsumL = np.cumsum(varsL)

        # n_components = model.num_pca_components(cumsum, pca_alpha)
        # n_components_R = modelR.num_pca_components(cumsumR, pca_alpha)
        # n_components_L = modelL.num_pca_components(cumsumL, pca_alpha)

        # model.pca_var_coverage(cumsum)
        # modelR.pca_var_coverage(cumsumR)
        # modelL.pca_var_coverage(cumsumL)

        # model.elbow_plot(df_train, [1, n_components])
        # modelR.elbow_plot(df_R_train, [1, n_components_R])
        # modelL.elbow_plot(df_L_train, [1, n_components_L])


        # FIT K-MEANS MODEL #
        print('||MSG', datetime.now(), '|| RUNNING K-MEANS')

        train_labels, train_centers = model.kmeans_fit_predict(df_model)
        R_train_labels, R_train_centers = modelR.kmeans_fit_predict(df_R_model)
        L_train_labels, L_train_centers = modelL.kmeans_fit_predict(df_L_model)

        df_train_labels = pd.DataFrame(train_labels, columns=['Label'])
        df_R_train_labels = pd.DataFrame(R_train_labels, columns=['Label'])
        df_L_train_labels = pd.DataFrame(L_train_labels, columns=['Label'])

        df_train_labeled = df_train.join(df_train_labels)
        df_R_train_labeled = df_R_train.join(df_R_train_labels)
        df_L_train_labeled = df_L_train.join(df_L_train_labels)


        # LABEL TEST DATA #
        # TO DO: Cross-label data for cross-model analysis
        print('||MSG', datetime.now(), '|| LABELING TEST DATASETS')

        df_test = pipeline.transform_data(df_test, features)
        df_L_test = pipelineL.transform_data(df_L_test, features)
        df_R_test = pipelineR.transform_data(df_R_test, features)

        df_model_test = model.pca_transform(df_test, features)
        df_R_model_test = modelR.pca_transform(df_R_test, features)
        df_L_model_test = modelL.pca_transform(df_L_test, features)

        test_labels = model.kmeans_predict(df_model_test)
        test_labels_xR = modelR.kmeans_predict(df_model_test)
        test_labels_xL = modelL.kmeans_predict(df_model_test)
        R_test_labels = modelR.kmeans_predict(df_R_model_test)
        R_test_labels_xAll = model.kmeans_predict(df_R_model_test)
        R_test_labels_xL = modelL.kmeans_predict(df_R_model_test)
        L_test_labels = modelL.kmeans_predict(df_L_model_test)
        L_test_labels_xAll = model.kmeans_predict(df_L_model_test)
        L_test_labels_xR = modelR.kmeans_predict(df_L_model_test)

        df_test_labels = pd.DataFrame(test_labels, columns=['Label_All'])
        df_test_labels_xR = pd.DataFrame(test_labels_xR, columns=['Label_R'])
        df_test_labels_xL = pd.DataFrame(test_labels_xL, columns=['Label_L'])
        df_R_test_labels = pd.DataFrame(R_test_labels, columns=['Label_R'])
        df_R_test_labels_xAll = pd.DataFrame(R_test_labels_xAll, columns=['Label_All'])
        df_R_test_labels_xL = pd.DataFrame(R_test_labels_xL, columns=['Label_L'])
        df_L_test_labels = pd.DataFrame(L_test_labels, columns=['Label_L'])
        df_L_test_labels_xAll = pd.DataFrame(L_test_labels_xAll, columns=['Label_All'])
        df_L_test_labels_xR = pd.DataFrame(L_test_labels_xR, columns=['Label_R'])

        df_test_labeled = df_test.join(df_test_labels)
        df_test_labeled = df_test_labeled.join(df_test_labels_xR)
        df_test_labeled = df_test_labeled.join(df_test_labels_xL)

        df_R_test_labeled = df_R_test.join(df_R_test_labels)
        df_R_test_labeled = df_R_test_labeled.join(df_R_test_labels_xAll)
        df_R_test_labeled = df_R_test_labeled.join(df_R_test_labels_xL)

        df_L_test_labeled = df_L_test.join(df_L_test_labels)
        df_L_test_labeled = df_L_test_labeled.join(df_L_test_labels_xAll)
        df_L_test_labeled = df_L_test_labeled.join(df_L_test_labels_xR)


        # OUTPUT RESULTS TO DB/CSV #
        print('||MSG', datetime.now(), '|| WRITING RESULTS TO DB')

        db = SessionManager()
        conn = db.session.connection()
        _engine = db.engine

        @event.listens_for(_engine, 'before_cursor_execute')
        def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
            if executemany:
                cursor.fast_executemany = True

        table_name_train = 'Model_' + version + '_Train_All_Labeled'
        table_name_train_R = 'Model_' + version + '_Train_R_Labeled'
        table_name_train_L = 'Model_' + version + '_Train_L_Labeled'
        table_name_test = 'Model_' + version + '_Test_All_Labeled'
        table_name_test_R = 'Model_' + version + '_Test_R_Labeled'
        table_name_test_L = 'Model_' + version + '_Test_L_Labeled'

        df_train_labeled.to_sql(table_name_train, conn, schema='output', if_exists='replace', index=False, dtype={col_name: mssql.FLOAT for col_name in df_train_labeled})
        df_R_train_labeled.to_sql(table_name_train_R, conn, schema='output', if_exists='replace', index=False, dtype={col_name: mssql.FLOAT for col_name in df_R_train_labeled})
        df_L_train_labeled.to_sql(table_name_train_L, conn, schema='output', if_exists='replace', index=False, dtype={col_name: mssql.FLOAT for col_name in df_L_train_labeled})
        df_test_labeled.to_sql(table_name_test, conn, schema='output', if_exists='replace', index=False, dtype={col_name: mssql.FLOAT for col_name in df_test_labeled})
        df_R_test_labeled.to_sql(table_name_test_R, conn, schema='output', if_exists='replace', index=False, dtype={col_name: mssql.FLOAT for col_name in df_R_test_labeled})
        df_L_test_labeled.to_sql(table_name_test_L, conn, schema='output', if_exists='replace', index=False, dtype={col_name: mssql.FLOAT for col_name in df_L_test_labeled})

        db.session.commit()

        csv_name_train = 'Model_' + version + '_Train_All_Labeled.csv'
        csv_name_train_R = 'Model_' + version + '_Train_R_Labeled.csv'
        csv_name_train_L = 'Model_' + version + '_Train_L_Labeled.csv'
        csv_name_test = 'Model_' + version + '_Test_All_Labeled.csv'
        csv_name_test_R = 'Model_' + version + '_Test_R_Labeled.csv'
        csv_name_test_L = 'Model_' + version + '_Test_L_Labeled.csv'

        csv_path_train = repo / 'src/PyPitch/output' / version / 'data' / csv_name_train
        csv_path_train_R = repo / 'src/PyPitch/output' / version / 'data' / csv_name_train_R
        csv_path_train_L = repo / 'src/PyPitch/output' / version / 'data' / csv_name_train_L
        csv_path_test = repo / 'src/PyPitch/output' / version / 'data' / csv_name_test
        csv_path_test_R = repo / 'src/PyPitch/output' / version / 'data' / csv_name_test_R
        csv_path_test_L = repo / 'src/PyPitch/output' / version / 'data' / csv_name_test_L

        df_train_labeled.to_csv(csv_path_train)
        df_R_train_labeled.to_csv(csv_path_train_R)
        df_L_train_labeled.to_csv(csv_path_train_L)
        df_test_labeled.to_csv(csv_path_test)
        df_R_test_labeled.to_csv(csv_path_test_R)
        df_L_test_labeled.to_csv(csv_path_test_L)


        # SAVE MODEL, OUTPUT DATA, AND PIPELINE OBJECTS #
        print('||MSG', datetime.now(), '|| SAVING PIPLINE AND MODEL CLASS INSTANCES')

        model_out = str(repo / 'src/PyPitch/output' / version / 'data/model.pkl')
        modelR_out = str(repo / 'src/PyPitch/output' / version / 'data/modelR.pkl')
        modelL_out = str(repo / 'src/PyPitch/output' / version / 'data/modelL.pkl')

        pipeline_out = str(repo / 'src/PyPitch/output' / version / 'data/pipeline.pkl')
        pipelineR_out = str(repo / 'src/PyPitch/output' / version / 'data/pipelineR.pkl')
        pipelineL_out = str(repo / 'src/PyPitch/output' / version / 'data/pipelineL.pkl')

        save_pickle(model_out, model)
        save_pickle(modelR_out, modelR)
        save_pickle(modelL_out, modelL)

        save_pickle(pipeline_out, pipeline)
        save_pickle(pipelineR_out, pipelineR)
        save_pickle(pipelineL_out, pipelineL)




        return 0

    except Exception as e:
        print('||ERR', datetime.now(), '|| ERROR MESSAGE:', e)


        return 1

