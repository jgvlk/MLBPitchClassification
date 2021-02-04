from PyPitch.classification.utils import EDA
from PyPitch.classification.pipeline import Pipeline


# output_dir_data = repo / 'src/PyPitch/output' / version / 'data'
# output_dir_viz = repo / 'src/PyPitch/output' / version / 'viz'


df, features = Pipeline.run()

# # CHECK NULLS #
# null_cols = EDA.null_check(df_data_model)

# if null_cols:
#     print('||WARN', datetime.now(), '|| NULL FEATURE VALUES EXIST')
# else:
#     print('||MSG', datetime.now(), '|| NO NULL FEATURE VALUES EXIST')



# # PANDAS DESCRIBE #
# out_file_describe = output_dir_data / 'Model_RawData_Describe2.csv'
# EDA.describe(df_data_model, out_file_describe)

# out_file_describe = output_dir_data / 'Model_RawData_Describe2_R.csv'
# EDA.describe(df_data_model_R, out_file_describe)

# out_file_describe = output_dir_data / 'Model_RawData_Describe2_L.csv'
# EDA.describe(df_data_model_L, out_file_describe)


# # PANDAS PROFILING #
# out_file_profile = output_dir_data / 'profile2.html'
# EDA.profile(df_data_model, out_file_profile)

# out_file_profile = output_dir_data / 'profile2_R.html'
# EDA.profile(df_data_model_R, out_file_profile)

# out_file_profile = output_dir_data / 'profile2_L.html'
# EDA.profile(df_data_model_L, out_file_profile)


# # FEATURE DENSITY PLOTS #
# EDA.feature_density_plot(df_data_model)
# EDA.feature_density_plot(df_data_model_R)
# EDA.feature_density_plot(df_data_model_L)


# # CORRELATION ANALYSIS #
# EDA.correlation_analysis(df_data_model, features)
# EDA.correlation_analysis(df_data_model_R, features)
# EDA.correlation_analysis(df_data_model_L, features)

# # Checking differences in corr in split datasets
# top10_neg, top10_pos = EDA.correlation_rank(df_data_model, features)
# top10_neg_R, top10_pos_R = EDA.correlation_rank(df_data_model_R, features)
# top10_neg_L, top10_pos_L = EDA.correlation_rank(df_data_model_L, features)

# d_corr = {}
# for i in range(10):
#     d_corr['neg_{}'.format(i)] = {'all data': top10_neg[i], 'righties': top10_neg_R[i], 'lefties': top10_neg_L[i]}
#     d_corr['pos_{}'.format(i)] = {'all data': top10_pos[i], 'righties': top10_pos_R[i], 'lefties': top10_pos_L[i]}
