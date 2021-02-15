import pickle


with open('/Users/jonathanvlk/dev/MLBPitchClassification/src/PyPitch/output/v1/data/modelR.pkl', 'rb') as pickle_file:
    _model = pickle.load(pickle_file)


_model.pca.components_

