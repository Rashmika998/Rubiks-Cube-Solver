import os.path

install_dir = os.path.dirname(os.path.realpath(__file__))
training_data_dir = os.path.join(install_dir, 'training_data')
pickled_model = os.path.join(install_dir, 'clf.joblib')
