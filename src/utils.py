import yaml
import joblib

CONFIG_DIR = 'config/config.yaml'

def load_config():
    "Create function to load config files"
    with open(CONFIG_DIR, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_pickle(path_file):
    "Create function to load the pickel files"
    return joblib.load(path_file)

def dump_pickle(data, path_file):
    "Create function to dump data into pickle"
    joblib.dump(data, path_file)