import pickle

# Global minimum values for each name
GLOBAL_MINIMUM = {
    'branin2': 0.397887,
    'hartmann3': -3.86278,
    'griewank5': 0,
    'robot3': 0,
    'robot4': 0,
    'hartmann6': -3.32237,
    'exp10': 0,
}


BAR_DICT = {
    'branin2': {
        "errorevery": 1,
        "beta": 0.5
    },
    'hartmann3': {
        "errorevery": 1,
        "beta": 0.5
    },
    'robot3': {
        "errorevery": 5,
        "beta": 0.5
    },
    'robot4': {
        "errorevery": 10,
        "beta": 0.5
    },
    'hartmann6': {
        "errorevery": 5,
        "beta": 0.5
    },
    'exp10': {
        "errorevery": 10,
        "beta": 0.5
    },
}

PARAMS_DICT = {
    "rbf": {
        "label": "RBF",
        "marker": "x",
        "linestyle": "dotted",
        "color": u"#1f77b4",
    },
    "rq": {
        "label": "RQ",
        "marker": "x",
        "linestyle": "dashed",
        "color": u"#ff7f0e",
    },
    "matern": {
        "label": "MA",
        "marker": "x",
        "linestyle": "dashdot",
        "color": u"#2ca02c",
    },
    "ABO": {
        "label": "ABO",
        "marker": "x",
        "linestyle": (0, (5, 10)),
        "color": u"#d62728",
    },
    "csm7": {
        "label": "CSM7",
        "marker": "p",
        "color": u"#8c564b",
    },
    "gsm7": {
        "label": "GSM7",
        "marker": "+",
        "color": u"#e377c2",
    },
    "csm3": {
        "label": "CSM3",
        "marker": "x",
        "linestyle": "dotted",
        "color": u"#1f77b4",
    },
    "csm5": {
        "label": "CSM5",
        "marker": "x",
        "linestyle": "dashed",
        "color": u"#ff7f0e",
    },
    "csm9": {
        "label": "CSM9",
        "marker": "x",
        "linestyle": "dashdot",
        "color": u"#2ca02c",
    },
    "gsm3": {
        "label": "GSM3",
        "marker": "x",
        "linestyle": (0, (5, 10)),
        "color": u"#d62728",
    },
    "gsm5": {
        "label": "GSM5",
        "marker": "x",
        "linestyle": (0, (3, 10, 1, 10)),
        "color": u"#9467bd",
    },
    "gsm9": {
        "label": "GSM9",
        "marker": "s",
        "linestyle": "dashdot",
        "color": u"#7f7f7f",
    },
}


# Function to load pickle file and extract results
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


# Function to parse filename into components
def parse_filename(filename):
    parts = filename.split('_')
    name = parts[0]
    kernel = parts[1]
    acq = parts[2].split('.')[0]  # Remove ".pkl"
    return name, kernel, acq
