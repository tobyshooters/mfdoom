### Gucci Gang and the Rise of Mumble Rap
#### An ensemble ANN-RNN approach to analyzing hip-hop song lyrics

Code:
- `learn_nn.py` and `learn_rnn.py` are the bulk of the machine learning of the project
- `features_nn.py` and `features_rnn.py` have the logic to extract features from the raw lyrics and structure them for consumption by an ANN or RNN respectively
- `feature_util.py` contains the overlapping logic for ANN and RNN feature extraction
- `baseline.py` performs trivial baseline calculations on the dataset
- `cluster.py` has logic for K-means clustering of the datapoints for additional analysis
- `visualize.py` generates plots of features and clusters
- `scraper/songs.py` extracts song titles from Billboards and stores them in a local database
- `scraper/lyrics.py` scrapes lyrics off of the Genius website
- `util.py` contains general utilities used throughout the project

Datasets:
- `data/1990` is a SQLite dataset containing songs (title, artist, peak position, time on charts, raw lyrics)
- `data/nn_features`, `data/rnn_features`, and `data/cluster` are text files serving as caches of feature extraction

Libraries used: keras, sqlite3, scipy, numpy, nltk, bs4
