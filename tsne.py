from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from imutils import paths
from random import shuffle
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.image as mpimg
import os.path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from matplotlib.pyplot import imshow
from PIL import Image
import matplotlib
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage

TSNE_IMG_FILENAME = "example-tSNE-caltech101.png"
DISTANCE_METRIC = 'euclidean'
FEATURE_DIMENSION_NUMBER = 100


def generate_tsne(img_paths, img_vector_features):
    # Perform PCA over the features
    num_feature_dimensions = FEATURE_DIMENSION_NUMBER  # Set the number of features
    pca = PCA(n_components=num_feature_dimensions)
    pca.fit(img_vector_features)
    feature_list_compressed = pca.transform(img_vector_features)

    # For speed and clarity, we'll analyze about first half of the dataset.
    selected_features = feature_list_compressed[:]
    selected_filenames = img_vector_features[:]

    tsne_results = TSNE(n_components=2, verbose=1, metric=DISTANCE_METRIC).fit_transform(selected_features)

    X = np.array(feature_list_compressed)
    tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2, verbose=2).fit_transform(X)

    tx, ty = tsne[:, 0], tsne[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

    width = 4000
    height = 3000
    max_dim = 100

    full_image = Image.new('RGBA', (width, height))
    for img, x, y in zip(img_paths, tx, ty):
        tile = Image.open(img)
        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        tile = tile.resize((int(tile.width / rs), int(tile.height / rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width - max_dim) * x), int((height - max_dim) * y)), mask=tile.convert('RGBA'))

    matplotlib.pyplot.figure(figsize=(16, 12))
    imshow(full_image)
    try:
        matplotlib.pyplot.show(full_image)
    except:
        pass
    full_image.save(TSNE_IMG_FILENAME)
