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
from tsne import generate_tsne

DATASET_PATH = r"D:\Projects\PyCharmProjects\watsonTraining\images_google\101_ObjectCategories\101_ObjectCategories"
# Increase the number if you want to scan all the repo
MAX_IMAGES_TO_SEARCH_FROM_REPO = 250
PICKLED_MODEL_FILENAME = 'model.sav'
QUERY_IMAGE_PATH = r"D:\Projects\PyCharmProjects\watsonTraining\images_google\gun.jpg"
IMAGE_SIZE = (224, 224)
NUMBER_OF_RESULTS = 3


def load_model():
    """
    Loads the pre-trained model.
    :return: the model
    """
    if not os.path.exists(PICKLED_MODEL_FILENAME):
        # save the model to disk
        print("Loading model and dumping it to the disk")
        model = VGG16(weights='imagenet', include_top=False)
        pickle.dump(model, open(PICKLED_MODEL_FILENAME, 'wb'))
    else:
        # load the model from disk if already exists.
        print("Loading model from the disk")
        model = pickle.load(open(PICKLED_MODEL_FILENAME, 'rb'))
    return model


def img_to_data(img_path):
    """
    Converts img to data(array)
    :param img_path: the path of the input image.
    :return: data of an image after preprocess.
    """
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    return img_data


def img_to_features_vector(img_data, model):
    """
    Feeding the image data to the model.
    :param img_data: the data of an image.
    :return: the vector the model outputs.
    """
    vgg16_feature = model.predict(img_data)
    vgg16_feature = np.array(vgg16_feature)
    return vgg16_feature.flatten()


def main(query_image_path=QUERY_IMAGE_PATH, dataset_path=DATASET_PATH, to_generate_tsne=True):
    model = load_model()
    image_paths = list(paths.list_images(dataset_path))
    shuffle(image_paths)
    img_paths = image_paths[:MAX_IMAGES_TO_SEARCH_FROM_REPO]
    img_vector_features = []
    for img_path in tqdm(img_paths):
        # convert image to data in order to enable to feed the image to the model.
        img_data = img_to_data(img_path)
        # get from the model the features vector returned from the vgg16 model.
        img_vector_features.append(img_to_features_vector(img_data, model))

    query_img_data = img_to_data(query_image_path)
    query_feature = img_to_features_vector(query_img_data, model)

    # Numbers of similar images that we want to show
    nbrs = NearestNeighbors(n_neighbors=NUMBER_OF_RESULTS, metric="cosine").fit(img_vector_features)

    distances, indices = nbrs.kneighbors([query_feature])
    similar_image_indices = indices.reshape(-1)
    img = mpimg.imread(QUERY_IMAGE_PATH)
    imgplot = plt.imshow(img)
    plt.show()
    for idx in similar_image_indices:
        print(img_paths[idx])
        img = mpimg.imread(img_paths[idx])
        imgplot = plt.imshow(img)
        plt.show()

    if to_generate_tsne:
        print("Generating TSNE...")
        generate_tsne(img_paths, img_vector_features)


if __name__ == '__main__':
    main()
