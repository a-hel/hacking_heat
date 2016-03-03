"""Main workflow for 'hacking heat'"""


import urllib2
import io
from apiclient.discovery import build
from PIL import Image

import numpy as np

try:
    import matlab
except ImportError:
    msg = """Matlab engine not installed: Visit

>> http://se.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

for more information."""
    print msg

def img_stream(tag, google_key="AIzaSyDckTUqtzDdKxKJqeat3HvPFwceAUVRbew", cx='009248554659861815589:hvztsnepnuc', startIndex=0):
    """Search google images for 'tag'.
    Arguments:
    tag (str): Google images search term
    startIndex (int, optional): First image to retrieve

    Returns: The next image
    """

    service = build("customsearch", "v1",
               developerKey=google_key)

    res = service.cse().list(
        q=tag,
        cx=cx,
        searchType='image',
        num=3,
        #imgType='clipart',
        #fileType='png',
        safe= 'off'
    ).execute()

    if not 'items' in res:
        print 'No result !!\nres is: {}'.format(res)
    else:
        for item in res['items']:
            #print('{}:\n\t{}'.format(item['title'], item['link']))
            url = item['link']
            imgRequest = urllib2.Request(url)#, headers=headers)
            imgData = urllib2.urlopen(imgRequest).read()
            image_file = io.BytesIO(imgData)
            greyscale_img = Image.open(image_file).convert('L')
            numeric_img = np.asarray(greyscale_img)
            yield numeric_img

def feature_extraction(img, feat_type="HOG"):
    """Extract the features from an image.
    Arguments:
    img (np.array): The greyscale image as a 2-dimensional numpy array
    feat_type (str, optional): The type of features to extract.

    Returns: Features according to feat_type
    """

    return img

def train_network(network, tags, feat_type="HOG", n_img=500, iterations=None):
    """Train the network with images.
    Arguments:
    network (pybrain.): The network to be trained
    tags (list): List of labels
    feat_type (str): Features used for feature extraction
    n_img (int, optional): Number of images used for training
    iterations (int, optional): Max number of training cycles. If omitted, will
        train until convergence.

    Returns: True after successful training

    Example:
    train_network(my_network, tags=['apples', 'oranges'], feat_type="HOG",
        n_img=250, iterations=20000)
    """

def predict_img(network, img):
    label = 'label'
    return label

    return True

def matlab_eng():
    """Initialize MATLAB engine.
    Install matlab engine:
    http://se.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
    """

    eng = matlab.engine.start_matlab()
    return eng

if __name__ == "__main__":
    #eng = matlab_eng()
    #print eng.isprime(37)
    for i in img_stream('orange'):
        print i.shape
        break
