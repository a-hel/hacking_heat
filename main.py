"""Main workflow for 'hacking heat'"""

import numpy as np
import matlab

def img_stream(tag):
    """Search google images for 'tag'.
    Arguments:
    tag (str): Google images search term

    Returns: The next image
    """
    yield tag

def feature_extraction(img, feat_type="HOG"):
    """Extract the features from an image.
    Arguments:
    img (np.array): The greyscale image as a 2-dimensional numpy array
    feat_type (str, optional): The type of features to extract.

    Returns: Features according to feat_type
    """

    return img

def matlab():
    """Initialize MATLAB engine.
    Install matlab engine:
    http://se.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
    """

    eng = matlab.engine.start_matlab()
    return eng

if __name__ == "__main__":
    eng = matlab()
    print eng.isprime(37)
