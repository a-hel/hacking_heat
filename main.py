"""Main workflow for 'hacking heat'"""


import urllib2
import io
import csv
#import cv2
import lasagne_example as network
reload(network)
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

from apiclient.discovery import build
from PIL import Image
from functools import partial

def get_credentials(f_name="credentials.conf"):
	credentials = {}
	with open(f_name, "r") as f:
		for line in f.readlines():
			key_, value_ = line.split("=")
			credentials[key_.rstrip().lstrip()] = value_.rstrip().lstrip()
	return credentials

def _img_stream(tag, credentials, startIndex=0):
    """Search google images for 'tag'.
    Arguments:
    tag (str): Google images search term
    credentials: 
    startIndex (int, optional): First image to retrieve

    Returns: The next image
    """


    service = build("customsearch", "v1",
               developerKey=credentials['google_key'])

    res = service.cse().list(
        q=tag,
        cx=credentials['cx'],
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
            yield (url, tag)

def fetch_img(url):
    imgRequest = urllib2.Request(url)#, headers=headers)
    try:
        imgData = urllib2.urlopen(imgRequest).read()
    except urllib2.HTTPError, e:
        print "Error at %s:\n%s" % (url, e)
        return False
    image_file = io.BytesIO(imgData)
    try:
        img = Image.open(image_file)
    except IOError, e:
        print "Error at %s:\n%s" % (url, e)
        return False    
    return img

def open_img(path):
    try:
        img = Image.open(path)
    except IOError, e:
        print "Error at %s:\n%s" % (path, e)
        return False    
    return img


def write_url_list(fname, size, tags):
    #output_data = [[None]] * (size*len(tags))
    output_data = []
    for t, tag in enumerate(tags):
        i = 0
        for img_url in img_stream(tag):
            pos = i + t*size
            #output_data[pos] = img_url
            i += 1
            output_data.append(img_url)
            if i >= size:
                break
            #output_data[pos][1] = tag
    with open(fname, 'a') as f:
        for line in output_data:
            f.write(",".join(line))
            f.write("\n")
    return fname
    
def _adjust_img(img, size=None, flatten=False, greyscale=True, alpha=False):
	#accomodate for alpha channel
    if img == False:
        numeric_img = np.zeros((size[0], size[1], 4), dtype="uint8")
    else:
    	if greyscale:
        	img = img.convert('L')
    	if size:
        	img = img.resize(size, Image.ANTIALIAS)
    	numeric_img = np.asarray(img)
    if flatten:
        numeric_img = numeric_img.reshape(-1)
    ret_val = numeric_img.reshape(-1,28,28)
    return ret_val[0:1,:,:]
    #return numeric_img[:,:,0:1]#,0:3] #Correct greyscale shit
        
def load_imgs(img_no, source="local", tags=None, fname=None, size=(28,28),
	flatten=False, greyscale=False, alpha=False):
    if source == "google":
        size_per_label = int(img_no/len(tags))
    elif source == "web":
        with open(fname, "r") as f:
            reader = csv.reader(f, delimiter=',')
            urls = []
            labels = []
            for row in reader:
                urls.append(row[0])
                labels.append(int(row[1]))
                #urls = urls[0:img_no]
                #labels = labels[0:img_no]
    	labels = np.array(labels, dtype="uint8")
    	training_imgs = map(fetch_img, urls)
    elif source == "local":
    	with open(fname, "r") as f:
            reader = csv.reader(f, delimiter=',')
            paths = []
            labels = []
            for row in reader:
                paths.append(row[0])
                labels.append(int(row[1]))
                #urls = urls[0:img_no]
                #labels = labels[0:img_no]
    	labels = np.array(labels, dtype="uint8")
    	training_imgs = map(open_img, paths)
    
    mapfunc = partial(_adjust_img, size=size, flatten=flatten, greyscale=greyscale,
    	alpha=alpha)
    training_imgs = map(mapfunc, training_imgs)

    img_array = np.array(training_imgs)
    return img_array, labels

def build_database(fname, size, tags, startIndex=0):
	"""Build an image database from Google image search.
	Arguments:
	fname (str): Filename; if the file already exists, the database will be extended
	size (int): Number of images to retrieve per tag
	tags (list): List of Google search terms
	startIndex (int, optional): Index, from which image to start

	Example:
	build_database('training_set.csv', 20, ['apples', 'oranges'], startIndex=20)
	"""

	if size < 1:
		raise ValueError, "size must be int greater than 1"

	credentials = get_credentials()
	for tag in tags:
		urls = [[]] * size
		i = 0
		for img in _img_stream(tag, credentials=credentials, startIndex=startIndex):
			urls[i] = img
			i += 1
			if i >= size:
				break
		with open(fname, 'a') as f:
			for url in urls:
				f.write(",".join(url))
				f.write("\n")

def build_network(f_train, f_val, f_test, img_size, greyscale=False, flatten=False,
	architecture="mlp", num_epochs=500):
	"""Build and train the network with the given image sets.
	Arguments:
	f_train (str): path to the training set file
	f_val (str): path to the validation set file
	f_test (str): path to the test set file
	img_size (tuple): size to which images will be resized
	greyscale (bool, optional): if True, images will be converted to greyscale
	flatten (bool, optional): if True, images will be flattened
	architecture (str, optional): Either 'mlp' (multi-layer perceptron)
		or 'cnn' (Convoluted neural network)
	num_epochs (int, optional): Number of training cycles

	Example:
	build_network('training.csv', 'validation.csv', 'test.csv', (128,128), num_epochs=100)
	"""
    pass
        



if __name__ == "__main__":
    credentials = get_credentials()
    #fname = "../DATA/urls.csv"
    fname = "test.csv"
    size = 5
    tags = ['Apple', 'Orange']
    build_database(fname, size=size, tags=tags)

 

        
    #network.main(X_train, y_train, X_val, y_val, X_test, y_test, num_epochs=5)
    #image = fetch_img('https://d3nevzfk7ii3be.cloudfront.net/igi/KRLMkuaBjm5mKDDP')
    #print _correct_img(image, size=(128,128), greyscale=False, flatten=False)
