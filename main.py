"""Main workflow for 'hacking heat'"""


import urllib2
import io
import time
import datetime
import lasagne_example as network
import numpy as np

from apiclient.discovery import build
from PIL import Image


def get_credentials(f_name="credentials.conf"):
	credentials = {}
	with open(f_name, "r") as f:
		for line in f.readlines():
			key_, value_ = line.split("=")
			credentials[key_.rstrip().lstrip()] = value_.rstrip().lstrip()
	return credentials

def _img_stream(tag, credentials, size, startIndex=1):
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
        num=size,
        start=startIndex,
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
    
def _adjust_img(img, size, flatten, greyscale):
	"""Preprocess the image: Resize, flatten, greyscale"""

	img = img.resize(size, Image.ANTIALIAS)
	if greyscale:
		img = img.convert('L')
	numeric_img = np.asarray(img)
	# not exactly sure how to handle that in
	# order to prevent dimension mismatch later
	if flatten:
	    numeric_img = numeric_img.reshape(-1)
	else:
		numeric_img = numeric_img.reshape(-1,size[0],size[1])
	return numeric_img
       
def _get_img(path):
	"""Create Image object from path"""

	if path.startswith("http"):
		imgRequest = urllib2.Request(path)#, headers=headers)
		try:
		    imgData = urllib2.urlopen(imgRequest).read()
		except urllib2.HTTPError, e:
		    print "\nError at %s:\n%s" % (path, e)
		    return False
		image_file = io.BytesIO(imgData)
		try:
		    img = Image.open(image_file)
		except IOError, e:
		    print "\nError at %s:\n%s" % (path, e)
		    return False
	else:
		try:
			img = Image.open(path)
		except IOError, e:
			print "\nError at %s:\n%s" % (path, e)
			return False

	img = img.convert('RGB')
	return img

def _read_file(f_name):
  """Open file and extract content"""

  with open(f_name, "r") as f:
    content = f.readlines()
  content = [line.split(',') for line in content if line not in ["\n",]]
  paths, labels = zip(*content)
  labels = [label.rstrip().lstrip() for label in labels]
  return paths, labels

def _load_dataset(f_name, img_size, greyscale, flatten):
	"""Load the dataset from f_name and preprocess images"""

	img_paths, labels = _read_file(f_name)
	raw_imgs = [_get_img(path) for path in img_paths]
	processed_labels = [labels[i] for i in range(len(labels)) if raw_imgs[i]]
	processed_imgs = np.array([_adjust_img(raw_img, img_size, greyscale=greyscale,
		flatten=flatten) for raw_img in raw_imgs if raw_img], dtype='uint8')
	return processed_imgs, processed_labels

def _load_testset(f_name, img_size, greyscale, flatten):
	"""Load the testset from f_name and preprocess images"""

	with open(f_name, "r") as f:
		img_paths = f.readlines()
	img_paths = [img_path.rstrip() for img_path in img_paths]
	raw_imgs = [_get_img(path) for path in img_paths]
	processed_imgs = np.array([_adjust_img(raw_img, img_size, greyscale=greyscale,
		flatten=flatten) for raw_img in raw_imgs if raw_img], dtype='uint8')
	return processed_imgs, img_paths

def _gen_lookup_table(label_names):
	"""Build lookup table to convert numerical to string labels"""

	lookup = {}
	rev_lookup = {}
	all_labels = set(label_names)
	for i, label in enumerate(all_labels):
		lookup[i] = [label]
		rev_lookup[label] = i
	return lookup, rev_lookup

def build_database(fname, size, tags, startIndex=1):
        """Build an image database from Google image search.
        Arguments:
        fname (str): Filename; if the file already exists, the database will be extended
    size (int): Number of images to retrieve per tag
        tags (list): List of Google search terms
        startIndex (int, optional): Index, from which image to start

        Example:
        build_database('training_set.csv', 20, ['apples', 'oranges'], startIndex=20)
        """

        if size < 1:
                raise ValueError, "size must be int greater than 1"

        credentials = get_credentials()
        num_imgs = 0
        for tag in tags:
                urls = [[]] * size
                i = 0
                for img in _img_stream(tag, credentials=credentials, size=size, startIndex=startIndex):
                        urls[i] = img
                        i += 1
                        if i >= size:
                                break
                with open(fname, 'a') as f:
                        for url in urls:
                                f.write(",".join(url))
                                f.write("\n")
                num_imgs += len(urls)
        return num_imgs






def build_network(f_train, f_val, f_test, img_size, greyscale=False, flatten=False,
	architecture="mlp", num_epochs=500):
	"""Build and train the network with the given image sets.
	Arguments:
	f_train (str): path to the training set file
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

   	X_train, y_train_n = _load_dataset(f_train, img_size=img_size, greyscale=greyscale, 
   		flatten=flatten)
   	X_val, y_val_n = _load_dataset(f_val, img_size=img_size, greyscale=greyscale, 
   		flatten=flatten)
   	X_test, test_urls = _load_testset(f_test, img_size=img_size, greyscale=greyscale, 
   		flatten=flatten)
   	lookup, rev_lookup = _gen_lookup_table(y_train_n)
   	y_train = np.array([rev_lookup[label] for label in y_train_n], dtype='uint8')
   	y_val = np.array([rev_lookup[label] for label in y_val_n], dtype='uint8')
   	if greyscale:
   		channels = 1
   	else:
   		channels = 3
   	predictions = network.main(X_train, y_train, X_val, y_val, X_test,
   		channels=channels, size=img_size, model=architecture, num_epochs=num_epochs)
   	pred_names = [lookup[lbl] for lbl in predictions]
   	pred_table = zip(test_urls, pred_names)
   	return pred_table

def deploy_builder(fname, size, tags, startIndex=1):
  """Runs the script until all results are fetched"""

  chunk_size = 10
  wait_time = 100
  for tag in tags:
    for cycle in xrange(int(size/chunk_size)):
      build_database(fname=fname, size=chunk_size, tags=[tag,], startIndex=(cycle*chunk_size+1))

      print "Total images: (%s/%s)\n Now looking for '%s'" % (cycle*chunk_size+1, size*len(tags), tag)
      time.sleep(wait_time)
    #if retrieved_imgs % daily_quota == 0:
    #  for i in range(24):
    #    print "Waking up in %s hours... (interrupt with ctrl+c)\r" % (24-i)
    #    time.sleep(60*60)
      

    

if __name__ == "__main__":
    f_train = "train.csv"
    f_val = "validation.csv"
    f_test = "test.csv"
    img_size = (128,128)
    #print build_network(f_train, f_val, f_test, img_size, greyscale=False, flatten=False,
	#	architecture="mlp", num_epochs=5)
    #deploy_builder('ofen2.csv', size=500, tags=['vedovn', 'rundbrenner ovn','peis'], startIndex=1)
    #build_database('ofen.csv', size=100, tags=['peis', 'vedovn','rundbrenner ovn'], startIndex=1)
    tags = [#'Contura 510 Style', 
      #'Contura Style', 'Morso 6140', 'Morso 7440', 'Termatech TT20 Bazic', 
      #'Rais Viva', 'Contura 510 Style', 'Contura Style', 'modern fireplace mantels', 'modern marble fireplace mantels',
      #' modern stone fireplace', 'modern rustic stone fireplace', 'modern stone fireplace surround', 
      'traditional fireplaces', 'traditional brick fireplace designs', 'cast iron wood stove', 'gussofen alt']
    set_size = 1000
    n_requests = 0
    credentials = get_credentials('credentials.conf')
    for tag in tags:
		with open('%s.csv' % tag, 'a') as f:
			for i in xrange(set_size/10):
				startIndex = 1+i*10
				if startIndex > 91:
					break
				for img in _img_stream(tag, credentials=credentials, size=10, startIndex=startIndex):
					f.write(",".join(img))
					f.write("\n")
				n_requests += 10
				time.sleep(1)
				#if n_requests > 91:
				#	print "going to sleep"
				#	time.sleep(100)



