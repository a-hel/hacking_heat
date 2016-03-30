"""Main workflow for 'hacking heat'"""

from __future__ import print_function

import urllib2
import io
import time
import datetime
import h5py
import warnings
import numpy as np
from PIL import Image
try:
	from apiclient.discovery import build
except ImportError:
	no_google_warn = """Google API not installed."""
	warnings.warn(no_google_warn, ImportWarning)


def get_credentials(f_name="credentials.conf"):
	"""Load the credentials for Google Custom Search API from f_name.
	"""
	credentials = {}
	with open(f_name, "r") as f:
		for line in f.readlines():
			if line == "\n":
				continue
			key_, value_ = line.split("=")
			credentials[key_.strip()] = value_.strip()
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
		print('No result !\nSearch returned: {}'.format(res))
	else:
		for item in res['items']:
			url = item['link']
			yield (url, tag)

def write_url_list(fname, size, tags):
	output_data = []
	for t, tag in enumerate(tags):
		i = 0
		for img_url in _img_stream(tag):
			pos = i + t*size
			i += 1
			output_data.append(img_url)
			if i >= size:
				break
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

	try:
		if path.startswith("http"):
			imgRequest = urllib2.Request(path)#, headers=headers)
			imgData = urllib2.urlopen(imgRequest).read()
			image_file = io.BytesIO(imgData)
			img = Image.open(image_file)
		else:
			img = Image.open(path)
		img = img.convert('RGB')
	except Exception, e:
		print("Error when processing image\n%s\n> %s\n" % (path, e))
		return False

	return img

def _read_file(f_name):
	"""Open file and extract content"""

	with open(f_name, "r") as f:
		content = [line.split(',') for line in f if line not in ["\n",]]
		paths, labels = zip(*content)
		labels = [label.strip() for label in labels]
	return paths, labels

def _load_dataset(f_name, img_size, greyscale, flatten):
	"""Load the dataset from f_name and preprocess images"""

	img_paths, labels = _read_file(f_name)
	raw_imgs = [_get_img(path) for path in img_paths]
	processed_labels = [labels[i] for i in range(len(labels)) if raw_imgs[i]]
	processed_imgs = np.array([_adjust_img(raw_img, img_size,
		greyscale=greyscale,
		flatten=flatten) for raw_img in raw_imgs if raw_img], dtype='Float64')
	return processed_imgs, processed_labels

def _load_testset(f_name, img_size, greyscale, flatten):
	"""Load the testset from f_name and preprocess images"""

	with open(f_name, "r") as f:
		img_paths = [img_path.rstrip() for img_path in f]
	raw_imgs = [_get_img(path) for path in img_paths]
	processed_imgs = np.array([_adjust_img(raw_img, img_size, greyscale=greyscale,
		flatten=flatten) for raw_img in raw_imgs if raw_img], dtype='Float64')
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
	fname (str): Filename; if the file already exists, the database will be
	extended
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
		for img in _img_stream(tag, credentials=credentials, size=size,
			startIndex=startIndex):
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

def show_set(fname, target):
	urls, labels = _read_file(fname)
	lines = zip(urls, labels)
	write_to_html(lines, target)

def write_to_html(classes, fname):

	header = """
<html>
<head>
<title>Predictions</title>
</head>
<body>"""
	footer = """
</body>
</html>"""
	img_line = """
	<p><a href='%s', title='%s'>
	<img width=128, height=128, src='%s'></a>%s</p>
	"""
	with open(fname, 'w') as f:
		f.write(header)
		for (url, tag) in classes:
			f.write(img_line % (url, tag[0], url, tag[0]))
		f.write(footer)
	return fname

def build_network(f_train, f_val, f_test, img_size, greyscale=False, flatten=False,
	num_epochs=500):
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
		channels=channels, size=img_size, num_epochs=num_epochs)
	pred_names = [lookup[lbl] for lbl in predictions]
	pred_table = zip(test_urls, pred_names)
	return pred_table

def img_downloader(sourcefile, target_folder, base_folder="IMG/",
	filetype="JPEG"):
	"""Download the images from sourcefile to the target_folder."""

	
	img_paths, labels = _read_file(base_folder + sourcefile)
	raw_imgs = [_get_img(path) for path in img_paths]
	new_paths = [''] * len(img_paths)
	for i, raw_img in enumerate(raw_imgs):
		path = "%s%s/IMG_%s.%s" % (base_folder, target_folder, i, filetype)
		try:
			raw_img.save(path, filetype)
		except AttributeError:
			continue
		new_paths[i] = path
	newfile = "%s/local_%s" % (base_folder, sourcefile)
	paths_labels = zip(new_paths, labels)
	with open(newfile, "w") as f:
		for path_label in paths_labels:
			f.write(','.join(path_label))
			f.write('\n')
	return True




def deploy_builder(fname, size, tags, startIndex=1):
  """Runs the script until all results are fetched"""

  chunk_size = 10
  wait_time = 100
  for tag in tags:
	for cycle in xrange(int(size/chunk_size)):
		build_database(fname=fname, size=chunk_size, tags=[tag,], 
			startIndex=(cycle*chunk_size+1))

		print("Total images: (%s/%s)\n Now looking for '%s'" % \
			(cycle*chunk_size+1, size*len(tags), tag))
		time.sleep(wait_time)

def pack(name, f_name, img_size=(227,227),
		greyscale=False, flatten=False, istest=False):
	"""Pack all datasets into one HDF5 file"""
	 
	dtype = "Float64" # Should be Float64
	data_folder = "DATA"
	hdfname = "%s.hdf5" % name

	f = h5py.File("%s/%s" % (data_folder, hdfname), "w")
	if istest:
		X, paths = _load_testset(f_name, img_size=img_size,
			greyscale=greyscale, flatten=flatten)
		xfile = f.create_dataset("/data", data=X, dtype=dtype)
	else:
		X, y = _load_dataset(f_name, img_size=img_size,
			greyscale=greyscale, flatten=flatten)
		lookup, rev_lookup = _gen_lookup_table(y)
		y_n = np.array([rev_lookup[label] for label in y], dtype='uint8')
		xfile = f.create_dataset("data", data=X, dtype=dtype)
		yfile = f.create_dataset("label", data=y_n, dtype=dtype)
		for keys in lookup:
			yfile.attrs[str(keys)] = lookup[keys]

	with open("%s/%s.txt" % (data_folder, name), "w") as ref:
		ref.write("%s/%s" % (data_folder, hdfname))
	print("Created Datasets:")
	for name in f:
		print("  - %s" % name)
	print("Dimensions:")
	print("  - %s" % ", ".join(str(i) for i in X.shape))
	if not istest:
		print("  - %s" % ", ".join(str(i) for i in y_n.shape))


if __name__ == "__main__":
	f_train = "IMG/local_training2.csv"
	f_val = "IMG/local_validation.csv"
	f_test = "IMG/local_test.csv"

	#pack("oven-test", f_test, istest=True)
	pack("oven-train", f_train)
	pack("oven-val", f_val)

	#show_set(f_train, '_local/training.html')
	#show_set(f_val, '_local/validation.html')
	#5/0
	#img_size = (128,128)
	#predictions = build_network(f_train, f_val, f_test, img_size, greyscale=True, flatten=False,
	#    num_epochs=500)
	#write_to_html(predictions, "_local/predictions.html")
	#print(\a)
	#deploy_builder('ofen2.csv', size=500, tags=['vedovn', 'rundbrenner ovn','peis'], startIndex=1)
   # build_database('ofen.csv', size=100, tags=['peis', 'vedovn','rundbrenner ovn'], startIndex=1)
	#tags = [#'Contura 510 Style',
	  #'Contura Style', 'Morso 6140', 'Morso 7440', 'Termatech TT20 Bazic',
	  #'Rais Viva', 'Contura 510 Style', 'Contura Style', 'modern fireplace mantels', 'modern marble fireplace mantels',
	  #' modern stone fireplace', 'modern rustic stone fireplace', 'modern stone fireplace surround',
	#  'traditional fireplaces', 'traditional brick fireplace designs', 'cast iron wood stove', 'gussofen alt']
	#tags = ['yoshi', 'bowser']
	#set_size = 100
	#n_requests = 0
	#credentials = get_credentials('credentials.conf')
	#for tag in tags:
	#    with open('%s.csv' % tag, 'a') as f:
	#       for i in xrange(set_size/10):
	#           startIndex = 1+i*10
	#           if startIndex > 91:
	#               break
	#           for img in _img_stream(tag, credentials=credentials, size=10, startIndex=startIndex):
	#               f.write(",".join(img))
	#               f.write("\n")
	#           n_requests += 10
	#           time.sleep(1)
				#if n_requests > 91:
				#   print "going to sleep"
				#   time.sleep(100)