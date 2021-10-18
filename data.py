import numpy as np
import nibabel as nib
from generator import sig2
def generator_data(image_file,st,label_file,sl):
	vol_dir = image_file + st
	image1 = nib.load(vol_dir)
	image = image1.get_data()
	affine0 = image1.affine.copy()
	image = np.asarray(image, dtype=np.float32)
	label_dir = label_file+sl
	label = nib.load(label_dir).get_data()
	label = np.asarray(label, dtype=np.float32)
	affine0 = np.asarray(affine0, dtype=np.float32)
	return image,label,affine0


def get_data(base_dir,image_file,label_file):
	train_file = open(base_dir+'train.txt')
	train_strings = train_file.readlines()
	train_label_file = open(base_dir+'train_mask.txt')
	train_label_strings = train_label_file.readlines()
	image = []
	label = []
	affine= []
	for i in range(0,len(train_strings)):
		st = train_strings[i].strip()
		sl = train_label_strings[i].strip()
		img,mask,affine1 = generator_data(image_file,st,label_file,sl)
		image.append(img)
		label.append(mask)
		affine.append(affine1)
	image = np.asarray(image, dtype=np.float32)
	label = np.asarray(label, dtype=np.float32)
	return image,label,affine

def get_valid_data(base_dir,image_file,label_file):
	train_file = open(base_dir+'valid.txt')
	train_strings = train_file.readlines()
	train_label_file = open(base_dir+'valid_mask.txt')
	train_label_strings = train_label_file.readlines()
	image = []
	label = []
	affine = []
	for i in range(0, len(train_strings)):
		st = train_strings[i].strip()
		sl = train_label_strings[i].strip()
		img, mask, affine1 = generator_data(image_file, st,  label_file, sl)
		image.append(img)
		label.append(mask)
		affine.append(affine1)
	image = np.asarray(image, dtype=np.float32)
	label = np.asarray(label, dtype=np.float32)
	return image,label,affine
def get_test_data(base_dir,image_file,label_file):
	train_file = open(base_dir+'test.txt')  # the name of test data
	train_strings = train_file.readlines()
	train_label_file = open(base_dir+'test_mask.txt')
	train_label_strings = train_label_file.readlines()
	image = []
	test_label = []
	affine = []
	for i in range(0,len(train_strings)):
		st = train_strings[i].strip()  #file name
		sl = train_label_strings[i].strip()
		img,mask1 ,affine1= generator_data(image_file,st,label_file,sl)
		image.append(img)
		test_label.append(mask1)
		affine.append(affine1)
	image = np.asarray(image, dtype=np.float32)
	test_label = np.asarray(test_label, dtype=np.float32)
	return image,test_label,affine

from munkres import Munkres
def best_map(L1,L2,img_shape):
	#L1 should be the labels and L2 should be the clustering number we got
	Label1 = np.unique(L1)       # remove repetitive
	nClass1 = len(Label1)
	Label2 = np.unique(L2)
	nClass2 = len(Label2)
	nClass = np.maximum(nClass1,nClass2)
	G = np.zeros((nClass,nClass))
	for i in range(nClass1):
		ind_cla1 = L1 == Label1[i]
		ind_cla1 = ind_cla1.astype(float)
		for j in range(nClass2):
			ind_cla2 = L2 == Label2[j]
			ind_cla2 = ind_cla2.astype(float)
			G[i,j] = np.sum(ind_cla2 * ind_cla1)
	m = Munkres()
	index = m.compute(-G.T)
	index = np.array(index)
	c = index[:,1]
	newL2 = np.zeros(img_shape)
	for i in range(nClass2):
		newL2[L2 == Label2[i]] = Label1[c[i]]
	return newL2