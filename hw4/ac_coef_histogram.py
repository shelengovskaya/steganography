import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from skimage import img_as_ubyte, color
from PIL import Image
import numpy as np
from numpy import r_
import scipy
import math
import sys
import matplotlib.pyplot as plt
from collections import Counter


def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )


def get_dct(im):
	imsize = im.shape
	dct = np.zeros(imsize)
	# Do 8x8 DCT on image (in-place)
	for i in r_[:imsize[0]:8]:
	    for j in r_[:imsize[1]:8]:
	        dct[i:(i+8),j:(j+8)] = dct2( im[i:(i+8),j:(j+8)])
	return dct.astype('int')

def get_ac_coef_info(dct):
	data = []

	imsize = dct.shape

	for i in r_[:imsize[0]:8]:
	    for j in r_[:imsize[1]:8]:

	    	# есть кусок [i:(i+8),j:(j+8)]
	    	# обрабатываем его

	    	for l in range(i, i+8):
	    		for k in range(j, j+8):
	    			if l >= imsize[0]:
	    				continue
	    			if k >= imsize[1]:
	    				continue
	    			if l == i and k == j:
	    				continue

	    			data.append(dct[l,k])
	return data

IMAGE_DIR = 'input_image/'
HISTOGRAM_DIR = 'histogram/'


def show(data, output_file):
	plt.rcParams["figure.figsize"] = [7.00, 10]
	plt.rcParams["figure.autolayout"] = True
	freqs = Counter(data)

	plt.ylim(ymax = min(max(freqs.values()), 2000))
	plt.xlim(xmax = max(freqs.keys()), xmin = min(freqs.keys()))

	plt.bar(freqs.keys(), freqs.values())
	plt.savefig(output_file)
	plt.show()


def main():
	test_number = sys.argv[1]
	input_file = IMAGE_DIR+'image{}.jpg'.format(test_number)
	output_file = HISTOGRAM_DIR+'histogram_image{}.png'.format(test_number)

	image = np.array(Image.open(input_file))
	dct_image = get_dct(image)

	data = get_ac_coef_info(dct_image)

	show(data, output_file)



if __name__=='__main__':
	main()