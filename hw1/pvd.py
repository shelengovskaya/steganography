from scipy.fftpack import dct, idct
from skimage import img_as_ubyte, color
from PIL import Image
import numpy as np
from numpy import r_
import scipy
import math
import sys
from math import floor, ceil, log2


sign = lambda x: math.copysign(1, x)



# -------- estimation ---------------

from math import log10, sqrt
from skimage.metrics import structural_similarity as get_ssim
import warnings
warnings.filterwarnings("ignore")

def get_mse(original, modified):
	mse = np.mean((original - modified) ** 2)
	print('MSE: ', mse)
	return mse

def get_rmse(original, modified):
	rmse = sqrt(get_mse(original, modified))
	print('RMSE:', rmse)
	return rmse

def get_psnr(original, modified):
    rmse = get_rmse(original, modified)
    if (rmse == 0):
    	return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / rmse)
    print('PSNR:', psnr)
    return psnr

def get_ec(watermark, original):
	original_size = original.shape
	watermark_size = watermark.shape

	ec = (watermark_size[0]*watermark_size[1]*8) / (original_size[0]*original_size[1])
	print('EC:', ec)

# def norm_data(data):
#     mean_data=np.mean(data)
#     std_data=np.std(data, ddof=1)
#     return (data-mean_data)/(std_data)

# def get_ncc(data0, data1):
#     """
#     normalized cross-correlation coefficient between two data sets
#     https://xcdskd.readthedocs.io/en/latest/cross_correlation/cross_correlation_coefficient.html
#     """
#     ncc = (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))
#     print('NCC:', ncc)
#     return ncc

def get_estimation(original, modified, watermark):
	get_psnr(original, modified)
	# print(type(original))
	# print(type(modified))
	ssim = get_ssim(original[:,:,2], modified[:,:,2])
	print('SSIM: ', ssim)
	get_ec(watermark, original)
	# get_ncc(original, modified)


# --------------------------------------



def string2bytes(st):

	result = ''
	for x in st:
		ord_symbol = format(ord(x), 'b')

		while len(ord_symbol) != 8:
			ord_symbol = '0' + ord_symbol

		result += ord_symbol

	return result


def string_of_bites_to_symbol(char):
	result = chr(string_of_bites_to_integer(char))
	return result

def string_of_bites_to_integer(char):
	e = 1
	ord_number = 0
	for i in range(len(char) - 1, -1, -1):
	    if char[i] == '1':
	        ord_number += e
	    e *= 2
	return ord_number


# for text

def bytes2string(msg):
	result = ''
	for i in range(0, len(msg), 8):
		byte = msg[i:(i+8)]
		result += string_of_bites_to_symbol(byte)
	return result



def image2bytes(image):
    result = ''
    for row in image:
        for x in row:
            s = str(format(x, 'b'))
            while len(s) < 8:
                s = '0' + s
            result += s
    return result

# for image

def bytes2image(msg):
    result = []
    for i in range(0, len(msg), 8):
        byte = msg[i:(i+8)]
        result.append(string_of_bites_to_integer(byte))
    return result


IMAGE_DIR = 'input_image/'
IMAGE_WITH_WATERMARK_DIR = 'image_with_watermark/'
WATERMARK_DIR = 'watermark/'
EXTRACTED_WATERMARK = 'extracted_watermark/'


def get_w(a, b):
	m = abs(a - b)
	if 0 <= m <= 7:
		return 8
	if 8 <= m <= 15:
		return 8
	if 16 <= m <= 31:
		return 16
	if 32 <= m <= 63:
		return 32
	if 64 <= m <= 127:
		return 64
	return 128


def get_pair(image, i, j):
	image_size = image.shape
	n = image_size[0]
	m = image_size[1]
	return (n-i-1, m-j-1)


def pvd_embed(im, message):
	index = 0
	image = im.copy()
	for i1 in range(image.shape[0]):
		for j1 in range(image.shape[1]-i1):
			i2, j2 = get_pair(image, i1, j1)
			# print(i1, j1, i2, j2)
			w = get_w(image[i1,j1,2], image[i2,j2,2])
			n = np.floor(log2(w))
			d = image[i1,j1,2] - image[i2,j2,2]
			b = ''
			while n and index < len(message):
				b += message[index]
				n -= 1
				index += 1
			b = string_of_bites_to_integer(b)

			if d >= 0:
				new_d = w + b
			else:
				new_d = -(w + b)

			# проверка из статьи: https://www.hindawi.com/journals/jam/2013/189706/

			if image[i1,j1,2] >= image[i2,j2,2] and new_d > d:
				image[i1,j1,2] = image[i1,j1,2] + ceil((new_d-d)/2)
				image[i2,j2,2] = image[i2,j2,2] - floor((new_d-d)/2)
			elif image[i1,j1,2] < image[i2,j2,2] and new_d > d:
				image[i1,j1,2] = image[i1,j1,2] - ceil((new_d-d)/2)
				image[i2,j2,2] = image[i2,j2,2] + floor((new_d-d)/2)
			elif image[i1,j1,2] >= image[i2,j2,2] and new_d <= d:
				image[i1,j1,2] = image[i1,j1,2] - ceil((new_d-d)/2)
				image[i2,j2,2] = image[i2,j2,2] + floor((new_d-d)/2)
			elif image[i1,j1,2] < image[i2,j2,2] and new_d <= d:
				image[i1,j1,2] = image[i1,j1,2] + ceil((new_d-d)/2)
				image[i2,j2,2] = image[i2,j2,2] - floor((new_d-d)/2)

	return image

def pvd_extract(image, length):
	result = ''
	for i1 in range(image.shape[0]):
		for j1 in range(image.shape[1]-i1):
			i2, j2 = get_pair(image, i1, j1)
			w = get_w(image[i1,j1,2], image[i2,j2,2])
			# n = log2(w)
			d = image[i1,j1,2] - image[i2,j2,2]
			b = abs(d) - w

			bits = format(b, 'b')
			result += bits[:min(length, len(bits))]
			length -= len(bits)
			if length <= 0:
				return result
	return result

from matplotlib import pyplot as plt
import cv2 as cv

def get_hist(file):
	girl = cv.imread(file)

	plt.hist(girl.ravel(), 256, [0, 256])
	plt.savefig('histogram/'+file)
	plt.close() 



def main():
	test_number = sys.argv[1]
	input_file = IMAGE_DIR+'image{}.jpeg'.format(test_number)
	output_file = IMAGE_WITH_WATERMARK_DIR+'image{}_with_watermark{}.jpg'.format(test_number, test_number)
	watermark_file = WATERMARK_DIR+'watermark{}.jpg'.format(test_number)
	extracted_watermark_file = EXTRACTED_WATERMARK+'result_watermark{}.jpg'.format(test_number)

	watermark = np.array(Image.open(watermark_file).convert('L'))
	message = image2bytes(watermark)
	print('Watermark:', watermark_file)

	image = np.array(Image.open(input_file))
	print('Input image:', input_file)

	result_image = pvd_embed(image, message)

	Image.fromarray(result_image.astype('uint8')).save(output_file)
	print('Image with watermark:', output_file)


	result_message = pvd_extract(result_image, len(message))

	result_message = bytes2image(result_message)
	result_message = np.reshape(np.array(result_message), watermark.shape)

	Image.fromarray(result_message.astype('uint8')).save(extracted_watermark_file)
	print('Extracted watermark:', extracted_watermark_file)

	get_estimation(image, result_image, watermark)

	get_hist(input_file)
	get_hist(output_file)


if __name__=='__main__':
	main()