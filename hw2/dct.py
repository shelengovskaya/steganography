from scipy.fftpack import dct, idct
from skimage import img_as_ubyte, color
from PIL import Image
import numpy as np
from numpy import r_
import scipy
import math
import sys

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

def get_estimation(original, modified, watermark):
	get_psnr(original, modified)
	# print(type(original))
	# print(type(modified))
	ssim = get_ssim(original[:,:,2], modified[:,:,2])
	print('SSIM: ', ssim)
	get_ec(watermark, original)

def get_ec(watermark, original):
	original_size = original.shape
	watermark_size = watermark.shape

	ec = (watermark_size[0]*watermark_size[1]*8) / (original_size[0]*original_size[1])
	print('EC:', ec)


# add NCC


# def get_ssim(original, modified):
# 	ssim = get_ssim(original, modified)
# 	print("SSIM: {}".format(ssim))

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

from math import floor, ceil

def get_Z(dc_coef, l):
	first = (np.max(dc_coef)+2*l) / (2*l)
	second = (np.min(dc_coef)-2*l) / (2*l)
	Z = floor(first)-floor(second)
	return Z


def get_QA_QB(dc_coef, Z, l):
	QA = np.array([])
	QB = np.array([])

	for k in range(1, Z+1):
		qa_val = np.min(dc_coef) + (2*k-4)*l
		QA = np.append(QA, qa_val)
		qb_val = np.min(dc_coef) + (2*k-5)*l
		QB = np.append(QB, qb_val)
	return (QA, QB)


def get_M(image, message):
	dc_coef = get_dc_coef(image)
	# print(dc_coef[:10])
	Z = get_Z(dc_coef, 20)
	new_dc_coef = np.array([0.]*len(dc_coef))
	index = 0
	for i in range(len(dc_coef)):
		if index >= len(message):
			new_dc_coef[i] = dc_coef[i]
			continue
		QA, QB = get_QA_QB(dc_coef, Z, 20)
		if message[index] == '0':
			min_val = 100000000
			min_k = 0
			for k in range(Z):
				if abs(dc_coef[i]-QA[k]) < min_val:
					min_val = abs(dc_coef[i]-QA[k])
					min_k = k
			new_dc_coef[i] = QA[min_k]
		elif message[index] == '1':
			min_val = 100000000
			min_k = 0
			for k in range(Z):
				if abs(dc_coef[i]-QB[k]) < min_val:
					min_val = abs(dc_coef[i]-QB[k])
					min_k = k
			new_dc_coef[i] = QB[min_k]
		index += 1

	M = new_dc_coef - dc_coef
	return M



def get_dc_coef(image):

	dc_coef = np.array([])

	index = 0
	imsize = image.shape

	# print(imsize)
	# print(image)
	
	for i in r_[:imsize[0]:8]:
	    for j in r_[:imsize[1]:8]:

	    	# есть кусок [i:(i+8),j:(j+8)]
	    	# обрабатываем его
	    	s = 0

	    	for l in range(i, i+8):
	    		for k in range(j, j+8):
	    			if l >= imsize[0]:
	    				continue
	    			if k >= imsize[1]:
	    				continue
	    			s += image[l,k,2]

	    	dc = s/sqrt(8*8)
	    	dc_coef = np.append(dc_coef, dc)

	return dc_coef


def embed(im, M):
	image = im.copy()
	index = 0
	imsize = image.shape

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
	    			if index >= len(M):
	    				continue
	    			image[l,k,2] += 1/8*M[index]

	    	index += 1

	return image


def extract(dc_coef, length):
	result = ''

	for i in range(min(length, len(dc_coef))):
		W = ceil(dc_coef[i]/20) % 2
		result += str(W)

	return result



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

	
	M = get_M(image, message)
	result_image = embed(image, M)


	Image.fromarray(result_image.astype('uint8')).save(output_file)
	print('Image with watermark:', output_file)


	new_dc_coef = get_dc_coef(result_image)


	result_message = extract(new_dc_coef, len(message))

	# print(message[:20])
	# print(result_message[:20])

	result_message = bytes2image(result_message)
	result_message = np.reshape(np.array(result_message), watermark.shape)

	Image.fromarray(result_message.astype('uint8')).save(extracted_watermark_file)
	print('Extracted watermark:', extracted_watermark_file)

	get_estimation(image, result_image, watermark)


if __name__=='__main__':
	main()


# def main_embed():
# 	test_number = sys.argv[1]
# 	input_file = IMAGE_DIR+'image{}.jpeg'.format(test_number)
# 	output_file = IMAGE_WITH_WATERMARK_DIR+'image{}_with_watermark{}.jpg'.format(test_number, test_number)
# 	watermark_file = WATERMARK_DIR+'watermark{}.jpg'.format(test_number)
# 	# extracted_watermark_file = EXTRACTED_WATERMARK+'result_watermark{}.jpg'.format(test_number)

# 	watermark = np.array(Image.open(watermark_file).convert('L'))
# 	message = image2bytes(watermark)
# 	print('Watermark:', watermark_file)

# 	image = np.array(Image.open(input_file))
# 	print('Input image:', input_file)
	
# 	result_image = embed(image, get_M(image, message))

# 	Image.fromarray(result_image.astype('uint8')).save(output_file)
# 	print('Image with watermark:', output_file)

# 	get_estimation(image, result_image, watermark)


# def main_extract():
# 	test_number = sys.argv[1]
# 	# input_file = IMAGE_DIR+'image{}.jpeg'.format(test_number)
# 	output_file = IMAGE_WITH_WATERMARK_DIR+'image{}_with_watermark{}.jpg'.format(test_number, test_number)
# 	watermark_file = WATERMARK_DIR+'watermark{}.jpg'.format(test_number)
# 	extracted_watermark_file = EXTRACTED_WATERMARK+'result_watermark{}.jpg'.format(test_number)

# 	watermark = np.array(Image.open(watermark_file).convert('L'))

# 	result_image = np.array(Image.open(output_file))

# 	dc_coef = get_dc_coef(result_image)

# 	result_message = extract(dc_coef, watermark.shape[0]*watermark.shape[1]*8)

# 	result_message = bytes2image(result_message)

# 	result_message = np.reshape(np.array(result_message), watermark.shape)

# 	Image.fromarray(result_message.astype('uint8')).save(extracted_watermark_file)
# 	print('Extracted watermark:', extracted_watermark_file)

# 	# get_estimation(image, result_image, watermark)



# if __name__=='__main__':
# 	if sys.argv[2] == 'embed':
# 		main_embed()
# 	elif sys.argv[2] == 'extract':
# 		main_extract()