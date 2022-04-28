from scipy.fftpack import dct, idct
from skimage import img_as_ubyte, color
from PIL import Image
import numpy as np
from numpy import r_
import scipy
import math
import sys

sign = lambda x: math.copysign(1, x)


def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')


def get_dct(im):
	imsize = im.shape
	dct = np.zeros(imsize)
	# Do 8x8 DCT on image (in-place)
	for i in r_[:imsize[0]:8]:
	    for j in r_[:imsize[1]:8]:
	        dct[i:(i+8),j:(j+8)] = dct2( im[i:(i+8),j:(j+8)])
	return dct.astype('int')


def get_idct(im):
	imsize = im.shape
	dct = np.zeros(imsize)
	# Do 8x8 iDCT on image (in-place)
	for i in r_[:imsize[0]:8]:
	    for j in r_[:imsize[1]:8]:
	        dct[i:(i+8),j:(j+8)] = idct2( im[i:(i+8),j:(j+8)])
	return dct


def f3_embed_dct(dct, message):
	index = 0
	imsize = dct.shape
	new_dct = np.zeros(imsize)
	# Do 8x8 iDCT on image (in-place)
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
	    				new_dct[l, k] = dct[l, k]
	    				continue

	    			if index < len(message) and ((abs(dct[l,k]) % 2 == 1 and message[index] == '0') or (abs(dct[l,k]) % 2 == 0 and message[index] == '1')):
	    				new_dct[l, k] = sign(dct[l, k]) * (abs(dct[l, k])-1)
	    			else:
	    				new_dct[l, k] = dct[l, k]

	    			if new_dct[l,k] != 0:
	    				index += 1

	return new_dct.astype('int')


def f3_extract_dct(dct, length):
	index = 0
	message = ''
	imsize = dct.shape
	new_dct = np.zeros(imsize)
	# Do 8x8 iDCT on image (in-place)
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
	    			if index >= length:
	    				return message

	    			# -----------------------
	    			if dct[l,k] == 0:
	    				continue
	    			if abs(dct[l, k]) % 2 == 0:
	    				message += '0'
	    			else:
	    				message += '1'
	    			# -----------------------

	    			index += 1

	return message


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
	ssim = get_ssim(original, modified)
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


def main():
	test_number = sys.argv[1]
	input_file = IMAGE_DIR+'image{}.jpg'.format(test_number)
	output_file = IMAGE_WITH_WATERMARK_DIR+'image{}_with_watermark{}.jpg'.format(test_number, test_number)
	watermark_file = WATERMARK_DIR+'watermark{}.jpg'.format(test_number)
	extracted_watermark_file = EXTRACTED_WATERMARK+'result_watermark{}.jpg'.format(test_number)

	watermark = np.array(Image.open(watermark_file).convert('L'))
	message = image2bytes(watermark)
	print('Watermark:', watermark_file)

	image = np.array(Image.open(input_file).convert('L'))
	print('Input image:', input_file)


	dct_image = f3_embed_dct(get_dct(image), message)
	result_image = get_idct(dct_image)
	Image.fromarray(result_image.astype('uint8')).save(output_file)
	print('Image with watermark:', output_file)


	result_message = f3_extract_dct(dct_image, len(message))

	result_message = bytes2image(result_message)
	result_message = np.reshape(np.array(result_message), watermark.shape)

	Image.fromarray(result_message.astype('uint8')).save(extracted_watermark_file)
	print('Extracted watermark:', extracted_watermark_file)

	get_estimation(image, result_image, watermark)



# def main3_image_embed():
# 	input_file = 'image1.jpg'
# 	output_file = 'image1_with_watermark1.jpg'
# 	watermark_file = 'watermark.jpg'

# 	watermark = np.array(Image.open(WATERMARK_DIR+watermark_file).convert('L'))
# 	message = image2bytes(watermark)

# 	image = np.array(Image.open(IMAGE_DIR+input_file).convert('L'))
# 	dct_image = f3_embed_dct(get_dct(image), message)
# 	result_image = get_idct(dct_image)
# 	Image.fromarray(result_image.astype('uint8')).save(IMAGE_WITH_WATERMARK_DIR+output_file)


# def main3_image_extract():

# 	message = '0'*32768

# 	input_file = 'output.jpg'

# 	image = np.array(Image.open(input_file).convert('L'))

# 	result_message = f3_extract_dct(image, len(message))

# 	# print(len(message), len(result_message))

# 	result_message = bytes2image(result_message)
# 	result_message = np.reshape(np.array(result_message), (64, 64))

# 	Image.fromarray(result_message.astype('uint8')).save('result_watermark.png')



if __name__=='__main__':
	main()