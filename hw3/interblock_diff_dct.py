from scipy.fftpack import dct, idct
from skimage import img_as_ubyte, color
from PIL import Image
import numpy as np
from numpy import r_
import scipy
import math
import sys
from math import floor, ceil

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
	ssim = get_ssim(original[:,:,2], modified[:,:,2])
	print('SSIM: ', ssim)
	get_ec(watermark, original)

def get_ec(watermark, original):
	original_size = original.shape
	watermark_size = watermark.shape

	ec = (watermark_size[0]*watermark_size[1]*8) / (original_size[0]*original_size[1])
	print('EC:', ec)

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


# ------------- Arnold ---------------

def Arnold(image):
	n, m = image.shape

	result_image = np.zeros((n, m))
	for i in range(n):
		for j in range(m):
			new_i, new_j = np.dot([[1, 1], [1, 2]], [i, j]) % n
			result_image[new_i, new_j] = image[i, j]
	return result_image

def revArnold(image):
	n, m = image.shape

	result_image = np.zeros((n, m))
	for i in range(n):
		for j in range(m):
			new_i, new_j = np.dot([[2, -1], [-1, 1]], [i, j]) % n
			result_image[new_i, new_j] = image[i, j]
	return result_image


# ------------- DCT ------------------

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
	return dct.astype('int')


# -------------- embed ----------------

# внедряем в 2 слой RGB

Z = 2
Th = 80
K = 12

def embed(im, message):

	# Step 1: от каждого пикселя отнимаем 128

	im = im.astype(int)
	im -= 128

	# Step 2: получение DCT коэффициентов

	image_dct = get_dct(im)

	# Step 3: get Med for blocks

	imsize = image_dct.shape

	M = np.array([[0.]*len(r_[:imsize[1]:8])] * len(r_[:imsize[0]:8]))

	for i in r_[:imsize[0]:8][:-1]:
	    for j in r_[:imsize[1]:8][:-1]:

	    	# есть кусок [i:(i+8),j:(j+8)]
	    	# обрабатываем его

	    	Med = np.array([])

	    	for l in range(i, i+4):
	    		for k in range(j, j+4):
	    			if 0 < l-i+k-j <= 3:
	    				Med = np.append(Med, image_dct[l,k,2])

	    	if abs(image_dct[i,j,2]) > 1000 or abs(image_dct[i,j,2]) < 1:
	    		M[int(i/8),int(j/8)] = abs(Z*np.median(Med))
	    	else:
	    		M[int(i/8),int(j/8)] = abs(Z*(image_dct[i,j,2]-np.median(Med))/image_dct[i,j,2])
	

	# Step 4: вычисляем Diff LR, встраиваем message в среднечастотные коэффициенты

	index = 0

	for i in r_[:imsize[0]:8]:
	    for j in r_[:imsize[1]:8]:

	    	if i == r_[:imsize[0]:8][-1] or j == r_[:imsize[1]:8][-1]:
	    		continue

	    	for l in range(i, i+8):
	    		for k in range(j, j+8):

	    			if index >= len(message):
	    				continue
	    			
	    			# проверяем среднечастотность коэффициента

	    			if l-i+k-j <= 3 or (8-(l-i + 1))+(8-(k-j+1)) <= 5:
	    				continue

	    			Diff = image_dct[l,k, 2] - image_dct[l+8,k, 2]

	    			if message[index] == '1':
	    				if Diff > Th - K:
	    					while Diff > Th - K:
		    					image_dct[l,k,2] -= M[int(l/8),int(k/8)]
		    					Diff = image_dct[l,k, 2] - image_dct[l+8,k, 2]
		    			elif Diff < K and Diff > -Th/2:
		    				while Diff < K:
		    					image_dct[l,k,2] += M[int(l/8),int(k/8)]
		    					Diff = image_dct[l,k, 2] - image_dct[l+8,k, 2]
		    			elif Diff < -Th/2:
		    				while Diff > -Th - K:
		    					image_dct[l,k,2] -= M[int(l/8),int(k/8)]
		    					Diff = image_dct[l,k, 2] - image_dct[l+8,k, 2]

		    		else:
		    			if Diff > Th/2:
		    				while Diff <= Th + K:
		    					image_dct[l,k,2] += M[int(l/8),int(k/8)]
		    					Diff = image_dct[l,k, 2] - image_dct[l+8,k, 2]
		    			elif Diff > -K and Diff < Th/2:
		    				while Diff >= -K:
		    					image_dct[l,k,2] -= M[int(l/8),int(k/8)]
		    					Diff = image_dct[l,k, 2] - image_dct[l+8,k, 2]
		    			elif Diff < K - Th:
		    				while Diff <= -Th + K:
		    					image_dct[l,k,2] += M[int(l/8),int(k/8)]
		    					Diff = image_dct[l,k, 2] - image_dct[l+8,k, 2]

	    			index += 1
	  

	# Step 5: обратное DCT

	image = get_idct(image_dct)

	# Step 6: прибавление к каждому значению 128

	image += 128

	return image


# -------------- extract ---------------

def extract(im, length):
	result = ''

	# Step 1: от каждого пикселя отнимаем 128

	im = im.astype(int)
	im -= 128

	# Step 2: получение DCT коэффициентов

	image_dct = get_dct(im)

	# Step 3: вычисляем Diff LR, встраиваем message

	imsize = image_dct.shape

	for i in r_[:imsize[0]:8]:
	    for j in r_[:imsize[1]:8]:

	    	if i == r_[:imsize[0]:8][-1] or j == r_[:imsize[1]:8][-1]:
	    		continue

	    	for l in range(i, i+8):
	    		for k in range(j, j+8):

	    			if length <= len(result):
	    				return result
	    			
	    			# проверяем среднечастотность коэффициента

	    			if l-i+k-j <= 3 or (7-(l-i))+(7-(k-j)) <= 5:
	    				continue

	    			Diff = image_dct[l,k, 2] - image_dct[l+8,k, 2]

	    			if Diff < -Th or (Diff > 0 and Diff < Th):
	    				result += '1'
	    			elif Diff > Th or (Diff > -Th and Diff < 0):
	    				result += '0'

	return result



def main():
	test_number = sys.argv[1]
	input_file = IMAGE_DIR+'image{}.jpeg'.format(test_number)
	output_file = IMAGE_WITH_WATERMARK_DIR+'image{}_with_watermark{}.jpg'.format(test_number, test_number)
	watermark_file = WATERMARK_DIR+'watermark{}.jpg'.format(test_number)
	extracted_watermark_file = EXTRACTED_WATERMARK+'result_watermark{}.jpg'.format(test_number)
	
	print('Watermark:', watermark_file)
	print('Input image:', input_file)
	print('Image with watermark:', output_file)
	print('Extracted watermark:', extracted_watermark_file)


	watermark = np.array(Image.open(watermark_file).convert('L'))
	message = image2bytes(watermark)
	
	image = np.array(Image.open(input_file))
	
	result_image = embed(image, message)

	Image.fromarray(result_image.astype('uint8')).save(output_file)

	result_message = extract(result_image, len(message))

	result_message = bytes2image(result_message)
	result_message = np.reshape(np.array(result_message), watermark.shape)

	Image.fromarray(result_message.astype('uint8')).save(extracted_watermark_file)

	get_estimation(image, result_image, watermark)


if __name__=='__main__':
	main()