#coding:latin_1

from Functions import *
from PIL import Image
from scipy.misc import imsave

#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- 1. Initial settings. -------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

resize_shape = (256, 256)  # Resized image's shape
sigma = 10                 # Noise standard dev.

window_shape = (16, 16)    # Patches' shape
step = 16                  # Patches' step
ratio = 1             # Ratio for the dictionary (training set).
ksvd_iter = 5              # Number of iterations for the K-SVD.

#-------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------- 2. Image import. ----------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

name = 'Lenna'
original_image = np.asarray(Image.open(name+'.png').convert('L').resize(resize_shape))
learning_image = np.asarray(Image.open('Lenna.png').convert('L').resize(resize_shape))

#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------- 3. Image processing. --------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

noise_layer = np.random.normal(0, sigma ^ 2, original_image.size).reshape(original_image.shape).astype(int)
noisy_image = original_image + noise_layer

imsave(name + '1 - Greysc image.jpg', Image.fromarray(np.uint8(original_image)))
imsave(name + '2 - Noisy image.jpg', Image.fromarray(np.uint8(noisy_image)))

#-------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------- 4. Denoising. -----------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

denoised_image, calc_time, n_total = denoising(noisy_image, learning_image, window_shape, step, sigma, ratio, ksvd_iter)

psnr = psnr(original_image, denoised_image)
print 'PSNR             : ' + str(psnr) + ' dB.'

imsave(name + '3 - Out - Step ' + str(step) + ' - kSVD ' + str(ksvd_iter) +
       ' - Ratio ' + str(ratio) + '.jpg', Image.fromarray(np.uint8(denoised_image)))
imsave(name + '4 - Difference - Step ' + str(step) + ' - kSVD ' + str(ksvd_iter) +
       ' - Ratio ' + str(ratio) + '.jpg', Image.fromarray(np.uint8(np.abs(noisy_image - denoised_image))))

txt = open(name + ' Parameters.txt', 'a')
txt.write('INITIAL SETTINGS\n----------------' +
              '\n\nResizing shape : ' + str(resize_shape) +
              '\nWindow shape   : ' + str(window_shape) +
              '\nWindow step    : ' + str(step) +
              '\nResizing shape : ' + str(resize_shape) +
              '\nSigma          : ' + str(sigma) +
              '\nLearning ratio : ' + str(ratio) +
              '\nK-SVD iter.    : ' + str(ksvd_iter) +
              '\n\nComputation time : ' + str(calc_time) + ' seconds.' +
              '\nPSNR           : ' + str(psnr) + ' dB.\n')
txt.close()






