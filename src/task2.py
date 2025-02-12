import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import cv2
import os

def get_each_image(folder):
	for file in os.listdir(folder):
		image_path = os.path.join(folder, file)
		try:
			img = Image.open(image_path)
			#print(f"Image: {file}")
			gray_img = magnitude_method(img)
			mag_delta_f = get_magnitude_gradient_f(gray_img)
			normalize_this = fit_to_better_visualize(mag_delta_f)
			show_image(gray_img, normalize_this)
			img.close()
		except Exception as e:
			print(f"Error opening {file}: {e}")


def magnitude_method(image):
	img = np.array(image, 'float64')
	magnitude_img = np.sort(img[:,:,0]**2 + img[:,:,1]**2 + img[:,:,2]**2)
	grayscale_img = (magnitude_img / np.max(magnitude_img)) * 255
	return grayscale_img

def get_magnitude_gradient_f(gray_img):
	hori_gradient = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, 3)
	verti_gradient = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, 3)
	magni_gradient = np.sqrt(hori_gradient**2 + verti_gradient**2)
	return magni_gradient

def fit_to_better_visualize(final_img):
	finally_image = ((final_img / np.max(final_img)) * 255).astype('uint8')
	return finally_image
 
def show_image(img, gray_img):
	fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
	ax[0].imshow(np.array(img), cmap = 'gray', vmin=0, vmax=255)
	ax[0].axis('off')
	ax[1].imshow(np.array(gray_img), cmap = 'gray', vmin=0, vmax=255)
	ax[1].axis('off')
	plt.show()

if __name__ == "__main__":
	folder_path = "/Users/vikrantdhekial/dronevision-onboarding/src/sky_classification_export/images"
	img = get_each_image(folder_path)



