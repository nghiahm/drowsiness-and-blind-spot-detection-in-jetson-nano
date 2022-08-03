import os
import shutil
import random
import argparse

parser = argparse.ArgumentParser(
    description='Splitting data for drowsiness detection')

parser.add_argument('--d', '--dataset-dir', default='data/dataset/',
    				help='Directory of dataset')
parser.add_argument('--i', '--images-dir', default='data/images/',
    				help='Directory for saving images')
parser.add_argument('--n', '--number-images', default=240, type=int,
                    help='Number of image per state')
parser.add_argument('--nt', '--number-train-images', default=180, type=int,
                    help='Number of train image per state')

args = parser.parse_args()

def main():
	DATASET_PATH = args.d
	IMAGES_PATH = args.i
	TRAIN_PATH = IMAGES_PATH + "train/"
	VAL_PATH = IMAGES_PATH + "val/"

	if not os.path.exists(TRAIN_PATH) or not os.path.exists(VAL_PATH):
		os.mkdir(TRAIN_PATH) 
		os.mkdir(VAL_PATH)
	else:
		pass

	close_eye_list = []
	open_eye_list = []

	close_eye_images = os.listdir(DATASET_PATH + "close_eye")
	if not os.path.exists(TRAIN_PATH + "close_eye") or not os.path.exists(VAL_PATH + "close_eye"):
		os.mkdir(TRAIN_PATH + "close_eye")
		os.mkdir(VAL_PATH + "close_eye")
	else:
		pass
	
	open_eye_images = os.listdir(DATASET_PATH + "open_eye")
	if not os.path.exists(TRAIN_PATH + "open_eye") or not os.path.exists(VAL_PATH + "open_eye"):
		os.mkdir(TRAIN_PATH + "open_eye")
		os.mkdir(VAL_PATH + "open_eye")
	else:
		pass

	total_images = args.n
	train_images = args.nt
	# val_images = total_images - train_images

	random_number = random.sample(range(0, total_images), total_images) # a list of random number

	for image in close_eye_images: # split train, val images for close_eye
		close_eye_list.append(image)
	count = 0
	while count < total_images:
		if count < train_images:
			shutil.copy(DATASET_PATH + "close_eye/" + close_eye_list[random_number[count]], TRAIN_PATH + "close_eye")
		else:
			shutil.copy(DATASET_PATH + "close_eye/" + close_eye_list[random_number[count]], VAL_PATH + "close_eye")
		count += 1


	for image in open_eye_images: # split train, val images for open_eye
		open_eye_list.append(image)
	count = 0
	while count < total_images:
		if count < train_images:
			shutil.copy(DATASET_PATH + "open_eye/" + open_eye_list[random_number[count]], TRAIN_PATH + "open_eye")
		else:
			shutil.copy(DATASET_PATH + "open_eye/" + open_eye_list[random_number[count]], VAL_PATH + "open_eye")
		count += 1

	print('Splitting complete.....................')

if __name__ == "__main__":
	main()