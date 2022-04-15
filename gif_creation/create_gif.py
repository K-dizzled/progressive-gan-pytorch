import imageio  # noqa
from PIL import Image # noqa
import numpy as np # noqa
import os
from alive_progress import alive_bar  # noqa

images = []
directory = "../gif/"

Image.MAX_IMAGE_PIXELS = 100000000000

with alive_bar(len(os.listdir(directory)), title='Processing images') as bar:
	for i, filename in enumerate(sorted(os.listdir(directory), key=lambda x: (len(x), x))):
		if i % 4 == 0:
			continue

		image = Image.open(directory + filename)
		image = image.convert('RGB')

		pixels = np.asarray(image)
		pixels = Image.fromarray(pixels)
		image1 = pixels.resize((800, 800))
		images.append(image1)

		bar()

imageio.mimsave('learning_process.gif', images)
