import imageio  # noqa
from PIL import Image # noqa
import numpy as np # noqa
import os

images = []
directory = "../run_2022-04-06_22-21/gif/"


for i, filename in enumerate(sorted(os.listdir(directory), key=lambda x: (len(x), x))):
    if i % 2 == 0:
        continue

    image = Image.open(directory + filename)
    image = image.convert('RGB')

    pixels = np.asarray(image)
    pixels = Image.fromarray(pixels)
    image1 = pixels.resize((800, 800))
    images.append(image1)

imageio.mimsave('learning_process.gif', images)
