import matplotlib
from PIL import Image # noqa
from alive_progress import alive_bar # noqa
import numpy as np # noqa
from matplotlib import pyplot
import os

matplotlib.use('agg')


def load_image(filename):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    return pixels


def resize_landscape(image, required_size=(256, 256)):
    # If image is high enough, we can split it into 2 images
    if image.shape[0] >= image.shape[1] * 2:
        middle = int(image.shape[0] / 2)
        image1 = image[:middle, :, :]
        image2 = image[middle:, :, :]
        # Resize images
        image1 = Image.fromarray(image1)
        image2 = Image.fromarray(image2)
        image1 = image1.resize(required_size)
        image2 = image2.resize(required_size)

        return np.asarray(image1), np.asarray(image2)

    # If image is wide enough, we can split it into 2 images
    if image.shape[1] >= image.shape[0] * 2:
        middle = int(image.shape[1] / 2)
        image1 = image[:, :middle, :]
        image2 = image[:, middle:, :]
        # Resize images
        image1 = Image.fromarray(image1)
        image2 = Image.fromarray(image2)
        image1 = image1.resize(required_size)
        image2 = image2.resize(required_size)

        return np.asarray(image1), np.asarray(image2)

    image = Image.fromarray(image)
    image = image.resize(required_size)
    return np.asarray(image), None


def load_images(directory, required_size=(256, 256)):
    files = os.listdir(directory)
    landscapes = list()
    with alive_bar(len(files), title='Processing images') as bar:
        for _, filename in enumerate(files):
            # Load the image
            image = load_image(directory + filename)
            # Resize
            resized = resize_landscape(image, required_size)
            # Store
            if resized[1] is not None:
                landscapes.append(resized[1])
            landscapes.append(resized[0])

            bar()

    return np.asarray(landscapes)


def save_samples(n_samples, images, iteration, log_folder):
    # Convert images and normalize
    images = images.numpy()
    images = np.transpose(images, axes=[0, 2, 3, 1])
    images = (images + 1) / 2
    images = np.clip(images, -1, 1)

    # Save images
    _, axs = pyplot.subplots(
        n_samples,
        n_samples,
        figsize=(images.shape[1], images.shape[1])
    )

    axs = axs.flatten()
    for img, ax in zip(images, axs):
        ax.axis('off')
        ax.imshow((img * 255).astype(np.uint8))

    # save plot to file
    filename = '%s/generated_img_iter_%s03d.png' % (log_folder, str(iteration + 1))
    pyplot.savefig(filename)
    pyplot.close()
