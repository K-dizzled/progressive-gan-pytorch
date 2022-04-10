from image_processing import *
import numpy as np # noqa


class ImagePool:
    def __init__(self, data_dir, resolution, batch_size):
        self.current_batch = 0
        self.resolution = resolution
        self.path = data_dir
        self.batch_size = batch_size

        # Get numpy array of images
        X = load_images(data_dir, (resolution, resolution))

        # Normalize data
        X = X.astype('float32')  # convert from ints to floats
        X = (X - 127.5) / 127.5  # scale from [0,255] to [-1,1]

        # TODO: Add random horizontal flip and maybe random crop
        # TODO: Also add random shuffle of images
        self.max_image_num = X.shape[0]
        self.images = X
        self.batches = np.array_split(
            self.images,
            self.max_image_num // self.batch_size,
            axis=0
        )

    def next(self):
        # Split array into small batches
        self.current_batch += 1

        if self.current_batch >= len(self.batches):
            self.current_batch = 0

        return self.batches[self.current_batch]
