from typing import Tuple
import numpy as np
from PIL import Image
import glob

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


def scale_pixel_values(array):
    array = np.asarray(array, dtype=np.float32)
    array /= 255
    return array


def load_images_as_array(path):
    image_files = glob.glob(path)
    image_list = [np.array(Image.open(image_file).resize((512, 512)))
                  for image_file in image_files]
    image_arr = np.asarray(image_list)
    return image_arr


def add_dummy_batch_dimension(array):
    return array.reshape((array.shape[0], 1) + (array.shape[1:]))


def load_dataset(paths, train_split_size):
    images = load_images_as_array(paths["images"])
    masks = load_images_as_array(paths["masks"])
    images = scale_pixel_values(images)
    masks = scale_pixel_values(masks)
    images = images.reshape(images.shape + (1,))
    masks = masks.reshape(masks.shape + (1,))
    images_train, images_test, masks_train, masks_test = train_test_split(
        images, masks, train_size=train_split_size, random_state=0)
    dataset = {"train": {"images": images_train, "masks": masks_train},
               "test": {"images": images_test, "masks": masks_test}}
    return dataset


def prepare_batch(batch: Tuple):
    images_batch, masks_batch = batch

    images_batch = add_dummy_batch_dimension(images_batch)
    masks_batch = add_dummy_batch_dimension(masks_batch)
    return images_batch, masks_batch


class UnetDataGenerator:
    def __init__(self, batch_size=4):
        self.batch_size = batch_size
        self.padding_batch_generator = None

    def append_batch_padding(self, batch, pad_batch):
        padding_size = self.batch_size - batch.shape[0]
        pad_batch = pad_batch[0:padding_size]
        pad_batch = add_dummy_batch_dimension(pad_batch)
        batch = np.append(batch, pad_batch, axis=0)
        return batch

    def pad_batch(self, batch):
        images_batch, masks_batch = batch
        if images_batch.shape[0] < self.batch_size:
            images_pad_batch, masks_pad_batch = self.get_pad_batch()
            images_batch = self.append_batch_padding(images_batch,
                                                     images_pad_batch)
            masks_batch = self.append_batch_padding(masks_batch,
                                                    masks_pad_batch)

        return {"images": images_batch, "masks": masks_batch}


class UnetTrainDataGenerator(UnetDataGenerator):
    def __init__(self, train_dataset, data_gen_args, batch_size=4, seed=0, steps_per_epoch=100):
        UnetDataGenerator.__init__(self, batch_size=batch_size)
        self.steps_per_epoch = steps_per_epoch

        masks_train_generator = ImageDataGenerator(**data_gen_args)
        masks_train_generator = ImageDataGenerator(**data_gen_args)

        masks_train_generator.fit(
            train_dataset["images"], augment=True, seed=seed)
        masks_train_generator.fit(
            train_dataset["masks"], augment=True, seed=seed)

        augmented_images_generator = masks_train_generator.flow(
            train_dataset["images"], batch_size=batch_size, shuffle=True, seed=seed
        )
        augmented_masks_generator = masks_train_generator.flow(
            train_dataset["masks"], batch_size=batch_size, shuffle=True, seed=seed
        )

        self.keras_generator = zip(
            augmented_images_generator, augmented_masks_generator)

    def generator(self):
        for _ in range(self.steps_per_epoch):
            batch = next(self.keras_generator)
            batch = prepare_batch(batch)
            batch = self.pad_batch(batch)
            images_batch = batch["images"]
            masks_batch = batch["masks"]
            yield {"image": images_batch, "mask": masks_batch}

    def get_pad_batch(self):
        return next(self.keras_generator)


class UnetTestDataGenerator(UnetDataGenerator):
    def __init__(self, test_dataset, batch_size=4):
        UnetDataGenerator.__init__(self, batch_size=batch_size)
        self.test_data_generator = self.get_test_data_generator(
            test_dataset)
        self.padding_batch_generator = self.get_test_data_generator(
            test_dataset)

    def get_test_data_generator(self, test_dataset):
        images = (tf.data.Dataset.from_tensor_slices(
            test_dataset["images"]).batch(self.batch_size))
        masks = (tf.data.Dataset.from_tensor_slices(
            test_dataset["masks"]).batch(self.batch_size))
        return zip(images, masks)

    def generator(self):
        for batch in self.test_data_generator:
            images_batch, masks_batch = batch
            batch = prepare_batch(
                (images_batch.numpy(), masks_batch.numpy()))
            batch = self.pad_batch(batch)
            images_batch = batch["images"]
            masks_batch = batch["masks"]
            yield {"image": images_batch, "mask": masks_batch}

    def get_pad_batch(self):
        images_batch, masks_batch = next(self.padding_batch_generator)
        return images_batch.numpy(), masks_batch.numpy()


if __name__ == "__main__":
    paths = {"images": "../data/isbi2015/train/image/*.png",
             "masks": "../data/isbi2015/train/label/*.png"}
    dataset = load_dataset(paths, train_split_size=0.5)
    data_gen_args = dict(
        rotation_range=15.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=50,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant'
    )

    train_datagen = UnetTrainDataGenerator(
        dataset['train'], data_gen_args, seed=1, batch_size=4, steps_per_epoch=10)

    test_datagen = UnetTestDataGenerator(dataset['test'], batch_size=4)

    print("train")
    for step, batch in enumerate(train_datagen.generator()):
        print(f'step {step} : {batch["image"].shape}')

    print("test")
    for step, batch in enumerate(test_datagen.generator()):
        print(f'step {step} : {batch["image"].shape}')
