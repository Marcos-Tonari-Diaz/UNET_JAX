from random import sample
import jax.numpy as jnp
import numpy as np
from sklearn import datasets
from unet_utils import get_augmented, plot_imgs
from PIL import Image
import glob
from sklearn.model_selection import train_test_split


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


def add_dummy_batch_dimension(image, mask):
    return image.reshape((1,)+image.shape), mask.reshape((1,)+mask.shape)


def add_dummy_batch_dimension(array):
    return array.reshape((1,)+array.shape)


def make_sample(image, mask):
    image, mask = add_dummy_batch_dimension(image, mask)
    sample = {"image": image, "mask": mask}
    return sample


def prepare_dataset(paths, train_split_size):
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


class UnetTrainDataGenerator:
    def __init__(self, images, masks, batch_size=4, seed=0):
        self.batch_size = batch_size
        self.keras_generator = get_augmented(
            images,
            masks,
            seed=seed,
            batch_size=batch_size,
            data_gen_args=dict(
                rotation_range=15.,
                width_shift_range=0.05,
                height_shift_range=0.05,
                shear_range=50,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='constant'
            ))

    def get_batch(self):
        images_batch, masks_batch = next(self.keras_generator)
        images_batch = images_batch.reshape(
            images_batch.shape[0], 1, images_batch.shape[1], images_batch.shape[2], images_batch.shape[3])
        masks_batch = masks_batch.reshape(
            masks_batch.shape[0], 1, masks_batch.shape[1], masks_batch.shape[2], masks_batch.shape[3])
        batch = {"image": images_batch,
                 "mask": masks_batch}
        # batch = {"image": [add_dummy_batch_dimension(image) for image in images_batch],
        #          "mask": [add_dummy_batch_dimension(mask) for mask in masks_batch]}
        return batch


def get_mini_batch_size_ajusted_samples(test_dataset, mini_batch_size):
    samples = [make_sample(image, mask) for image, mask in zip(
        test_dataset["images"], test_dataset["masks"])]
    samples_gap_number = len(samples) % mini_batch_size

    if samples_gap_number != 0:
        for index_offset in range(mini_batch_size - samples_gap_number):
            samples.append(samples[index_offset])

    assert len(samples) % mini_batch_size == 0
    return samples


def test_data_batch_generator(test_dataset, mini_batch_size):
    samples = get_mini_batch_size_ajusted_samples(
        test_dataset, mini_batch_size)
    batches = np.array(samples).reshape((mini_batch_size, -1))
    for batch in batches:
        yield batch


if __name__ == "__main__":
    paths = {"images": "../data/isbi2015/train/image/*.png",
             "masks": "../data/isbi2015/train/label/*.png"}
    dataset = prepare_dataset(paths, train_split_size=0.5)

    unet_datagen = UnetTrainDataGenerator(
        dataset["train"]["images"], dataset["train"]["masks"], seed=1, batch_size=4)
    batch = unet_datagen.get_batch()
    # # print(batch.shape)
    # print(batch["image"].shape)

    # unet_datagen = UnetTrainDataGenerator(
    #     dataset["train"]["images"], dataset["train"]["masks"], seed=1, batch_size=4)
    # batch = unet_datagen.get_batch()
    # # print(batch.shape)
    # print(batch[0]["image"].shape)

    # for i in test_data_batch_generator(dataset["test"], 4):
    #     print(i.shape)
    #     print(i[0]["image"].shape)
    #     break
