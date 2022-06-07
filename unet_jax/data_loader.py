import jax
import numpy as np
from sklearn import datasets
from keras_unet_utils import get_augmented, plot_imgs
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
    def __init__(self, images, masks, seed):
        self.keras_generator = get_augmented(
            images,
            masks,
            seed=seed,
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

    def get_batch_jax(self):
        image, mask = next(self.keras_generator)
        image = jax.device_put(image)
        mask = jax.device_put(mask)
        return {"image": image, "mask": mask}

    def get_batch_numpy(self):
        image, mask = next(self.keras_generator)
        return {"image": image, "mask": mask}


if __name__ == "__main__":
    paths = {"images": "../data/isbi2015/train/image/*.png",
             "masks": "../data/isbi2015/train/label/*.png"}
    dataset = prepare_dataset(paths, train_split_size=0.5)
    unet_datagen = UnetTrainDataGenerator(
        dataset["train"]["images"], dataset["train"]["masks"], seed=1)
    batch = unet_datagen.get_batch_jax()
    plot_imgs(batch["image"], batch["mask"], nm_img_to_plot=1)
