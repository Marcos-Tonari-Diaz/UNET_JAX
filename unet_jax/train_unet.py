from typing import Dict

import jax
import jax.numpy as jnp
import optax
import numpy as np

from data_loader import prepare_dataset, UnetTrainDataGenerator
from model import UnetJAX
from unet_training import UnetTrainState, print_metrics, get_loss_grad
from keras_unet_utils import plot_imgs, get_date_string

from torch.utils.tensorboard import SummaryWriter


def plot_predictions(dataset, unet, unet_train_state, epoch):
    pred_logits = []
    pred_masks = []
    for test_img in dataset['test']['images']:
        logits_pred = unet.apply(
            {"params": unet_train_state.unet_train_state.params}, test_img.reshape((1,)+test_img.shape))[0]
        mask_pred = jnp.round(jax.nn.sigmoid(logits_pred))
        pred_logits.append(logits_pred)
        pred_masks.append(mask_pred)
    pred_masks = np.array(pred_masks)
    pred_logits = np.array(pred_logits)
    original_imgs = np.array(dataset['test']['images'])
    original_masks = np.array(dataset['test']['masks'])

    plot_imgs(
        original_imgs,
        original_masks,
        pred_logits=pred_logits,
        pred_masks=pred_masks,
        nm_img_to_plot=1,
        figsize=10,
        save_path=f"experiment-{get_date_string()}_epoch-{epoch}_prediction.png")


def train_unet():
    input_img_size = 512
    learning_rate = 1e-2
    momentum = 0.99
    num_epochs = 1
    steps_per_epoch = 1
    mini_batch_size = 1
    train_split_size = 0.5
    rng_seed = 0
    data_generator_seed = 1

    summary_writer = SummaryWriter("logs/"+get_date_string())

    paths = {"images": "../data/isbi2015/train/image/*.png",
             "masks": "../data/isbi2015/train/label/*.png"}
    dataset = prepare_dataset(paths, train_split_size=train_split_size)
    unet_datagen = UnetTrainDataGenerator(
        dataset["train"]["images"], dataset["train"]["masks"], batch_size=mini_batch_size, seed=data_generator_seed)

    unet = UnetJAX(input_image_size=input_img_size,
                   use_activation=False, use_padding=True)
    rng_key = jax.random.PRNGKey(rng_seed)
    unet_params = unet.init_params(rng_key)

    optimizer = optax.sgd(learning_rate=learning_rate, momentum=momentum)

    unet_train_state = UnetTrainState.create(
        apply_fn=unet.apply,
        params=unet_params,
        tx=optimizer,
        compute_loss_grad=get_loss_grad())

    UnetTrainState.steps_per_epoch = steps_per_epoch

    for epoch in range(num_epochs):
        unet_train_state, train_metrics = UnetTrainState.train_epoch(unet_train_state,
                                                                     data_generator=unet_datagen)
        print_metrics(train_metrics, epoch, "train epoch: ")

        test_metrics = unet_train_state.eval_model(unet_train_state,
                                                   test_dataset=dataset["test"])
        print_metrics(test_metrics, epoch, "test epoch: ")

        summary_writer.add_scalars(f'loss', {"train": float(np.array(train_metrics["loss"])),
                                             "test": float(np.array(test_metrics["loss"]))}, epoch)
        summary_writer.add_scalars(f'accuracy', {"train": float(np.array(train_metrics["accuracy"])),
                                                 "test": float(np.array(test_metrics["accuracy"]))}, epoch)
        # plot_predictions(dataset, unet, unet_train_state, epoch)

    summary_writer.close()


if __name__ == "__main__":
    train_unet()
