from typing import List, Tuple

from model import UnetJAX
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

from flax.training import train_state
from data_loader import UnetTrainDataGenerator

from jax import tree_util


def make_batch(image, mask):
    batch = {"image": image.reshape((1,)+image.shape),
             "mask": mask.reshape((1,)+mask.shape)}
    return batch


def loss_function(logits, labels):
    return optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    # return optax.softmax_cross_entropy(logits=logits, labels=labels).mean()


def logits_to_binary(logits):
    logits = nn.sigmoid(logits)
    logits = logits.round()
    return logits


def compute_accuracy(logits, masks):
    return jnp.mean(logits_to_binary(logits) == masks)


def compute_IOU(logits, masks):
    predictions = logits_to_binary(logits)
    intersection = jnp.logical_and(masks, predictions)
    union = jnp.logical_or(masks, predictions)
    return jnp.sum(intersection) / jnp.sum(union)


def compute_average_metrics(metrics_list):
    loss_list = [metrics["loss"] for metrics in metrics_list]
    accuracy_list = [metrics["accuracy"] for metrics in metrics_list]
    iou_list = [metrics["iou"] for metrics in metrics_list]
    loss_arr = jnp.array(loss_list)
    accuracy_arr = jnp.array(accuracy_list)
    iou_arr = jnp.array(iou_list)
    average_metrics = {
        'loss': jnp.mean(loss_arr),
        'accuracy': jnp.mean(accuracy_arr),
        'iou': jnp.mean(iou_arr)
    }
    return average_metrics


def print_metrics(metrics, epoch, description: str):
    print(description + ': epoch: %d, loss: %.8f, accuracy: %.4f, iou: %.8f' %
          (epoch, metrics["loss"], metrics["accuracy"], metrics["iou"]))


class UnetTrainState():
    train_state: train_state.TrainState
    unet: UnetJAX
    current_epoch: int = 0
    rng: jax.random.PRNGKey
    steps_per_epoch: int = 30

    def __init__(self, unet: UnetJAX, optimizer, seed, steps_per_epoch=30):
        self.unet = unet
        self.rng = jax.random.PRNGKey(seed)
        self.create_training_state(optimizer)
        self.steps_per_epoch = steps_per_epoch

    def _tree_flatten(self):
        children = (self.train_state, self.unet, self.current_epoch)
        aux_data = {"steps_per_epoch": self.steps_per_epoch, "rng": self.rng}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    @staticmethod
    def registerAsPyTree():
        tree_util.register_pytree_node(
            UnetTrainState, UnetTrainState._tree_flatten, UnetTrainState._tree_unflatten)

    def create_training_state(self, optimizer):
        unet_params = self.unet.init_params(self.rng)
        self.train_state = train_state.TrainState.create(
            apply_fn=self.unet.apply, params=unet_params, tx=optimizer)

    # @jax.jit
    def train_step(self, batch):
        def compute_loss_function(params):
            logits = self.unet.apply(
                {'params': params}, batch['image'])
            loss = loss_function(logits, batch['mask'])
            return loss, logits
        compute_loss_grads = jax.value_and_grad(
            compute_loss_function, has_aux=True)

        (loss, logits), grads = compute_loss_grads(self.train_state.params)

        self.train_state = self.train_state.apply_gradients(grads=grads)
        batch_accuracy = compute_accuracy(logits, batch['mask'])
        batch_iou = compute_accuracy(logits, batch['mask'])
        return {"loss": loss, "accuracy": batch_accuracy, "iou": batch_iou}

    # @jax.jit
    def eval_step(self, batch):
        logits = self.unet.apply(
            {'params': self.train_state.params}, batch['image'])
        loss = loss_function(logits, batch['mask'])
        batch_accuracy = compute_accuracy(logits, batch['mask'])
        batch_iou = compute_accuracy(logits, batch['mask'])
        return {"loss": loss, "accuracy": batch_accuracy, "iou": batch_iou}

    # batch size is 1 image (paper)
    def train_epoch(self, data_generator: UnetTrainDataGenerator):
        train_metrics: List[Tuple] = []
        for _ in range(self.steps_per_epoch):
            batch = data_generator.get_batch_jax()
            batch_metrics = self.train_step(batch)
            print_metrics(batch_metrics, self.current_epoch, "train step")
            # print('trainable param: '+str(self.train_state.params['contracting_block_1']
            #                               ['conv_1']['kernel'][0, 0, 0, 0]))
            train_metrics.append(batch_metrics)
        self.current_epoch += 1
        average_metrics = compute_average_metrics(train_metrics)
        print_metrics(average_metrics, self.current_epoch, "train epoch")
        return average_metrics

    def eval_model(self, test_dataset):
        test_metrics = []
        for image, mask in zip(test_dataset["images"], test_dataset["masks"]):
            batch = make_batch(image, mask)
            batch_metrics = self.eval_step(batch)
            batch_metrics = jax.device_get(batch_metrics)
            test_metrics.append(batch_metrics)
        average_metrics = compute_average_metrics(test_metrics)
        print_metrics(average_metrics, self.current_epoch, "eval")
        return average_metrics
