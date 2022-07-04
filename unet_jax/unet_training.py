import functools
from typing import List, Tuple, Callable, Dict

import jax
import optax
import flax

import jax.numpy as jnp
from flax.training import train_state

from data_loader import UnetTestDataGenerator, UnetTrainDataGenerator


def loss_function(logits, labels):
    return optax.sigmoid_binary_cross_entropy(logits, labels).mean()


def logits_to_binary(logits):
    logits = flax.linen.sigmoid(logits)
    logits = logits.round()
    return logits


def compute_accuracy(logits, masks):
    return jnp.mean(logits_to_binary(logits) == masks)


def compute_IOU(logits, masks):
    predictions = logits_to_binary(logits)
    intersection = jnp.logical_and(masks, predictions)
    union = jnp.logical_or(masks, predictions)
    return jnp.sum(intersection) / jnp.sum(union)


def compute_metrics(logits, masks):
    return {"accuracy": compute_accuracy(logits, masks), "iou": compute_IOU(logits, masks)}


def compute_average_metrics(metrics_list):
    loss_arr = jnp.array([metrics["loss"] for metrics in metrics_list])
    accuracy_arr = jnp.array([metrics["accuracy"] for metrics in metrics_list])
    iou_arr = jnp.array([metrics["iou"] for metrics in metrics_list])
    average_metrics = {
        'loss': jnp.mean(loss_arr),
        'accuracy': jnp.mean(accuracy_arr),
        'iou': jnp.mean(iou_arr)
    }
    return average_metrics


def print_metrics(metrics, step, description: str):
    print(
        f'{description} {step}, loss: {metrics["loss"]}, accuracy: {metrics["accuracy"]}, iou: {metrics["iou"]}')


def compute_loss_function(params, image, mask, apply_function):
    logits = apply_function(
        {'params': params}, image)
    loss = loss_function(logits, mask)
    return loss, logits


def get_loss_grad():
    return jax.value_and_grad(compute_loss_function, argnums=[0], has_aux=True)


def params_mean(param_list):
    return jax.tree_map(
        lambda *leaves: jnp.asarray(sum(leaves)/len(leaves)
                                    ).astype(jnp.asarray(leaves[0]).dtype),
        *param_list)[0]


@jax.jit
def train_step(train_state, mini_batch):
    batch_loss_arr = jnp.array([])
    batch_logits_list = []
    batch_grads_list = []
    batch_accuracy_arr = jnp.array([])
    batch_iou_arr = jnp.array([])
    # for image, mask in zip(batch["image"], batch["mask"]):
    for batch_index in range(len(mini_batch["image"])):
        image = mini_batch["image"][batch_index]
        mask = mini_batch["mask"][batch_index]
        (loss, logits), grads = train_state.compute_loss_grad(
            train_state.params, image, mask, train_state.apply_fn)
        batch_loss_arr = jnp.append(batch_loss_arr, loss)
        batch_logits_list.append(logits)
        batch_grads_list.append(grads)
    batch_logits_arr = jnp.array(batch_logits_list)
    grads = params_mean(batch_grads_list)
    new_state = train_state.apply_gradients(grads=grads)
    for logits_index in range(len(batch_logits_arr)):
        batch_metrics = compute_metrics(
            batch_logits_arr[logits_index], mini_batch["mask"][logits_index])
        batch_accuracy_arr = jnp.append(
            batch_accuracy_arr, batch_metrics["accuracy"])
        batch_iou_arr = jnp.append(batch_iou_arr, batch_metrics["iou"])
    accuracy = jnp.mean(batch_accuracy_arr)
    iou = jnp.mean(batch_iou_arr)
    loss = jnp.mean(batch_loss_arr)
    return new_state, {"loss": loss, "accuracy": accuracy, "iou": iou}


@jax.jit
def eval_step(train_state, mini_batch):
    batch_loss_arr = jnp.array([])
    batch_accuracy_arr = jnp.array([])
    batch_iou_arr = jnp.array([])
    for batch_index in range(len(mini_batch["image"])):
        logits = train_state.apply_fn(
            {'params': train_state.params}, mini_batch["image"][batch_index])
        loss = loss_function(logits, mini_batch["mask"][batch_index])
        batch_loss_arr = jnp.append(batch_loss_arr, loss)
        batch_metrics = compute_metrics(
            logits, mini_batch["mask"][batch_index])
        batch_accuracy_arr = jnp.append(
            batch_accuracy_arr, batch_metrics["accuracy"])
        batch_iou_arr = jnp.append(batch_iou_arr, batch_metrics["iou"])
    accuracy = jnp.mean(batch_accuracy_arr)
    iou = jnp.mean(batch_iou_arr)
    loss = jnp.mean(batch_loss_arr)
    return {"loss": loss, "accuracy": accuracy, "iou": iou}


def train_epoch(state, data_generator: UnetTrainDataGenerator):
    train_metrics: List[Tuple] = []
    for step, batch in enumerate(data_generator.generator()):
        print("inside train loop")
        state, batch_metrics = train_step(state, batch)
        batch_metrics = jax.device_get(batch_metrics)
        train_metrics.append(batch_metrics)
        print_metrics(batch_metrics, step, "train step")
    average_metrics = compute_average_metrics(train_metrics)
    return state, average_metrics


def eval_model(state, data_generator: UnetTestDataGenerator):
    test_metrics: List[Tuple] = []
    for batch in data_generator.generator():
        batch_metrics = eval_step(state, batch)
        batch_metrics = jax.device_get(batch_metrics)
        test_metrics.append(batch_metrics)
    average_metrics = compute_average_metrics(test_metrics)
    return average_metrics


class UnetTrainState(train_state.TrainState):
    compute_loss_grad: Callable = flax.struct.field(pytree_node=False)
