import functools
from statistics import mean
from typing import List, Tuple, Callable, Dict

import jax
from matplotlib.pyplot import axis
import optax
import flax

import jax.numpy as jnp
from flax.training import train_state
from sklearn.model_selection import GridSearchCV

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


def compute_average_batched_metrics(batched_metrics: Dict):
    return {metric: array.mean() for metric, array in batched_metrics.items()}


def print_metrics(metrics, step, description: str):
    print(
        f'{description} {step}, loss: {metrics["loss"]}, accuracy: {metrics["accuracy"]}, iou: {metrics["iou"]}')


def compute_loss_function(params, batch, apply_function):
    logits = apply_function(
        {'params': params}, batch['image'])
    loss = loss_function(logits, batch['mask'])
    return loss, logits


def get_loss_grad():
    return jax.value_and_grad(compute_loss_function, argnums=[0], has_aux=True)


def get_vmap_loss_grad():
    return jax.vmap(get_loss_grad(), in_axes=(None, 0, None))


def get_vmap_compute_metrics():
    return jax.vmap(compute_metrics)


def grads_mean(batched_params):
    return jax.tree_map(
        lambda leaf: leaf.mean(axis=0),
        batched_params)[0]


def print_params_leaves(batched_params):
    return jax.tree_map(
        lambda leaf: print(f'leaf: {leaf.shape} \n {leaf}'), batched_params)


def train_step(train_state, batch):
    (loss_arr, logits_arr), grads = train_state.compute_loss_grad(
        train_state.params, batch, train_state.apply_fn)
    grads = grads_mean(grads)
    new_state = train_state.apply_gradients(grads=grads)
    batched_metrics = train_state.vmap_compute_metrics(
        logits_arr, batch['mask'])
    batched_metrics["loss"] = loss_arr
    mini_batch_metrics = compute_average_batched_metrics(batched_metrics)
    return new_state, mini_batch_metrics


@functools.partial(jax.vmap, in_axes=(None, 0))
def eval_step(train_state, batch):
    logits = train_state.apply_fn(
        {'params': train_state.params}, batch['image'])
    loss = loss_function(logits, batch['mask'])
    batch_metrics = compute_metrics(logits, batch['mask'])
    return {"loss": loss, "accuracy": batch_metrics["accuracy"], "iou": batch_metrics["iou"]}


def train_epoch(state, data_generator: UnetTrainDataGenerator):
    train_metrics: List[Tuple] = []
    for step, mini_batch in enumerate(data_generator.generator()):
        state, mini_batch_metrics = train_step(state, mini_batch)
        mini_batch_metrics = jax.device_get(mini_batch_metrics)
        train_metrics.append(mini_batch_metrics)
        print_metrics(mini_batch_metrics, step, "train step")
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
    vmap_compute_metrics: Callable = flax.struct.field(pytree_node=False)
