from model import UnetJAX
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

from flax.training import train_state
from data_loader import prepare_dataset, UnetTrainDataGenerator


def loss_function(logits, masks):
    return optax.sigmoid_binary_cross_entropy(logits, masks).mean()


def logits_to_binary(logits):
    logits = nn.sigmoid(logits)
    logits = logits.round()
    return logits


def compute_accuracy(logits, masks):
    return jnp.mean(logits_to_binary(logits) == masks)


def compute_metrics(logits, masks):
    loss = loss_function(logits, masks)
    accuracy = compute_accuracy(logits, masks)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics


def compute_average_metrics(metrics_list):
    loss_list = [metrics["loss"] for metrics in metrics_list]
    accuracy_list = [metrics["accuracy"] for metrics in metrics_list]
    average_metrics = {
        'loss': jnp.mean(loss_list),
        'accuracy': jnp.mean(accuracy_list)
    }
    return average_metrics


class UnetTrainState():
    train_state: train_state.TrainState
    unet: UnetJAX
    current_epoch: int = 0
    rng: jax.random.PRNGKey
    unet_params = None
    dataset_size: int = 30

    def __init__(self, unet: UnetJAX, optimizer, seed):
        self.unet = unet
        self.rng = jax.random.PRNGKey(seed)
        self.create_training_state(optimizer)

    def create_training_state(self, optimizer):
        self.unet_params = unet.init_params(self.rng)
        self.train_state = train_state.TrainState.create(
            apply_fn=unet.apply, params=self.unet_params, tx=optimizer)

    def print_train_metrics(self, metrics):
        print('train epoch: %d, loss: %.4f, accuracy: %.4f' %
              (self.current_epoch, metrics["loss"], metrics["accuracy"]))

    def print_eval_metrics(self, metrics):
        print('model eval: %d, loss: %.4f, accuracy: %.4f' %
              (self.current_epoch, metrics["loss"], metrics["accuracy"]))

    def train_step(self, batch):
        def compute_loss_function(params):
            logits = unet.apply(self.unet_params, batch['image'])
            loss = loss_function(logits, batch['mask'])
            return loss
        compute_loss_grads = jax.grad(compute_loss_function)
        grads = compute_loss_grads(self.train_state.params)
        self.train_state = self.train_state.apply_gradients(grads=grads)

    def eval_step(self, batch):
        logits = unet.apply(self.train_state.params, batch['image'])
        return compute_metrics(logits, batch['mask'])

    # batch size is 1 image (paper)
    def train_epoch(self, data_generator: UnetTrainDataGenerator):
        for _ in range(self.dataset_size):
            batch = data_generator.get_batch_jax()
            self.train_step(batch)
            batch_metrics = self.eval_step(batch)
            batch_metrics = jax.device_get(batch_metrics)
            self.print_train_metrics(batch_metrics)
            self.current_epoch += 1

    def eval_model(self, test_dataset):
        test_metrics = []
        for image_index in len(test_dataset["images"]):
            batch = {"image": test_dataset["images"][image_index],
                     "mask": test_dataset["masks"][image_index]}
            metrics = self.eval_step(batch)
            metrics = jax.device_get(metrics)
            test_metrics.append(metrics)
        average_metrics = compute_average_metrics(test_metrics)
        self.print_eval_metrics(average_metrics)


if __name__ == "__main__":
    # dataset
    paths = {"images": "../data/isbi2015/train/image/*.png",
             "masks": "../data/isbi2015/train/label/*.png"}
    dataset = prepare_dataset(paths, train_split_size=0.5)
    unet_datagen = UnetTrainDataGenerator(
        dataset["train"]["images"], dataset["train"]["masks"], seed=1)
    # batch = unet_datagen.get_batch_jax()

    unet = UnetJAX(input_image_size=512,
                   use_activation=False, use_padding=False)
    optimizer = optax.sgd(learning_rate=0.1, momentum=0.99)
    unet_train_state = UnetTrainState(unet, optimizer, seed=0)
    unet_train_state.train_epoch(data_generator=unet_datagen)
    # unet_train_state.eval_model(test_dataset=dataset["test"])
