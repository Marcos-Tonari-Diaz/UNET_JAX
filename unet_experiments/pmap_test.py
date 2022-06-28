import jax
import jax.numpy as jnp
import functools

from flax import linen as nn
from flax import jax_utils
import optax
from flax.training.train_state import TrainState

print(jax.device_count())

model = nn.Dense(1)
x = jnp.ones((jax.device_count(), 3))
params = model.init(jax.random.PRNGKey(0), x)
tx = optax.adam(learning_rate=1e-3)
state = TrainState.create(
    apply_fn=model.apply, params=params, tx=tx,
)
state = jax_utils.replicate(state)

# print(state)
print("x:", x)


@functools.partial(jax.pmap, axis_name='batch')
def loss_fn(state, x):
    logits = (model.apply(state.params, x) ** 2.0)
    return jax.lax.pmean(logits, axis_name='batch')


print(loss_fn(state, x))
