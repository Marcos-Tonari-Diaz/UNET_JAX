from jax.lib import xla_bridge
import jax
print(xla_bridge.get_backend().platform)
print(jax.default_backend())
