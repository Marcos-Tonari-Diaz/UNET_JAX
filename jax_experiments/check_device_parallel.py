from jax.lib import xla_bridge
import jax
print(xla_bridge.get_backend().platform)
print(jax.default_backend())
print("device count" + str(jax.device_count()))
print("devices" + str(jax.devices()))