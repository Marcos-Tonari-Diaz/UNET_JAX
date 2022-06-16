import time
import functools
import numpy as np
from jax import random, jit, device_put, vmap, pmap, profiler
import jax.numpy as jnp


def perf_timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"({func.__name__}) Elapsed time: {elapsed_time:2.10f} seconds")
        return value
    return wrapper_timer


def numpy_random_sum(array):
    return np.sum(array)


def jax_random_sum(array):
    rand_sum = jnp.sum(array).block_until_ready()
    return rand_sum


def jax_random_sum_device_put(array):
    array = device_put(array)
    rand_sum = jnp.sum(array).block_until_ready()
    return rand_sum


def jax_random_sum_jit_factory():
    def jax_random_sum(array):
        rand_sum = jnp.sum(array)
        return rand_sum
    return jit(jax_random_sum)


def jax_random_sum_batched(array):
    rand_sum = jnp.sum(array, axis=0)
    return rand_sum


def jax_random_sum_vmap_factory():
    return vmap(jax_random_sum_batched)


def jax_random_sum_pmap_factory():
    return pmap(jax_random_sum_batched)


if __name__ == "__main__":
    profiler.start_trace("tensorboard_logs")

    dummy_key = random.PRNGKey(0)

    arrary_size = 1e8
    rand_vec_numpy = np.random.normal(size=int(arrary_size))
    perf_timer(numpy_random_sum)(rand_vec_numpy)

    prng_key = random.PRNGKey(1)
    rand_vec_jax = random.normal(prng_key, (1, int(arrary_size)))
    perf_timer(jax_random_sum)(rand_vec_jax)

    prng_key = random.PRNGKey(2)
    rand_vec_jax = random.normal(prng_key, (1, int(arrary_size)))
    perf_timer(jax_random_sum_device_put)(rand_vec_jax)

    print("jitted:")
    prng_key = random.PRNGKey(3)
    rand_vec_jax = random.normal(prng_key, (1, int(arrary_size)))
    dummy = random.normal(dummy_key, (1, int(arrary_size)))
    jited_random_sum = jax_random_sum_jit_factory()
    jited_random_sum(dummy)  # discard first run
    perf_timer(jited_random_sum)(rand_vec_jax)

    prng_key = random.PRNGKey(4)
    rand_vec_jax = random.normal(prng_key, (1, int(arrary_size)))
    batched_arr = jnp.reshape(rand_vec_jax, (4, -1))
    print(perf_timer(jax_random_sum_batched)(batched_arr))

    print("vmap:")
    prng_key = random.PRNGKey(5)
    rand_vec_jax = random.normal(prng_key, (1, int(arrary_size)))
    dummy = random.normal(dummy_key, (1, int(arrary_size)))
    vmap_random_sum = jax_random_sum_vmap_factory()
    vmap_random_sum(dummy)  # discard first run
    print(perf_timer(vmap_random_sum)(rand_vec_jax))

    print("pmap:")
    prng_key = random.PRNGKey(5)
    rand_vec_jax = random.normal(prng_key, (1, int(arrary_size)))
    dummy = random.normal(dummy_key, (1, int(arrary_size)))
    pmap_random_sum = jax_random_sum_pmap_factory()
    pmap_random_sum(dummy)  # discard first run
    print(perf_timer(pmap_random_sum)(rand_vec_jax))

    profiler.stop_trace()
