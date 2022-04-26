import time
import functools
import numpy as np
from jax import random, jit
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
    rand_sum = jnp.sum(array)
    return rand_sum


def jax_random_sum_manual_batched(array):
    rand_sum = jnp.sum(array)
    return rand_sum


def jax_random_sum_jit_factory():
    return jit(jax_random_sum)


if __name__ == "__main__":
    rand_vec_numpy = np.random.normal(size=int(5e7))
    perf_timer(numpy_random_sum)(rand_vec_numpy)
    prng_key = random.PRNGKey(1)
    rand_vec_jax = random.normal(prng_key, (1, int(5e7)))
    perf_timer(jax_random_sum)(rand_vec_jax)
    prng_key = random.PRNGKey(2)
    rand_vec_jax = random.normal(prng_key, (1, int(5e7)))
    perf_timer(jax_random_sum_jit_factory())(rand_vec_jax)
    prng_key = random.PRNGKey(3)
    rand_vec_jax = random.normal(prng_key, (1, int(5e7)))
    batched_arr = jnp.reshape(rand_vec_jax, (4, -1))
    perf_timer(jax_random_sum_manual_batched)(batched_arr)
