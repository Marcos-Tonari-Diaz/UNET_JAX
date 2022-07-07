import time
import functools
import numpy as np
from jax import random, device_put
import jax.numpy as jnp
from jax.lax import psum
import jax
import functools


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


def perf_timer_jax(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs).block_until_ready()
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


def jax_random_sum_device_put(array):
    array = device_put(array)
    rand_sum = jnp.sum(array)
    return rand_sum


@jax.jit
def jax_random_sum_jit(array):
    rand_sum = jnp.sum(array)
    return rand_sum

@jax.jit
def jax_random_sum_jit_device_put(array):
    array = device_put(array)
    rand_sum = jnp.sum(array)
    return rand_sum

def jax_random_sum_batched(array):
    rand_sum = jnp.sum(array, axis=0)
    rand_sum = jnp.sum(rand_sum)
    return rand_sum


@jax.vmap
def sum_batch(array):
    return jnp.sum(array)


def jax_random_sum_vmap(array):
    rand_sum = sum_batch(array)
    rand_sum = jnp.sum(rand_sum)
    return rand_sum

@jax.jit
def jax_random_sum_vmap_jit(array):
    rand_sum = sum_batch(array)
    rand_sum = jnp.sum(rand_sum)
    return rand_sum

@functools.partial(jax.pmap, axis_name="batch")
def jax_random_sum_pmap(array):
    rand_sum = jnp.sum(array, axis=0)
    rand_sum = psum(rand_sum, axis_name="batch")
    return rand_sum

@functools.partial(jax.pmap, axis_name="batch")
@jax.jit
def jax_random_sum_pmap_jit(array):
    rand_sum = jnp.sum(array, axis=0)
    rand_sum = psum(rand_sum, axis_name="batch")
    return rand_sum


if __name__ == "__main__":

    dummy_key = random.PRNGKey(0)

    arrary_size = 1e8
    rand_vec_numpy = np.random.normal(size=int(arrary_size))
    prng_key = random.PRNGKey(1)
    rand_vec_jax = random.normal(prng_key, (1, int(arrary_size)))
    batched_arr = jnp.reshape(rand_vec_jax, (4, -1))
    dummy = random.normal(dummy_key, (1, int(arrary_size)))

    print(f'sum: {perf_timer(numpy_random_sum)(rand_vec_numpy)}')

    print(f'sum: {perf_timer_jax(jax_random_sum)(rand_vec_jax)}')

    print(f'sum: {perf_timer_jax(jax_random_sum_device_put)(rand_vec_jax)}')

    jax_random_sum_jit(dummy)  # discard first run
    print(f'sum: {perf_timer_jax(jax_random_sum_jit)(rand_vec_jax)}')

    jax_random_sum_jit_device_put(dummy)  # discard first run
    print(f'sum: {perf_timer_jax(jax_random_sum_jit_device_put)(rand_vec_jax)}')

    print(f'sum: {perf_timer_jax(jax_random_sum_batched)(batched_arr)}')

    jax_random_sum_vmap(dummy)  # discard first run
    print(f'sum: {perf_timer_jax(jax_random_sum_vmap)(batched_arr)}')

    jax_random_sum_vmap_jit(dummy)  # discard first run
    print(f'sum: {perf_timer_jax(jax_random_sum_vmap_jit)(batched_arr)}')

    jax_random_sum_pmap(dummy)  # discard first run
    print(f'sum: {perf_timer_jax(jax_random_sum_pmap)(batched_arr)}')

    jax_random_sum_pmap_jit(dummy)  # discard first run
    print(f'sum: {perf_timer_jax(jax_random_sum_pmap_jit)(batched_arr)}')
