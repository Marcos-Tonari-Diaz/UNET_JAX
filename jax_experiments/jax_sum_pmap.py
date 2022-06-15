from jax import pmap, random
from jax_sum_cpu import jax_random_sum, perf_timer


def jax_random_sum_pmap_factory():
    return pmap(jax_random_sum)


if __name__ == "__main__":
    prng_key = random.PRNGKey(5)
    rand_vec_jax = random.normal(prng_key, (1, int(2e7)))
    perf_timer(jax_random_sum_pmap_factory())(rand_vec_jax)
