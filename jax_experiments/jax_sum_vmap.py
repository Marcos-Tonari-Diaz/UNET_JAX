from jax import vmap, random
from jax_sum_cpu import jax_random_sum, perf_timer


def jax_random_sum_vmap_factory():
    return vmap(jax_random_sum)


if __name__ == "__main__":
    prng_key = random.PRNGKey(4)
    rand_vec_jax = random.normal(prng_key, (1, int(2e7)))
    perf_timer(jax_random_sum_vmap_factory())(rand_vec_jax)
