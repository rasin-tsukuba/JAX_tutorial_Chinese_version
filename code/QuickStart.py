import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

# generate random number
key = random.PRNGKey(0)
x = random.normal(key, (10, ))

print(x)

# multiply two big matrics
size = 3000
x = random.normal(key, (size, size), dtype=jnp.float32)
print(x)