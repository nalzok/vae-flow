from typing import Any, Mapping, Sequence

import numpy as np
import jax.numpy as jnp


Array = jnp.ndarray
Batch = Mapping[str, np.ndarray]
PRNGKey = jnp.ndarray
OptState = Any

MNIST_IMAGE_SHAPE: Sequence[int] = (28, 28, 1)
