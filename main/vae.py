# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Variational Autoencoder example on binarized MNIST dataset."""

from typing import NamedTuple, Sequence, Tuple

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from .common import MNIST_IMAGE_SHAPE


class Encoder(hk.Module):
    """Encoder model."""

    def __init__(self, hidden_size: int, latent_size: int):
        super().__init__()
        self._hidden_size = hidden_size
        self._latent_size = latent_size

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = hk.Flatten()(x)
        x = hk.Linear(self._hidden_size)(x)
        x = jax.nn.relu(x)

        mean = hk.Linear(self._latent_size)(x)
        log_stddev = hk.Linear(self._latent_size)(x)
        stddev = jnp.exp(log_stddev)

        return mean, stddev


class Decoder(hk.Module):
    """Decoder model."""

    def __init__(
        self,
        hidden_size: int,
        output_shape: Sequence[int] = MNIST_IMAGE_SHAPE,
    ):
        super().__init__()
        self._hidden_size = hidden_size
        self._output_shape = output_shape

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        z = hk.Linear(self._hidden_size)(z)
        z = jax.nn.relu(z)

        logits = hk.Linear(np.prod(self._output_shape))(z)
        logits = jnp.reshape(logits, (-1, *self._output_shape))

        return logits


class VAEOutput(NamedTuple):
    variational_distrib: distrax.Distribution
    likelihood_distrib: distrax.Distribution
    z: jnp.ndarray


class VAE(hk.Module):
    """Main VAE model class, uses Encoder & Decoder under the hood."""

    def __init__(
        self,
        latent_size: int,
        hidden_size: int,
        output_shape: Sequence[int] = MNIST_IMAGE_SHAPE,
    ):
        super().__init__()
        self._latent_size = latent_size
        self._hidden_size = hidden_size
        self._output_shape = output_shape

    def __call__(self, x: jnp.ndarray) -> VAEOutput:
        x = x.astype(jnp.float32)

        # q(z|x) = N(mean(x), covariance(x))
        mean, stddev = Encoder(self._hidden_size, self._latent_size)(x)
        variational_distrib = distrax.MultivariateNormalDiag(
            loc=mean, scale_diag=stddev
        )
        z = variational_distrib.sample(seed=hk.next_rng_key())

        # p(x|z) = \Prod Bernoulli(logits(z))
        logits = Decoder(self._hidden_size, self._output_shape)(z)
        likelihood_distrib = distrax.Independent(
            distrax.Bernoulli(logits=logits),
            reinterpreted_batch_ndims=len(self._output_shape),
        )  # 3 non-batch dims

        return VAEOutput(variational_distrib, likelihood_distrib, z)

    def sample(self, z: jnp.ndarray) -> jnp.ndarray:
        # p(x|z) = \Prod Bernoulli(logits(z))
        logits = Decoder(self._hidden_size, self._output_shape)(z)
        likelihood_distrib = distrax.Independent(
            distrax.Bernoulli(logits=logits),
            reinterpreted_batch_ndims=len(self._output_shape),
        )  # 3 non-batch dims

        # Generate images from the likelihood
        image = likelihood_distrib.sample(seed=hk.next_rng_key())

        return image
