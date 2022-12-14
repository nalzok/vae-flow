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

from typing import Any, Iterator, Mapping, Sequence, Tuple

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from absl import app, flags, logging

from .vae import VAE, VAEOutput
from .flow import make_flow_model


flags.DEFINE_float("beta", 1, "Hyperparameter beta as in beta-VAE")
flags.DEFINE_integer("latent_size", 256, "Dimension of VAE latent vector")
flags.DEFINE_integer("vae_hidden_size", 512, "Hidden size of the VAE")

flags.DEFINE_integer("flow_num_layers", 8, "Number of layers to use in the flow.")
flags.DEFINE_integer(
    "mlp_num_layers", 2, "Number of layers to use in the MLP conditioner."
)
flags.DEFINE_integer("mlp_hidden_size", 512, "Hidden size of the MLP conditioner.")
flags.DEFINE_integer(
    "num_bins", 4, "Number of bins to use in the rational-quadratic spline."
)

flags.DEFINE_integer("batch_size", 128, "Size of the batch to train on")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the vae_optimizer")
flags.DEFINE_integer("training_steps", 5001, "Number of training steps to run")
flags.DEFINE_integer("eval_frequency", 500, "How often to evaluate the model")
FLAGS = flags.FLAGS


OptState = Any
PRNGKey = jnp.ndarray
Batch = Mapping[str, np.ndarray]

MNIST_IMAGE_SHAPE: Sequence[int] = (28, 28, 1)


def load_dataset(split: str, batch_size: int) -> Iterator[Batch]:
    ds = tfds.load("binarized_mnist", split=split, shuffle_files=True)
    ds = ds.shuffle(buffer_size=10 * batch_size)
    ds = ds.batch(batch_size)
    ds = ds.cache()
    ds = ds.prefetch(buffer_size=5)
    ds = ds.repeat()
    return iter(tfds.as_numpy(ds))


def vae_fn():
    model = VAE(FLAGS.latent_size, FLAGS.vae_hidden_size)

    def init(x, z):
        forward(x)
        sample(z)

    def forward(x):
        return model(x)

    def sample(z):
        return model.sample(z)

    return init, (forward, sample)


def flow_fn():
    event_shape = (FLAGS.latent_size,)

    bijector = make_flow_model(
        event_shape=event_shape,
        num_layers=FLAGS.flow_num_layers,
        hidden_sizes=[FLAGS.mlp_hidden_size] * FLAGS.mlp_num_layers,
        num_bins=FLAGS.num_bins
    )
    base_distribution = distrax.MultivariateNormalDiag(
        loc=jnp.zeros(event_shape),
        scale_diag=jnp.ones(event_shape)
    )
    transformed = distrax.Transformed(base_distribution, bijector)

    def init(z, epsilon):
        log_prob(z)
        transform(epsilon)

    def log_prob(z: jnp.ndarray) -> jnp.ndarray:
        return transformed.log_prob(z)

    def transform(epsilon: jnp.ndarray) -> jnp.ndarray:
        return distrax.Inverse(bijector).forward(epsilon)

    return init, (log_prob, transform)


def main(_):
    vae = hk.multi_transform(vae_fn)
    flow =  hk.multi_transform(flow_fn)

    forward, sample = vae.apply
    log_prob, transform = flow.apply
    rng_seq = hk.PRNGSequence(42)

    vae_params = vae.init(
        next(rng_seq),
        jnp.empty((1, *MNIST_IMAGE_SHAPE)),
        jnp.empty((1, FLAGS.latent_size)),
    )
    flow_params = flow.init(
        next(rng_seq),
        jnp.empty((1, FLAGS.latent_size)),
        jnp.empty((1, FLAGS.latent_size)),
    )
    params = hk.data_structures.merge(vae_params, flow_params, check_duplicates=True)

    optimizer = optax.adam(FLAGS.learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params: hk.Params, rng_key: PRNGKey, batch: Batch) -> jnp.ndarray:
        """Loss = -ELBO, where ELBO = E_q[log p(x|z)] - beta * KL(q(z|x) || p(z))."""

        outputs: VAEOutput = forward(params, rng_key, batch["image"])
        log_likelihood = outputs.likelihood_distrib.log_prob(batch["image"])

        llk_p = outputs.variational_distrib.log_prob(outputs.z)
        llk_q = log_prob(params, None, outputs.z)

        elbo = log_likelihood - FLAGS.beta * (llk_p - llk_q)

        return -jnp.mean(elbo)

    @jax.jit
    def update(
        params: hk.Params,
        rng_key: PRNGKey,
        opt_state: OptState,
        batch: Batch,
    ) -> Tuple[hk.Params, OptState]:
        """Single SGD update step."""
        grads = jax.grad(loss_fn)(params, rng_key, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    @jax.jit
    def generate(params: hk.Params, rng_key_epsilon: PRNGKey, rng_key_x: PRNGKey) -> jnp.ndarray:
        """Generate sample."""
        prior_epsilon = distrax.MultivariateNormalDiag(
            loc=jnp.zeros((FLAGS.latent_size,)),
            scale_diag=jnp.ones((FLAGS.latent_size,)),
        )
        epsilon = prior_epsilon.sample(seed=rng_key_epsilon)
        z = transform(params, None, epsilon)
        image = sample(params, rng_key_x, z)
        return image

    train_ds = load_dataset(tfds.Split.TRAIN, FLAGS.batch_size)
    valid_ds = load_dataset(tfds.Split.TEST, FLAGS.batch_size)

    for step in range(FLAGS.training_steps):
        params, opt_state = update(params, next(rng_seq), opt_state, next(train_ds))

        if step % FLAGS.eval_frequency == 0:
            val_loss = loss_fn(params, next(rng_seq), next(valid_ds))
            logging.info("STEP: %5d; Validation ELBO: %.3f", step, -val_loss)


    nrows, ncols = 4, 4
    fig, axes = plt.subplots(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            image = generate(params, next(rng_seq), next(rng_seq))
            image = image.reshape(MNIST_IMAGE_SHAPE[:-1])
            axes[i, j].imshow(image, cmap="gray")
            axes[i, j].axis("off")
            plt.tight_layout()

    fig.savefig("figures/generated.png")


if __name__ == "__main__":
    app.run(main)
