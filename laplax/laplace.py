import logging

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())

import numpy as np
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from .utils import seeds_like


class ADLaplace:
    def __init__(self, prior, bijectors, get_likelihood=None):
        self.prior = prior
        self.guide = {key: 0 for key in self.prior}
        self.bijectors = bijectors

        self.get_likelihood = get_likelihood
        self.normal_distributions = jax.tree_map(
            lambda _, bijector, dist: tfd.TransformedDistribution(dist, tfb.Invert(bijector)),
            self.guide,
            self.bijectors,
            self.prior,
        )

    def init(self, seed):
        seeds = seeds_like(seed, self.guide)
        params = jax.tree_map(
            lambda seed, dist: dist.sample(seed=seed),
            seeds,
            self.normal_distributions,
        )
        for param in params.values():
            assert param.ndim <= 1, "params must be scalar or vector"
        return params

    def loss_fn(self, params, data, aux=None):
        transformed_params = jax.tree_map(lambda param, bijector: bijector.forward(param), params, self.bijectors)
        prior_log_probs = jax.tree_map(
            lambda param, dist: dist.log_prob(param),
            params,
            self.normal_distributions,
        )

        def likelihood_log_prob(params, data, aux):
            likelihood = self.get_likelihood(params, aux)
            return likelihood.log_prob(data)

        likelihood_log_probs = jax.vmap(likelihood_log_prob, in_axes=(None, 0, 0))(transformed_params, data, aux)
        loss = -(likelihood_log_probs.sum() + sum(jax.tree_leaves(prior_log_probs)))
        return loss

    def apply(self, params, data, aux=None):
        precision = jax.hessian(self.loss_fn)(params, data, aux)
        return Posterior(params, precision, self.bijectors)


class Posterior:
    def __init__(self, params, precision, bijectors):
        self.params = jax.tree_map(jnp.atleast_1d, params)
        self.guide = jax.tree_structure(self.params)
        self.precision = jax.tree_map(lambda x: x, precision)
        self.bijectors = jax.tree_map(lambda x: x, bijectors)
        
        self.size = jax.tree_map(lambda param: param.size, self.params)
        self.cumsum_size = np.cumsum(jax.tree_leaves(self.size))[:-1]

    def untree_precision(self, keys):
        precision_matrix = []
        for start in keys:
            precision_matrix.append([])
            for end in keys:
                element = self.precision[start][end].reshape((self.size[start], self.size[end]))
                precision_matrix[-1].append(element)
        return jnp.block(precision_matrix)

    def get_normal_dist(self, keys):
        precision_matrix = self.untree_precision(keys)
        covariance_matrix = jnp.linalg.inv(precision_matrix)
        loc = jnp.concatenate(jax.tree_map(jnp.atleast_1d, jax.tree_leaves(self.params)))
        if len(loc) == 1:
            dist = tfd.Normal(loc.ravel(), covariance_matrix.ravel()**0.5)
        else:
            dist = tfd.MultivariateNormalFullCovariance(loc, covariance_matrix)
        return dist

    def sample(self, seed, sample_shape=()):
        dist = self.get_normal_dist(self.params.keys())
        normal_sample = dist.sample(sample_shape=sample_shape, seed=seed)
        split_sample = jnp.split(normal_sample, self.cumsum_size, axis=-1)
        split_sample = jax.tree_unflatten(self.guide, split_sample)
        return jax.tree_map(lambda sample, bijector: bijector.forward(sample), split_sample, self.bijectors)

    def log_prob(self, sample):
        normal_sample = jax.tree_map(lambda sample, bijector: bijector.inverse(sample), sample, self.bijectors)
        log_jacobian = jax.tree_map(lambda samp, bijector: bijector.inverse_log_det_jacobian(samp), sample, self.bijectors)
        log_jacobian = jax.vmap(lambda x: sum(jax.tree_leaves(x)))(log_jacobian)
        flat_sample = jnp.concatenate(jax.tree_map(jnp.atleast_1d, jax.tree_leaves(normal_sample)))
        dist = self.get_normal_dist(sample.keys())
        normal_log_prob = dist.log_prob(flat_sample)
        assert normal_log_prob.shape == log_jacobian.shape
        return normal_log_prob + log_jacobian
    
    def sample_and_log_prob(self, seed, sample_shape=()):
        samples = self.sample(seed, sample_shape)
        return samples, self.log_prob(samples)
        
        