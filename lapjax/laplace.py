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
    def __init__(self, prior, bijectors, likelihood, get_likelihood_params):
        self.prior = prior
        self.guide = {key: 0 for key in self.prior}
        self.bijectors = bijectors

        self.likelihood = likelihood
        self.get_likelihood_params = get_likelihood_params
        self.transformed_distributions = jax.tree_map(
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
            self.transformed_distributions,
        )
        for param in params.values():
            assert param.ndim <= 1, "params must be scalar or vector"
        return params

    def loss_fun(self, params, data, aux=None):
        transformed_params = jax.tree_map(lambda param, bijector: bijector.forward(param), params, self.bijectors)
        prior_log_probs = jax.tree_map(
            lambda param, dist: dist.log_prob(param),
            params,
            self.transformed_distributions,
        )

        def likelihood_log_prob(params, data, aux):
            likelihood_params = self.get_likelihood_params(params, aux)
            likelihood = self.likelihood(**likelihood_params)
            return likelihood.log_prob(data)

        likelihood_log_probs = jax.vmap(likelihood_log_prob, in_axes=(None, 0, 0))(transformed_params, data, aux)
        loss = -likelihood_log_probs.sum() - sum(jax.tree_leaves(prior_log_probs))
        return loss / len(data)

    def apply(self, params, data, aux=None):
        precision = jax.hessian(self.loss_fun)(params, data, aux)
        return Posterior(params, precision, self.bijectors)


class Posterior:
    def __init__(self, params, precision, bijectors):
        self.params = jax.tree_map(lambda x: x, params)
        self.guide = jax.tree_structure(self.params)
        self.precision = jax.tree_map(lambda x: x, precision)
        self.bijectors = jax.tree_map(lambda x: x, bijectors)
        self.size = jax.tree_map(lambda param: param.size, self.params)
        self.cumsum_size = np.cumsum(jax.tree_leaves(self.size))[:-1]

    def untree_precision(self):
        covariance_matrix = []
        for start in self.precision:
            covariance_matrix.append([])
            for end in self.precision[start]:
                element = self.precision[start][end]
                element = jnp.atleast_2d(element)
                if start > end:
                    element = element.T
                covariance_matrix[-1].append(element)
        return jnp.block(covariance_matrix)

    def get_normal_dist(self):
        precision_matrix = self.untree_precision()
        covariance_matrix = jnp.linalg.inv(precision_matrix)
        loc = jnp.concatenate(jax.tree_map(jnp.atleast_1d, jax.tree_leaves(self.params)))
        dist = tfd.MultivariateNormalFullCovariance(loc, covariance_matrix)
        return dist

    def sample(self, seed, sample_shape=()):
        dist = self.get_normal_dist()
        normal_sample = dist.sample(sample_shape=sample_shape, seed=seed)
        split_sample = jax.tree_unflatten(self.guide, jnp.split(normal_sample, self.cumsum_size, axis=-1))
        return jax.tree_map(lambda sample, bijector: bijector.forward(sample), split_sample, self.bijectors)


    # TODO: implement log_prob
    # def log_prob(self, sample):
    #     sample = jax.tree_map(lambda x: x, sample)
    #     dist = self.get_normal_dist()
        