import logging
from statistics import covariance

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())

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

    def loss_fun(self, params, data, aux):
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

    def apply(self, params, data, aux):
        precision = jax.hessian(self.loss_fun)(params, data, aux)
        return Posterior(params, precision)


class Posterior:
    def __init__(self, params, precision):
        self.params = params
        self.precision = precision

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
