import pytest

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from lapjax import ADLaplace


@pytest.mark.parametrize("data", [jnp.array([1]), jnp.array([1, 0])])
def coin_toss_test(data):
    prior = {"p_of_heads": tfd.Beta(2, 3)}
    bijectors = {"p_of_heads": tfb.Sigmoid()}
    likelihood = tfd.Bernoulli
    get_likelihood_params = lambda params, aux: {"probs": params["p_of_heads"]}

    laplace = ADLaplace(prior, bijectors, likelihood, get_likelihood_params)

    init_seed = jax.random.PRNGKey(0)
    params = laplace.init(init_seed)
    loss = laplace.loss_fun(params, data)
    posterior = laplace.apply(params, data)
    sample = posterior.sample(seed=jax.random.PRNGKey(0))

    pass


def linear_regression_test():
    seed = jax.random.PRNGKey(0)
    N = 100
    D = 3

    prior = {
        "weights": tfd.MultivariateNormalDiag(loc=jnp.zeros(D), scale_diag=jnp.ones(D)),
        "noise_scale": tfd.Gamma(1, 1),
    }
    bijectors = {"weights": tfb.Identity(), "noise_scale": tfb.Exp()}
    likelihood = tfd.Normal

    X = jax.random.uniform(seed, shape=(N, D))
    y = 4 * X.sum(axis=1) + jax.random.normal(jax.random.PRNGKey(0), shape=(N,))

    def get_likelihood_params(params, aux):
        return {"loc": (aux["x"] * params["weights"]).sum(), "scale": params["noise_scale"]}

    laplace = ADLaplace(prior, bijectors, likelihood, get_likelihood_params)

    init_seed = jax.random.PRNGKey(0)
    params = laplace.init(init_seed)
    loss = laplace.loss_fun(params, data=y, aux={"x": X})
    posterior = laplace.apply(params, data=y, aux={"x": X})

    pass
