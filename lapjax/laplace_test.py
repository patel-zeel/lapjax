from distrax import Bijector
import pytest

import jax
import jax.numpy as jnp
from sklearn import mixture
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from lapjax import ADLaplace

import optax

## Problems
def get_coin_toss():
    h0 = 2.0
    t0 = 5.0
    h = 50
    t = 10

    prior = {"p_of_heads": tfd.Beta(h0, t0)}
    bijectors = {"p_of_heads": tfb.Sigmoid()}
    likelihood = tfd.Bernoulli
    get_likelihood_params = lambda params, aux: {"probs": params["p_of_heads"]}
    data = jnp.array([1] * h + [0] * t)
    aux = None
    return prior, bijectors, likelihood, get_likelihood_params, data, aux


def get_mean_and_cov_estimate():
    d = 5
    prior = {
        "mu": tfd.MultivariateNormalDiag(loc=jnp.zeros(d), scale_diag=jnp.ones(d)),
        "corr_chol": tfd.CholeskyLKJ(d, 0.5),
        "sigma_diag": tfd.Independent(tfd.Gamma(jnp.ones(5), jnp.ones(5)), reinterpreted_batch_ndims=1),
    }
    bijectors = {"mu": tfb.Identity(), "corr_chol": tfb.CorrelationCholesky(), "sigma_diag": tfb.Exp()}
    likelihood = tfd.MultivariateNormalTriL

    def get_likelihood_params(params, aux):
        sigma = jnp.diag(params["sigma_diag"])
        scale_tril = sigma @ params["corr_chol"]
        return {"loc": params["mu"], "scale_tril": scale_tril}

    data = jax.random.normal(jax.random.PRNGKey(0), shape=(100, d))
    aux = None
    return prior, bijectors, likelihood, get_likelihood_params, data, aux


def get_lin_reg():
    prior = {"theta": tfd.MultivariateNormalDiag(loc=[0.0, 1.0], scale_diag=[1.0, 2.0]), "sigma": tfd.Gamma(1, 1)}
    bijectors = {"theta": tfb.Identity(), "sigma": tfb.Exp()}
    likelihood = tfd.Normal

    def get_likelihood_params(params, aux):
        mean = jnp.sum(aux["x"] * params["theta"])
        scale = params["sigma"]
        return {"loc": mean, "scale": scale}

    aux = {"x": jax.random.uniform(jax.random.PRNGKey(0), shape=(100, 2))}
    data = jax.random.uniform(jax.random.PRNGKey(1), shape=(100,))
    return prior, bijectors, likelihood, get_likelihood_params, data, aux


def get_gmm_1d():
    prior = {}
    bijectors = {}
    K = 5
    for i in range(K):
        prior.update({f"mu_{i}": tfd.Normal(loc=0.0, scale=1.0), f"sigma_{i}": tfd.Gamma(1.0, 1.0)})
        bijectors.update({f"mu_{i}": tfb.Identity(), f"sigma_{i}": tfb.Exp()})

    prior.update({"pi": tfd.Dirichlet(2.0 * jnp.ones(K))})
    bijectors.update({"pi": tfb.SoftmaxCentered()})
    likelihood = tfd.MixtureSameFamily

    def get_likelihood_params(params, aux):
        mixture_distribution = tfd.Categorical(probs=params["pi"])
        loc_list = []
        scale_list = []
        for i in range(len(params["pi"])):
            loc_list.append(params[f"mu_{i}"])
            scale_list.append(params[f"sigma_{i}"])

        components_distribution = tfd.Normal(loc=loc_list, scale=scale_list)
        return {"mixture_distribution": mixture_distribution, "components_distribution": components_distribution}

    aux = None
    data = jax.random.uniform(jax.random.PRNGKey(1), shape=(100,))
    return prior, bijectors, likelihood, get_likelihood_params, data, aux


def get_gmm_2d():
    prior = {}
    bijectors = {}
    K = 5
    d = 3
    for i in range(K):
        prior.update(
            {
                f"mu_{i}": tfd.MultivariateNormalDiag(loc=jnp.zeros(d), scale_diag=jnp.ones(d)),
                f"chol_corr_{i}": tfd.CholeskyLKJ(d, 0.5),
                f"sigma_{i}": tfd.Independent(tfd.Gamma(jnp.ones(d), jnp.ones(d)), reinterpreted_batch_ndims=1),
            }
        )
        bijectors.update(
            {f"mu_{i}": tfb.Identity(), f"chol_corr_{i}": tfb.CorrelationCholesky(), f"sigma_{i}": tfb.Exp()}
        )

    prior.update({"pi": tfd.Dirichlet(2.0 * jnp.ones(K))})
    bijectors.update({"pi": tfb.SoftmaxCentered()})
    likelihood = tfd.MixtureSameFamily

    def get_likelihood_params(params, aux):
        mixture_distribution = tfd.Categorical(probs=params["pi"])
        loc_list = []
        scale_tril_list = []
        for i in range(len(params["pi"])):
            loc_list.append(params[f"mu_{i}"])
            scale_tril_list.append(jnp.diag(params[f"sigma_{i}"]) @ params[f"chol_corr_{i}"])

        components_distribution = tfd.MultivariateNormalTriL(loc=loc_list, scale_tril=scale_tril_list)
        return {"mixture_distribution": mixture_distribution, "components_distribution": components_distribution}

    aux = None
    data = jax.random.uniform(jax.random.PRNGKey(1), shape=(100, d))
    return prior, bijectors, likelihood, get_likelihood_params, data, aux


problems = [get_coin_toss, get_mean_and_cov_estimate, get_lin_reg, get_gmm_1d, get_gmm_2d]

## Useful functions
def get_init_params(prior, bijectors):
    laplace = ADLaplace(prior, bijectors)
    init_seed = jax.random.PRNGKey(0)
    params = laplace.init(init_seed)
    return params


def get_loss(prior, bijectors, likelihood, get_likelihood_params, data, aux):
    laplace = ADLaplace(prior, bijectors, likelihood, get_likelihood_params)
    params = laplace.init(jax.random.PRNGKey(0))
    loss = laplace.loss_fun(params, data, aux)
    return loss


## Test parameter initialization
@pytest.mark.parametrize("problem", problems)
def init_test(problem):
    prior, bijectors, *_ = problem()
    get_init_params(prior, bijectors)


## Test loss computation
@pytest.mark.parametrize("problem", problems)
def loss_test(problem):
    prior, bijectors, likelihood, get_likelihood_params, data, aux = problem()
    loss = get_loss(prior, bijectors, likelihood, get_likelihood_params, data, aux)
    assert loss.shape == ()


## Test optimize
@pytest.mark.parametrize("problem", problems)
def optimize_test(problem):
    prior, bijectors, likelihood, get_likelihood_params, data, aux = problem()
    laplace = ADLaplace(prior, bijectors, likelihood, get_likelihood_params)
    value_and_grad_fun = jax.jit(jax.value_and_grad(laplace.loss_fun))

    tx = optax.adam(learning_rate=0.1)

    params = laplace.init(jax.random.PRNGKey(0))
    state = tx.init(params)

    epochs = 10

    losses = []

    for _ in range(epochs):
        loss, grads = value_and_grad_fun(params, data, aux)
        losses.append(loss)
        updates, state = tx.update(grads, state)
        params = optax.apply_updates(params, updates)
    assert losses[-1] < losses[0]
