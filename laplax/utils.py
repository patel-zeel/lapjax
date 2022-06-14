import jax
import optax


def train_model(model, data, aux, optimizer, n_epochs, seed):
    params = model.init(seed)
    value_and_grad_fn = jax.value_and_grad(model.loss_fn)
    state = optimizer.init(params)

    @jax.jit
    def one_step(params_and_state, xs):
        params, state = params_and_state
        loss, grads = value_and_grad_fn(params, data, aux)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
        return (params, state), loss

    params_and_states, losses = jax.lax.scan(one_step, (params, state), xs=None, length=n_epochs)
    return params_and_states[0], losses


def seeds_like(seed, params):
    values, treedef = jax.tree_flatten(params)
    return jax.tree_unflatten(treedef, jax.random.split(seed, len(values)))


def fill_params(seed, params, initializer):
    assert seed is not None
    values, treedef = jax.tree_flatten(params)
    seeds = seeds_like(seed, values)
    values = jax.tree_map(lambda seed, value: initializer(seed, value.shape), seeds, values)
    return jax.tree_unflatten(treedef, values)
