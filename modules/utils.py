import torch
import numpy as np
from functools import reduce, partial
from operator import mul
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations


def chain_functions(*functions):
    return lambda initial: reduce(lambda x, f: f(x), functions, initial)


def remove_fx_parametrisation(fx):
    def remover(m):
        if not is_parametrized(m):
            return
        for k in list(m.parametrizations.keys()):
            remove_parametrizations(m, k)

    fx.apply(remover)
    return fx


def get_chunks(keys, original_shapes):
    (position, _), *_ = filter(lambda i_k: "U.original" in i_k[1], enumerate(keys))
    original_chunks = list(map(partial(reduce, mul), original_shapes))
    U_matrix_shape = original_shapes[position]

    dimensions_not_need = np.ravel_multi_index(
        np.tril_indices(**dict(zip(("n", "m"), U_matrix_shape))), U_matrix_shape
    ) + sum(original_chunks[:position])

    selected_chunks = (
        original_chunks[:position]
        + [original_chunks[position] - dimensions_not_need.size]
        + original_chunks[position + 1 :]
    )
    return selected_chunks, position, U_matrix_shape, dimensions_not_need


def vec2statedict(
    x: torch.Tensor,
    keys,
    original_shapes,
    selected_chunks,
    position,
    U_matrix_shape,
):
    chunks = list(torch.split(x, selected_chunks))
    U = x.new_zeros(reduce(mul, U_matrix_shape))
    U[
        np.ravel_multi_index(
            np.triu_indices(n=U_matrix_shape[0], k=1, m=U_matrix_shape[1]),
            U_matrix_shape,
        )
    ] = chunks[position]
    chunks[position] = U

    state_dict = dict(
        zip(
            keys,
            map(lambda x, shape: x.reshape(*shape), chunks, original_shapes),
        )
    )
    return state_dict
