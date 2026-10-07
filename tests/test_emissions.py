# GSPN-GPT-FIXED: Analytic checks use the canonical component-density emission interface.
import itertools
import math

import pytest
import torch
from torch_geometric.data import Batch, Data

from model import (
    GSPN,
    CategoricalParameters,
    GaussianParameters,
    GSPNCategoricalEmission,
    GSPNEmission,
    GSPNGaussianEmission,
    GSPNMultiCategoricalEmission,
    MultiCategoricalParameters,
    SupGSPN,
    infer_mixture,
)


def config(emission="model.GSPNGaussianEmission", layers=2, **kwargs):
    return dict(
        num_layers=layers,
        num_mixtures=2,
        emission_class=emission,
        convolution_class="model.GSPNBaseConv",
        avg_parameters_across_layers=False,
        **kwargs,
    )


def graph():
    return Batch.from_data_list(
        [
            Data(
                x=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                edge_index=torch.tensor([[0, 1], [1, 0]]),
                y=torch.tensor([1]),
                mask=torch.tensor([[True, False], [False, False], [True, True]]),
            )
        ]
    )


def test_categorical_missing_and_integer_column():
    x = torch.tensor([[1.0], [float("nan")]])
    before = x.clone()
    probabilities = torch.tensor([[0.8, 0.2], [0.1, 0.9]], requires_grad=True)
    params = CategoricalParameters(probabilities)
    weights = torch.tensor([[0.3, 0.7], [0.3, 0.7]], requires_grad=True)
    emission = GSPNCategoricalEmission(2, 2)
    components = emission.component_log_prob(
        x, params=params, observed_mask=torch.tensor([[True], [False]])
    )
    ll, posterior = infer_mixture(components, weights)
    assert ll[0].item() == pytest.approx(math.log(0.3 * 0.2 + 0.7 * 0.9))
    assert ll[1].item() == pytest.approx(0, abs=1e-6)
    torch.testing.assert_close(posterior[1], weights[1])
    torch.testing.assert_close(x, before, equal_nan=True)
    (-ll.sum()).backward()
    assert torch.isfinite(probabilities.grad).all()


def test_multicategorical_joint_normalizes_and_masks():
    emission = GSPNMultiCategoricalEmission(2, 2, [2, 3])
    params = MultiCategoricalParameters(
        (
            CategoricalParameters(torch.tensor([[0.8, 0.2], [0.1, 0.9]])),
            CategoricalParameters(torch.tensor([[0.1, 0.3, 0.6], [0.7, 0.2, 0.1]])),
        )
    )
    assignments = torch.tensor(list(itertools.product(range(2), range(3))))
    weights = torch.tensor([[0.4, 0.6]]).expand(6, -1)
    ll, _ = infer_mixture(emission.component_log_prob(assignments, params=params), weights)
    assert ll.exp().sum().item() == pytest.approx(1.0)
    expected = 0.4 * 0.8 * 0.1 + 0.6 * 0.1 * 0.7
    assert ll[0].exp().item() == pytest.approx(expected)
    components = emission.component_log_prob(
        assignments, params=params, observed_mask=torch.tensor([[True, False]]).expand(6, -1)
    )
    ll_mask, _ = infer_mixture(components, weights)
    assert ll_mask[0].exp().item() == pytest.approx(0.4 * 0.8 + 0.6 * 0.1)
    imputed = emission.predictive_mean(params, weights=weights)
    torch.testing.assert_close(imputed[:, :2].sum(-1), torch.ones(6))
    torch.testing.assert_close(imputed[:, 2:].sum(-1), torch.ones(6))


def test_gaussian_marginal_and_shortcut():
    params = GaussianParameters(
        torch.tensor([[0.0, 4.0], [5.0, 8.0]]), torch.tensor([[2.0, 3.0], [1.0, 2.0]])
    )
    x = torch.tensor([[1.0, float("nan")]])
    weights = torch.tensor([[0.25, 0.75]])
    emission = GSPNGaussianEmission(2, 2)
    components = emission.component_log_prob(
        x, params=params, observed_mask=torch.tensor([[True, False]])
    )
    ll, _ = infer_mixture(components, weights)
    expected = (
        0.25 * torch.distributions.Normal(0.0, 2.0).log_prob(torch.tensor(1.0)).exp()
        + 0.75 * torch.distributions.Normal(5.0, 1.0).log_prob(torch.tensor(1.0)).exp()
    )
    torch.testing.assert_close(ll.exp(), expected.reshape(1))
    averaged = emission.combine_shortcut_parameters([params, params])
    torch.testing.assert_close(averaged.stddev, params.stddev / math.sqrt(2))
    torch.testing.assert_close(averaged.mean, params.mean)


@pytest.mark.parametrize("cls", [GSPN, SupGSPN])
@pytest.mark.parametrize("layers", [1, 3])
@pytest.mark.parametrize("shortcut", [False, True])
def test_model_missing_evidence_and_gradients(cls, layers, shortcut):
    cfg = config(layers=layers)
    cfg.update(avg_parameters_across_layers=shortcut, num_graph_mixtures=3, global_readout="mean")
    model = cls((2, 0), 2, cfg)
    data = graph()
    before = data.x.clone()
    result = model.infer(data)
    preds, emb, extras = result.to_reference_outputs()
    assert len(extras) == 10
    assert emb.shape == (3, layers * 2)
    assert torch.isfinite(extras[0]).all()
    torch.testing.assert_close(emb.reshape(3, layers, 2).sum(-1), torch.ones(3, layers))
    torch.testing.assert_close(data.x, before)
    torch.testing.assert_close(extras[5], ~data.mask)
    final_h = emb.reshape(3, layers, 2)[:, -1]
    expected = model.emissions[-1].predictive_mean(result.layers[-1].parameters, weights=final_h)
    torch.testing.assert_close(extras[4], expected)
    loss = -extras[0].mean() + (preds.square().mean() if preds is not None else 0)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.testing.assert_close(final_h[1], extras[7][1])


def test_kmeans_small_batch_ignores_hidden_truth():
    data = graph()
    data.x[~data.mask] = 12345
    model = GSPN((2, 0), 2, config(init_kmeans=True, init_max_variance=4))
    model(data)
    assert model.initialized.item()
    assert model.emissions[0].mean.max() < 10
    assert model.emissions[0].distribution_parameters().stddev.square().max() <= 4.0001
    with pytest.raises(ValueError):
        GSPN((2, 0), 2, config("model.GSPNCategoricalEmission", init_kmeans=True))


def test_abstract_emission_contract():
    with pytest.raises(TypeError):
        GSPNEmission(1, 1)
