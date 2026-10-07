"""
Graph-Induced Sum-Product Networks

Files: model.py

Authors:  Federico Errica (federico.errica@neclab.eu)
     Mathias Niepert (mathias.niepert@ki.uni-stuttgart.de)

NEC Laboratories Europe GmbH, Copyright (c) 2024, All rights reserved.

THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

PROPRIETARY INFORMATION ---

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor.

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""

# GSPN-GPT-FIXED: Canonical GSPN inference; the superseded implementations are retired.
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from mlwiz.model.interface import ModelInterface
from mlwiz.util import s2c
from sklearn.cluster import KMeans
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
from torch_geometric.utils import add_self_loops, remove_self_loops


# GSPN-GPT-FIXED: Retain the dimension helper used by graph baselines and predictor modules.
def graph_dimensions(dim_input_features):
    return (
        tuple(dim_input_features)
        if isinstance(dim_input_features, (tuple, list))
        else (dim_input_features, 0)
    )


# GSPN-GPT-FIXED: Named, normalized distribution parameters hide packed tensor layouts.
@dataclass(frozen=True)
class CategoricalParameters:
    probabilities: Tensor  # [components, categories]

    def to_reference_tensor(self):
        return self.probabilities.unsqueeze(0)


@dataclass(frozen=True)
class GaussianParameters:
    mean: Tensor  # [components, features]
    stddev: Tensor

    def to_reference_tensor(self):
        return torch.stack((self.mean, self.stddev), dim=-1).unsqueeze(0)


@dataclass(frozen=True)
class MultiCategoricalParameters:
    blocks: tuple[CategoricalParameters, ...]

    def to_reference_tensor(self):
        return torch.cat([block.probabilities for block in self.blocks], dim=-1)


DistributionParameters = CategoricalParameters | GaussianParameters | MultiCategoricalParameters


# GSPN-GPT-FIXED: One observed-mask convention; broadcasting never mutates inputs.
def observed_feature_mask(x: Tensor, observed_mask: Tensor | None = None) -> Tensor:
    if observed_mask is None:
        return torch.ones_like(x, dtype=torch.bool)
    mask = observed_mask.to(device=x.device, dtype=torch.bool)
    if mask.ndim == 1 and x.ndim == 2:
        mask = mask[:, None]
    return torch.broadcast_to(mask, x.shape)


# GSPN-GPT-FIXED: Normalize priors once per query, preserving exact zero support.
def _joint_log_prob(component_log_prob: Tensor, prior: Tensor) -> Tensor:
    normalized = prior / prior.sum(-1, keepdim=True)
    log_prior = torch.where(
        normalized > 0,
        normalized.clamp_min(torch.finfo(normalized.dtype).tiny).log(),
        -torch.inf,
    )
    return component_log_prob + log_prior


def infer_mixture(component_log_prob: Tensor, prior: Tensor) -> tuple[Tensor, Tensor]:
    joint = _joint_log_prob(component_log_prob, prior)
    return torch.logsumexp(joint, dim=-1), torch.softmax(joint, dim=-1)


def mixture_log_prob(component_log_prob: Tensor, prior: Tensor) -> Tensor:
    return torch.logsumexp(_joint_log_prob(component_log_prob, prior), dim=-1)


# GSPN-GPT-FIXED: Emissions describe components; mixture inference belongs to the caller.
class GSPNEmission(nn.Module, ABC):
    supports_kmeans = False

    def __init__(self, dim_observable: int, num_components: int):
        super().__init__()
        self.dim_observable = dim_observable
        self.num_components = num_components

    @classmethod
    def from_dimensions(cls, dim_observable, num_components, categories=None):
        return cls(dim_observable, num_components)

    @abstractmethod
    def distribution_parameters(self) -> DistributionParameters:
        raise NotImplementedError

    @abstractmethod
    def component_log_prob(self, x, *, params, observed_mask=None) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def predictive_mean(self, params, *, weights) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def combine_shortcut_parameters(self, parameters_per_layer) -> DistributionParameters:
        raise NotImplementedError

    @abstractmethod
    def valid_values(self, x) -> Tensor:
        raise NotImplementedError

    def reference_state_mapping(self):
        """Custom emissions may override this to translate their reference parameter names."""
        return {key: key for key in self.state_dict()}

    def initialize_from_centers(self, centers, max_variance):
        raise NotImplementedError("This emission does not support Gaussian K-means")


# GSPN-GPT-FIXED: A categorical component evaluates evidence without mixing components.
class GSPNCategoricalEmission(GSPNEmission):
    def __init__(self, num_categories, num_components):
        super().__init__(num_categories, num_components)
        self.num_categories = num_categories
        self.category_logits = nn.Parameter(torch.rand(num_components, num_categories))

    def distribution_parameters(self):
        return CategoricalParameters(torch.softmax(self.category_logits, dim=1))

    def valid_values(self, x):
        valid = torch.isfinite(x)
        if x.ndim == 1 or x.shape[1] == 1:
            valid = valid & (x >= 0) & (x < self.num_categories) & (x == x.floor())
        return valid

    def component_log_prob(self, x, *, params, observed_mask=None):
        observed = observed_feature_mask(x, observed_mask)
        if x.ndim == 2 and x.shape[1] > 1:
            if torch.any(observed.any(1) != observed.all(1)):
                raise ValueError("A one-hot categorical variable must be masked as a whole")
            available = observed.all(1)
            labels = torch.where(observed, x, torch.zeros_like(x)).argmax(1)
        else:
            available = observed.reshape(-1)
            labels = torch.where(available, x.reshape(-1), torch.zeros_like(x.reshape(-1)))
        if torch.any(available & ~self.valid_values(labels)):
            raise ValueError("Observed categorical labels must be valid integer category IDs")
        probabilities = params.probabilities.unsqueeze(0).expand(labels.shape[0], -1, -1)
        components = (
            probabilities.gather(2, labels.long()[:, None, None].expand(-1, self.num_components, 1))
            .squeeze(2)
            .log()
        )
        return torch.where(available[:, None], components, torch.zeros_like(components))

    def predictive_mean(self, params, *, weights):
        return (params.probabilities.unsqueeze(0) * weights[:, :, None]).sum(1)

    def combine_shortcut_parameters(self, parameters_per_layer):
        return CategoricalParameters(
            torch.stack([p.probabilities for p in parameters_per_layer]).mean(0)
        )

    def reference_state_mapping(self):
        return {"categorical_probs": "category_logits"}


# GSPN-GPT-FIXED: Independent feature evidence shares a single latent mixture component.
class GSPNMultiCategoricalEmission(GSPNEmission):
    def __init__(self, dim_observable, num_components, categories):
        super().__init__(dim_observable, num_components)
        self.categories = tuple(categories)
        if len(self.categories) != dim_observable or any(c < 1 for c in self.categories):
            raise ValueError("One positive category count is required per input feature")
        self.emissions = nn.ModuleList(
            GSPNCategoricalEmission(c, num_components) for c in self.categories
        )

    @classmethod
    def from_dimensions(cls, dim_observable, num_components, categories=None):
        if categories is None:
            raise ValueError("Multi-categorical emissions require dim_categorical_features")
        return cls(dim_observable, num_components, categories)

    def distribution_parameters(self):
        return MultiCategoricalParameters(
            tuple(emission.distribution_parameters() for emission in self.emissions)
        )

    def valid_values(self, x):
        return torch.stack(
            [emission.valid_values(x[:, i]) for i, emission in enumerate(self.emissions)], dim=1
        )

    def component_log_prob(self, x, *, params, observed_mask=None):
        observed = observed_feature_mask(x, observed_mask)
        components = params.blocks[0].probabilities.new_zeros(x.shape[0], self.num_components)
        for i, (emission, block) in enumerate(zip(self.emissions, params.blocks)):
            components = components + emission.component_log_prob(
                x[:, i], params=block, observed_mask=observed[:, i]
            )
        return components

    def predictive_mean(self, params, *, weights):
        return weights @ params.to_reference_tensor()

    def combine_shortcut_parameters(self, parameters_per_layer):
        return MultiCategoricalParameters(
            tuple(
                emission.combine_shortcut_parameters([p.blocks[i] for p in parameters_per_layer])
                for i, emission in enumerate(self.emissions)
            )
        )

    def reference_state_mapping(self):
        return {
            f"emissions.{i}.{old}": f"emissions.{i}.{new}"
            for i, emission in enumerate(self.emissions)
            for old, new in emission.reference_state_mapping().items()
        }


# GSPN-GPT-FIXED: Named Gaussian views retain packed trainable state and RNG draw order.
class GSPNGaussianEmission(GSPNEmission):
    supports_kmeans = True

    def __init__(self, dim_observable, num_components):
        super().__init__(dim_observable, num_components)
        self.raw_parameters = nn.Parameter(torch.rand(num_components, dim_observable, 2))

    @property
    def mean(self):
        return self.raw_parameters[..., 0]

    @property
    def raw_scale(self):
        return self.raw_parameters[..., 1]

    def distribution_parameters(self):
        return GaussianParameters(self.mean, torch.nn.functional.softplus(self.raw_scale) + 1e-8)

    def valid_values(self, x):
        return torch.isfinite(x)

    def component_log_prob(self, x, *, params, observed_mask=None):
        observed = observed_feature_mask(x, observed_mask)
        safe = torch.where(observed, x, torch.zeros_like(x))
        normal = torch.distributions.Normal(params.mean.unsqueeze(0), params.stddev.unsqueeze(0))
        feature_log_prob = normal.log_prob(safe[:, None, :])
        return torch.where(
            observed[:, None, :], feature_log_prob, torch.zeros_like(feature_log_prob)
        ).sum(-1)

    def predictive_mean(self, params, *, weights):
        return (params.mean.unsqueeze(0) * weights[:, :, None]).sum(1)

    def combine_shortcut_parameters(self, parameters_per_layer):
        count = len(parameters_per_layer)
        return GaussianParameters(
            torch.stack([p.mean for p in parameters_per_layer]).mean(0),
            torch.stack([p.stddev for p in parameters_per_layer]).square().sum(0).sqrt() / count,
        )

    def reference_state_mapping(self):
        return {"normal_params": "raw_parameters"}

    def initialize_from_centers(self, centers, max_variance):
        with torch.no_grad():
            self.mean.copy_(centers)
            scale = (torch.rand_like(self.raw_scale) * max_variance).clamp_min(1e-4).sqrt()
            self.raw_scale.copy_(scale + torch.log(-torch.expm1(-scale)))


# GSPN-GPT-FIXED: Transitions encapsulate pair-component reduction and never edit topology.
class GSPNBaseConv(MessagePassing):
    def __init__(self, num_components, use_prior):
        super().__init__(aggr="mean")
        self.num_components = num_components
        self.use_prior = use_prior
        shape = (num_components,) if use_prior else (num_components, num_components)
        values = torch.rand(shape)
        self.transition_logits = nn.Parameter(values / values.sum(dim=0, keepdim=True))

    def forward(self, previous_posterior, *, edge_index, num_nodes):
        table = torch.softmax(self.transition_logits, dim=0).unsqueeze(0)
        if self.use_prior:
            return table.expand(num_nodes, -1)
        weighted = table * previous_posterior.unsqueeze(1)
        aggregated = self.propagate(
            edge_index, x=weighted.reshape(num_nodes, -1), size=(num_nodes, num_nodes)
        )
        return aggregated.reshape(num_nodes, self.num_components, self.num_components).sum(-1)

    def reference_state_mapping(self):
        return {"transition_table": "transition_logits"}


# GSPN-GPT-FIXED: Named results replace positional tuples inside the model only.
@dataclass(frozen=True)
class Evidence:
    x: Tensor
    observed_mask: Tensor
    has_mask: bool
    edge_index: Tensor
    batch: Tensor
    targets: Tensor | None


@dataclass(frozen=True)
class LayerResult:
    prior: Tensor
    parameters: DistributionParameters
    component_log_prob: Tensor
    log_prob: Tensor
    posterior: Tensor


@dataclass(frozen=True)
class GraphResult:
    predictions: Tensor
    log_prob: Tensor | None = None
    prior: Tensor | None = None
    parameters: DistributionParameters | None = None
    component_log_prob: Tensor | None = None


@dataclass(frozen=True)
class InferenceResult:
    evidence: Evidence
    layers: tuple[LayerResult, ...]
    node_posteriors: Tensor  # [nodes, layers, components]
    imputation: Tensor | None = None
    conditional_log_prob: Tensor | None = None
    graph: GraphResult | None = None

    @property
    def embeddings(self):
        return self.node_posteriors.reshape(self.evidence.x.shape[0], -1)

    @property
    def predictions(self):
        return None if self.graph is None else self.graph.predictions

    def to_reference_outputs(self):
        final = self.layers[-1]
        observed = self.evidence.observed_mask if self.evidence.has_mask else None
        extras = [
            final.log_prob,
            None if self.graph is None else self.graph.log_prob,
            self.evidence.x,
            self.evidence.x,
            self.imputation,
            None if observed is None else ~observed,
            observed,
            final.prior,
            final.parameters.to_reference_tensor(),
            self.conditional_log_prob,
        ]
        return self.predictions, self.embeddings, extras


# GSPN-GPT-FIXED: Graph heads predict without targets and score only supplied targets.
class GraphHead(nn.Module, ABC):
    @abstractmethod
    def forward(self, node_posteriors, batch, targets=None) -> GraphResult:
        raise NotImplementedError

    def reference_state_mapping(self):
        return {key: key for key in self.state_dict()}


def _pooling(name):
    if name not in ("sum", "mean"):
        raise ValueError("Graph pooling must be sum or mean")
    return global_add_pool if name == "sum" else global_mean_pool


# GSPN-GPT-FIXED: Share probabilistic readout stages while preserving each variant's math.
class ProbabilisticGraphReadout(GraphHead):
    use_layer_attention = True
    local_activation = staticmethod(lambda x: torch.softmax(x, dim=1))

    def __init__(self, dim_target, config):
        super().__init__()
        layers = config["num_layers"]
        width = config["num_mixtures"] * layers
        values = torch.rand(layers)
        self.layer_logits = nn.Parameter(values / values.sum())
        self.node_transform = nn.Linear(width, width)
        self.graph_transform = nn.Linear(width, width)
        self.pool = _pooling(config["global_pooling"])
        self.emission = _resolve_type(config["graph_emission_class"], GSPNEmission).from_dimensions(
            dim_target, width, config.get("graph_dim_categorical_features")
        )

    def mixture_weights(self, node_posteriors, batch):
        if self.use_layer_attention:
            attention = torch.softmax(self.layer_logits, dim=0)[None, :, None]
            node_posteriors = attention * node_posteriors
        local = self.local_activation(
            self.node_transform(node_posteriors.reshape(node_posteriors.shape[0], -1))
        )
        return torch.softmax(self.graph_transform(self.pool(local, batch)), dim=1)

    def forward(self, node_posteriors, batch, targets=None):
        prior = self.mixture_weights(node_posteriors, batch)
        params = self.emission.distribution_parameters()
        predictions = self.emission.predictive_mean(params, weights=prior)
        components = log_prob = None
        if targets is not None:
            components = self.emission.component_log_prob(targets, params=params)
            log_prob = mixture_log_prob(components, prior)
        return GraphResult(predictions, log_prob, prior, params, components)

    def reference_state_mapping(self):
        return {
            "Lg": "layer_logits",
            **{
                key: key
                for key in self.state_dict()
                if key.startswith(("node_transform.", "graph_transform.", "out."))
            },
            **{
                f"emission.{old}": f"emission.{new}"
                for old, new in self.emission.reference_state_mapping().items()
            },
        }


class ProbabilisticGraphReadoutNoLayerAttention(ProbabilisticGraphReadout):
    use_layer_attention = False


class ProbabilisticGraphReadoutNoLayerAttentionMLPVersion2(
    ProbabilisticGraphReadoutNoLayerAttention
):
    local_activation = staticmethod(torch.relu)


# GSPN-GPT-FIXED: Retain unused legacy readout parameters for faithful state/gradient transfer.
class ProbabilisticGraphReadoutNoLayerAttentionMLP(ProbabilisticGraphReadout):
    def __init__(self, dim_target, config):
        super().__init__(dim_target, config)
        self.out = nn.Linear(config["num_mixtures"] * config["num_layers"], dim_target)

    def forward(self, node_posteriors, batch, targets=None):
        local = torch.relu(
            self.node_transform(node_posteriors.reshape(node_posteriors.shape[0], -1))
        )
        pooled = self.pool(local, batch)
        predictions = self.out(torch.relu(self.graph_transform(pooled)))
        log_prob = (
            None
            if targets is None
            else -torch.nn.functional.cross_entropy(predictions, targets, reduction="none")
        )
        return GraphResult(predictions, log_prob)


# GSPN-GPT-FIXED: Equation 5 supervised pooling is a separate head, not another inference loop.
class SupervisedGraphReadout(GraphHead):
    def __init__(self, num_components, num_layers, num_graph_components, dim_target, pooling):
        super().__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        self.pool = _pooling(pooling)
        self.transition_logits = nn.Parameter(
            torch.rand(num_components * num_layers, num_graph_components)
        )
        self.classifier = nn.Linear(num_graph_components, dim_target, bias=False)

    def forward(self, node_posteriors, batch, targets=None):
        embeddings = node_posteriors.reshape(node_posteriors.shape[0], -1)
        local = embeddings @ torch.softmax(self.transition_logits, dim=1)
        pooled = self.pool(local, batch)
        pooled = torch.softmax(pooled, dim=1) if self.pooling == "sum" else pooled / self.num_layers
        return GraphResult(self.classifier(pooled))


# GSPN-GPT-FIXED: Resolve canonical classes directly, without importing retired implementations.
def _resolve_type(spec, base):
    implementation = s2c(spec) if isinstance(spec, str) else spec
    if not isinstance(implementation, type) or not issubclass(implementation, base):
        raise TypeError(f"{spec!r} must implement the {base.__name__} contract")
    return implementation


# GSPN-GPT-FIXED: Canonical implementation preserves the framework boundary and output positions.
class GSPN(ModelInterface):
    def __init__(self, dim_input_features, dim_target, config):
        if "add_self_loops" in config:
            raise ValueError("Remove add_self_loops: GSPN always ensures one self-loop")
        super().__init__(dim_input_features, dim_target, config)
        self.dim_node_features, self.dim_edge_features = graph_dimensions(dim_input_features)
        self.num_layers = config["num_layers"]
        self.num_components = config["num_mixtures"]
        if self.num_layers < 1 or self.num_components < 1:
            raise ValueError("num_layers and num_mixtures must be positive")
        self.use_shortcut = config.get("avg_parameters_across_layers", True)
        self.use_kmeans = config.get("init_kmeans", False)
        self.register_buffer("initialized", torch.tensor(False))
        categories = config.get("dim_categorical_features")
        if isinstance(categories, dict):
            categories = list(categories.values())
        emission_type = _resolve_type(config["emission_class"], GSPNEmission)
        transition_type = _resolve_type(
            config.get("convolution_class", "model.GSPNBaseConv"), GSPNBaseConv
        )
        self.emissions = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for layer in range(self.num_layers):
            self.emissions.append(
                emission_type.from_dimensions(
                    self.dim_node_features, self.num_components, categories
                )
            )
            self.transitions.append(transition_type(self.num_components, use_prior=layer == 0))
        if self.use_kmeans and not all(e.supports_kmeans for e in self.emissions):
            raise ValueError("K-means initialization requires Gaussian emissions")
        self.head = (
            _resolve_type(config["readout"], GraphHead)(dim_target, config)
            if config.get("readout")
            else None
        )

    def _prepare_evidence(self, data):
        x = data.x
        if x.shape[0] == 0:
            raise ValueError("GSPN requires at least one node")
        observed = getattr(data, "mask", None)
        edges, _ = remove_self_loops(data.edge_index)
        edges, _ = add_self_loops(edges, num_nodes=x.shape[0])
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return Evidence(
            x,
            observed_feature_mask(x, observed),
            observed is not None,
            edges,
            batch,
            getattr(data, "y", None),
        )

    def initialize_from_evidence(self, x, observed_mask=None):
        """Fit once on observed evidence in training mode; never initialize during evaluation."""
        if not self.training or not self.use_kmeans or self.initialized.item():
            return
        if x.shape[0] == 0:
            raise ValueError("GSPN requires at least one node")
        observed = observed_feature_mask(x, observed_mask)
        evidence = x.detach().clone().float().masked_fill(~observed, torch.nan)
        means = torch.nan_to_num(torch.nanmean(evidence, dim=0), nan=0.0)
        evidence = torch.where(torch.isnan(evidence), means[None, :], evidence)
        count = min(self.num_components, evidence.shape[0])
        centers = (
            KMeans(n_clusters=count, random_state=self.config.get("seed", 42), n_init=10)
            .fit(evidence.cpu().numpy())
            .cluster_centers_
        )
        centers = torch.as_tensor(centers, dtype=x.dtype, device=x.device)
        centers = centers[torch.arange(self.num_components, device=x.device) % count]
        max_variance = self.config.get("init_max_variance", 10.0)
        if max_variance <= 0:
            raise ValueError("init_max_variance must be positive")
        for emission in self.emissions:
            emission.initialize_from_centers(centers, max_variance)
        self.initialized.fill_(True)

    def _infer_layer(self, evidence, previous_posterior, layer_index, previous_parameters):
        emission = self.emissions[layer_index]
        prior = self.transitions[layer_index](
            previous_posterior, edge_index=evidence.edge_index, num_nodes=evidence.x.shape[0]
        )
        params = (
            emission.combine_shortcut_parameters(previous_parameters)
            if self.use_shortcut and layer_index == self.num_layers - 1 and layer_index > 0
            else emission.distribution_parameters()
        )
        components = emission.component_log_prob(
            evidence.x, params=params, observed_mask=evidence.observed_mask
        )
        log_prob, posterior = infer_mixture(components, prior)
        return LayerResult(prior, params, components, log_prob, posterior)

    def _infer_nodes(self, evidence):
        layers, parameters = [], []
        previous = None
        for index in range(self.num_layers):
            result = self._infer_layer(evidence, previous, index, parameters)
            layers.append(result)
            parameters.append(result.parameters)
            previous = result.posterior
        return tuple(layers)

    def _run_node_inference(self, data):
        evidence = self._prepare_evidence(data)
        self.initialize_from_evidence(evidence.x, evidence.observed_mask)
        layers = self._infer_nodes(evidence)
        return evidence, layers, torch.stack([layer.posterior for layer in layers], dim=1)

    def _evaluate_missing_features(self, evidence, final):
        missing = ~evidence.observed_mask
        has_missing = missing.reshape(evidence.x.shape[0], -1).any(1)
        known = (
            (evidence.observed_mask | self.emissions[-1].valid_values(evidence.x))
            .reshape(evidence.x.shape[0], -1)
            .all(1)
        )
        safe = torch.where(
            known.reshape((-1,) + (1,) * (evidence.x.ndim - 1)) & torch.isfinite(evidence.x),
            evidence.x,
            torch.zeros_like(evidence.x),
        )
        complete = self.emissions[-1].component_log_prob(safe, params=final.parameters)
        conditional = mixture_log_prob(complete, final.prior) - final.log_prob
        return torch.where(
            has_missing & ~known, torch.full_like(conditional, torch.nan), conditional
        )

    def infer(self, data, *, include_imputation=True, include_diagnostics=True):
        evidence, layers, posteriors = self._run_node_inference(data)
        final = layers[-1]
        imputation = (
            self.emissions[-1].predictive_mean(final.parameters, weights=final.posterior)
            if include_imputation
            else None
        )
        conditional = (
            self._evaluate_missing_features(evidence, final) if include_diagnostics else None
        )
        graph = (
            self.head(posteriors, evidence.batch, evidence.targets)
            if self.head is not None
            else None
        )
        return InferenceResult(evidence, layers, posteriors, imputation, conditional, graph)

    def encode(self, data):
        """Compute differentiable node embeddings without heads, imputation, or diagnostics."""
        evidence, _, posteriors = self._run_node_inference(data)
        return posteriors.reshape(evidence.x.shape[0], -1)

    def forward(self, data):
        return self.infer(data).to_reference_outputs()

    def reference_state_mapping(self):
        mapping = {"initialized": "initialized"}
        for name in ("emissions", "transitions"):
            for i, module in enumerate(getattr(self, name)):
                mapping.update(
                    {
                        f"{name}.{i}.{old}": f"{name}.{i}.{new}"
                        for old, new in module.reference_state_mapping().items()
                    }
                )
        if isinstance(self.head, SupervisedGraphReadout):
            mapping.update(
                {
                    "readout_node": "head.transition_logits",
                    "readout_graph.weight": "head.classifier.weight",
                }
            )
        elif self.head is not None:
            mapping.update(
                {
                    f"readout.{old}": f"head.{new}"
                    for old, new in self.head.reference_state_mapping().items()
                }
            )
        return mapping

    def load_reference_state_dict(self, state_dict: Mapping[str, Tensor]):
        """Validate the complete reference mapping before updating any model state."""
        mapping, destination = self.reference_state_mapping(), self.state_dict()
        missing = set(mapping) - set(state_dict)
        unexpected = set(state_dict) - set(mapping)
        if missing or unexpected:
            raise ValueError(
                f"Incompatible reference state: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        if set(mapping.values()) != set(destination) or len(set(mapping.values())) != len(mapping):
            raise ValueError("Reference mapping must cover every model state entry exactly once")
        converted = {}
        for old, new in mapping.items():
            value = state_dict[old]
            if not isinstance(value, Tensor) or value.shape != destination[new].shape:
                raise ValueError(f"Incompatible reference tensor shape for {old} -> {new}")
            converted[new] = value
        return self.load_state_dict(converted, strict=True)


# GSPN-GPT-FIXED: Supervised models compose the same inference core with the Eq. 5 head.
class SupGSPN(GSPN):
    def __init__(self, dim_input_features, dim_target, config):
        super().__init__(dim_input_features, dim_target, {**config, "readout": None})
        pooling = config.get("global_readout", "mean")
        _pooling(pooling)
        graph_components = config.get("num_graph_mixtures")
        if graph_components is not None:
            self.head = SupervisedGraphReadout(
                self.num_components, self.num_layers, graph_components, dim_target, pooling
            )
