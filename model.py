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

# GSPN-GPT-FIXED: MLWiz model contract and numerically stable NB inference (Sec. 4.1).
import torch
from mlwiz.model.interface import ModelInterface
from mlwiz.util import s2c
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


def graph_dimensions(dim_input_features):
    """MLWiz graph inputs carry node and edge widths as a pair."""
    return (
        tuple(dim_input_features)
        if isinstance(dim_input_features, (tuple, list))
        else (dim_input_features, 0)
    )


def exp_normalize_trick(m, dim):
    # GSPN-GPT-FIXED: softmax already performs stable normalization; clamping breaks sums.
    return torch.softmax(m, dim=dim)


def log_weights(weights):
    # GSPN-GPT-FIXED: Accept normalized probabilities, retaining exact zero support.
    weights = weights / weights.sum(-1, keepdim=True)
    return torch.where(
        weights > 0, weights.clamp_min(torch.finfo(weights.dtype).tiny).log(), -torch.inf
    )


def mixture_log_likelihood(components, weights):
    return torch.logsumexp(components + log_weights(weights), dim=-1)


def posterior(components, weights):
    # GSPN-GPT-FIXED: Bayes normalization in log space avoids density underflow (Eq. 3/4).
    return torch.softmax(components + log_weights(weights), dim=-1)


def feature_mask(mask, x):
    if mask is None:
        return torch.zeros_like(x, dtype=torch.bool)
    mask = mask.to(device=x.device, dtype=torch.bool)
    if mask.ndim == 1 and x.ndim == 2:
        mask = mask[:, None]
    return torch.broadcast_to(mask, x.shape)


class GSPNBaseConv(MessagePassing):
    def __init__(self, dim_edge_features, num_mixtures, num_hidden_neurons, use_prior):
        super().__init__(aggr="mean")
        self.dim_edge_features = dim_edge_features
        self.num_mixtures = num_mixtures
        self.num_hidden_neurons = num_hidden_neurons
        self.use_prior = use_prior
        shape = (num_mixtures,) if use_prior else (num_mixtures, num_mixtures)
        values = torch.rand(shape)
        self.transition_table = Parameter(values / values.sum(dim=0, keepdim=True))

    def forward(self, edge_index, edge_attr, h_v, batch_size):
        # Preserve the original convolution's self-neighbor convention.
        edge_index, _ = add_self_loops(edge_index, num_nodes=batch_size)
        table = exp_normalize_trick(self.transition_table, dim=0).unsqueeze(0)
        if self.use_prior:
            return table.expand(batch_size, -1)
        weighted = table * h_v.unsqueeze(1)
        return self.propagate(
            edge_index, x=weighted.reshape(batch_size, -1), size=(batch_size, batch_size)
        )


class GSPNEmission(nn.Module):
    def __init__(self, dim_observable, num_mixtures, num_hidden_neurons):
        super().__init__()
        self.dim_observable = dim_observable
        self.num_mixtures = num_mixtures
        self.num_hidden_neurons = num_hidden_neurons

    @staticmethod
    def average_parameters(parameters_per_layer):
        # GSPN-GPT-FIXED: Abstract operations must raise, never return exception objects.
        raise NotImplementedError("Use a concrete GSPNEmission subclass")

    @staticmethod
    def log_likelihood(x, mixture_weights, parameters, masked_nodes=None):
        raise NotImplementedError("Use a concrete GSPNEmission subclass")

    def forward(self, x, mixture_weights, masked_nodes=None):
        raise NotImplementedError("Use a concrete GSPNEmission subclass")


class GSPNCategoricalEmission(GSPNEmission):
    @staticmethod
    def average_parameters(parameters_per_layer):
        # GSPN-GPT-FIXED: Categorical shortcut remains on the simplex (Eq. 10).
        return torch.stack(parameters_per_layer).mean(0)

    @staticmethod
    def log_likelihood(x, mixture_weights, parameters, masked_nodes=None):
        # GSPN-GPT-FIXED: Marginalize a missing categorical variable (Sec. 4.2).
        if x.ndim == 2 and x.shape[1] > 1:
            mask = feature_mask(masked_nodes, x)
            if torch.any(mask.any(1) != mask.all(1)):
                raise ValueError("A one-hot categorical variable must be masked as a whole")
            missing = mask.all(1)
            labels = torch.where(mask, torch.zeros_like(x), x).argmax(1)
        else:
            labels = x.reshape(-1)
            missing = feature_mask(masked_nodes, x).reshape(-1)
            labels = torch.where(missing, torch.zeros_like(labels), labels)
        if torch.any(
            ~missing
            & (
                ~torch.isfinite(labels)
                | (labels != labels.floor())
                | (labels < 0)
                | (labels >= parameters.shape[-1])
            )
        ):
            raise ValueError("Observed categorical labels must be valid integer category IDs")
        labels = labels.long()
        params = parameters if parameters.ndim == 3 else parameters.unsqueeze(0)
        params = params.expand(labels.shape[0], -1, -1)
        comp = (
            params.gather(2, labels[:, None, None].expand(-1, params.shape[1], 1)).squeeze(2).log()
        )
        comp = torch.where(missing[:, None], torch.zeros_like(comp), comp)
        return mixture_log_likelihood(comp, mixture_weights), comp

    def __init__(self, dim_observable, num_mixtures, num_hidden_neurons):
        super().__init__(dim_observable, num_mixtures, num_hidden_neurons)
        self.num_categories = dim_observable
        self.categorical_probs = Parameter(torch.rand(num_mixtures, dim_observable))

    def forward(self, x, mixture_weights, masked_nodes=None):
        params = exp_normalize_trick(self.categorical_probs, 1).unsqueeze(0)
        ll, comp = self.log_likelihood(x, mixture_weights, params, masked_nodes)
        return params, ll, comp, self.impute(params, posterior(comp, mixture_weights))

    def impute(self, params, mixture_weights):
        return (params * mixture_weights.unsqueeze(2)).sum(1)


class GSPNMultiCategoricalEmission(GSPNEmission):
    def __init__(self, dim_observable, num_mixtures, num_hidden_neurons, dim_categorical_features):
        super().__init__(dim_observable, num_mixtures, num_hidden_neurons)
        self.dim_categorical_features = list(dim_categorical_features)
        if len(self.dim_categorical_features) != dim_observable:
            raise ValueError("One category count is required per input feature")
        self.emissions = nn.ModuleList(
            [
                GSPNCategoricalEmission(d, num_mixtures, num_hidden_neurons)
                for d in self.dim_categorical_features
            ]
        )

    @staticmethod
    def average_parameters(parameters_per_layer):
        return torch.stack(parameters_per_layer).mean(0)

    def log_likelihood(self, x, mixture_weights, parameters, masked_nodes=None):
        # GSPN-GPT-FIXED: Shared latent component: product first, mixture second (Sec. 4.1).
        missing = feature_mask(masked_nodes, x)
        components = mixture_weights.new_zeros(x.shape[0], self.num_mixtures)
        for i, (emission, params) in enumerate(
            zip(self.emissions, parameters.split(self.dim_categorical_features, dim=-1))
        ):
            _, comp = emission.log_likelihood(x[:, i], mixture_weights, params, missing[:, i])
            components = components + comp
        return mixture_log_likelihood(components, mixture_weights), components

    def forward(self, x, mixture_weights, masked_nodes=None):
        params = torch.cat(
            [exp_normalize_trick(e.categorical_probs, 1) for e in self.emissions], dim=-1
        )
        ll, comp = self.log_likelihood(x, mixture_weights, params, masked_nodes)
        return params, ll, comp, self.impute(params, posterior(comp, mixture_weights))

    def impute(self, params, mixture_weights):
        # GSPN-GPT-FIXED: Return one posterior predictive probability block per feature.
        return (
            mixture_weights @ params
            if params.ndim == 2
            else (params * mixture_weights[:, :, None]).sum(1)
        )


class GSPNGaussianEmission(GSPNEmission):
    @staticmethod
    def average_parameters(parameters_per_layer):
        # GSPN-GPT-FIXED: Distribution of the mean of independent Gaussians (Eq. 9).
        stack = torch.stack(parameters_per_layer)
        count = len(parameters_per_layer)
        return torch.stack(
            (stack[..., 0].mean(0), stack[..., 1].square().sum(0).sqrt() / count), dim=-1
        )

    @staticmethod
    def log_likelihood(x, mixture_weights, parameters, masked_nodes=None):
        # GSPN-GPT-FIXED: Mask before density evaluation; never mutate the caller's input.
        missing = feature_mask(masked_nodes, x)
        safe = torch.where(missing, torch.zeros_like(x), x)
        normal = torch.distributions.Normal(parameters[..., 0], parameters[..., 1])
        feature_ll = normal.log_prob(safe[:, None, :])
        comp = torch.where(missing[:, None, :], torch.zeros_like(feature_ll), feature_ll).sum(-1)
        return mixture_log_likelihood(comp, mixture_weights), comp

    def __init__(self, dim_observable, num_mixtures, num_hidden_neurons):
        super().__init__(dim_observable, num_mixtures, num_hidden_neurons)
        self.normal_params = Parameter(torch.rand(num_mixtures, dim_observable, 2))

    def initialize_means(self, cluster_centers):
        with torch.no_grad():
            self.normal_params[..., 0].copy_(cluster_centers)

    def forward(self, x, mixture_weights, masked_nodes=None):
        params = torch.stack(
            (
                self.normal_params[..., 0],
                torch.nn.functional.softplus(self.normal_params[..., 1]) + 1e-8,
            ),
            -1,
        ).unsqueeze(0)
        ll, comp = self.log_likelihood(x, mixture_weights, params, masked_nodes)
        return params, ll, comp, self.impute(params, posterior(comp, mixture_weights))

    def impute(self, params, mixture_weights):
        return (params[..., 0] * mixture_weights[:, :, None]).sum(1)


class GSPN(ModelInterface):
    def __init__(self, dim_input_features, dim_target, config):
        # GSPN-GPT-FIXED: Adopt MLWiz's model constructor without changing module paths.
        super().__init__(dim_input_features, dim_target, config)
        self.dim_node_features, self.dim_edge_features = graph_dimensions(dim_input_features)
        self.num_layers = config["num_layers"]
        self.num_mixtures = config["num_mixtures"]
        if self.num_layers < 1 or self.num_mixtures < 1:
            raise ValueError("num_layers and num_mixtures must be positive")
        self.num_hidden_neurons = config.get("num_hidden_neurons", 0)
        self.convolution_class = s2c(config.get("convolution_class", "model.GSPNBaseConv"))
        self.emission_class = s2c(config["emission_class"])
        self.avg_parameters_across_layers = config.get("avg_parameters_across_layers", True)
        self.use_kmeans = config.get("init_kmeans", False)
        self.add_self_loops = config.get("add_self_loops", False)
        self.register_buffer("initialized", torch.tensor(False))
        if self.use_kmeans and not issubclass(self.emission_class, GSPNGaussianEmission):
            raise ValueError("K-means initialization requires Gaussian emissions")
        categories = config.get("dim_categorical_features")
        if isinstance(categories, dict):
            categories = list(categories.values())
        self.emissions = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for layer in range(self.num_layers):
            args = (self.dim_node_features, self.num_mixtures, self.num_hidden_neurons)
            self.emissions.append(
                self.emission_class(*args, categories)
                if issubclass(self.emission_class, GSPNMultiCategoricalEmission)
                else self.emission_class(*args)
            )
            self.transitions.append(
                self.convolution_class(
                    self.dim_edge_features,
                    self.num_mixtures,
                    self.num_hidden_neurons,
                    use_prior=layer == 0,
                )
            )
        readout = config.get("readout")
        self.readout = (
            s2c(readout)(self.dim_node_features, self.dim_edge_features, dim_target, config)
            if readout
            else None
        )

    def _initialize(self, x, missing):
        # GSPN-GPT-FIXED: First training batch only; never use held-out missing ground truth.
        if not self.training or not self.use_kmeans or self.initialized.item():
            return
        evidence = x.detach().clone().float().masked_fill(missing, torch.nan)
        means = torch.nan_to_num(torch.nanmean(evidence, dim=0), nan=0.0)
        evidence = torch.where(torch.isnan(evidence), means[None, :], evidence)
        count = min(self.num_mixtures, evidence.shape[0])
        centers = (
            KMeans(n_clusters=count, random_state=self.config.get("seed", 42), n_init=10)
            .fit(evidence.cpu().numpy())
            .cluster_centers_
        )
        centers = torch.as_tensor(centers, dtype=x.dtype, device=x.device)
        centers = centers[torch.arange(self.num_mixtures, device=x.device) % count]
        max_variance = self.config.get("init_max_variance", 10.0)
        if max_variance <= 0:
            raise ValueError("init_max_variance must be positive")
        with torch.no_grad():
            for emission in self.emissions:
                emission.initialize_means(centers)
                scale = (
                    (torch.rand_like(emission.normal_params[..., 1]) * max_variance)
                    .clamp_min(1e-4)
                    .sqrt()
                )
                emission.normal_params[..., 1].copy_(scale + torch.log(-torch.expm1(-scale)))
            self.initialized.fill_(True)

    def forward(self, data):
        # GSPN-GPT-FIXED: Shared masked inference for supervised and unsupervised variants.
        x = data.x
        if x.shape[0] == 0:
            raise ValueError("GSPN requires at least one node")
        observed = getattr(data, "mask", None)
        missing = feature_mask(None if observed is None else ~observed.bool(), x)
        self._initialize(x, missing)
        edge_index = data.edge_index
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])
        embeddings, parameters = [], []
        h = None
        for layer, (transition, emission) in enumerate(zip(self.transitions, self.emissions)):
            weights = transition(edge_index, getattr(data, "edge_attr", None), h, x.shape[0])
            if layer:
                weights = weights.reshape(-1, self.num_mixtures, self.num_mixtures).sum(-1)
            params, ll, components, _ = emission(x, weights, missing)
            if layer == self.num_layers - 1 and layer and self.avg_parameters_across_layers:
                # Shortcut uses previously computed layers, excluding the top emission.
                params = emission.average_parameters(parameters)
                ll, components = emission.log_likelihood(x, weights, params, missing)
            parameters.append(params)
            h = posterior(components, weights)
            embeddings.append(h)
        imputed = emission.impute(params, h)
        # Conditional evaluation uses the same graph context and final mixture; never trains on hidden truth.
        has_missing = missing.reshape(x.shape[0], -1).any(1)
        known_features = torch.isfinite(x)
        if isinstance(emission, GSPNMultiCategoricalEmission):
            for i, categories in enumerate(emission.dim_categorical_features):
                known_features[:, i] &= (
                    (x[:, i] >= 0) & (x[:, i] < categories) & (x[:, i] == x[:, i].floor())
                )
        elif isinstance(emission, GSPNCategoricalEmission) and (x.ndim == 1 or x.shape[1] == 1):
            known_features &= (x >= 0) & (x < emission.num_categories) & (x == x.floor())
        known = (~missing | known_features).reshape(x.shape[0], -1).all(1)
        safe_complete = torch.where(
            known.reshape((-1,) + (1,) * (x.ndim - 1)) & torch.isfinite(x), x, torch.zeros_like(x)
        )
        complete_ll, _ = emission.log_likelihood(safe_complete, weights, params)
        conditional = torch.where(
            has_missing & ~known, torch.full_like(ll, torch.nan), complete_ll - ll
        )
        stacked = torch.stack(embeddings, dim=1)
        predictions = graph_ll = None
        if self.readout is not None:
            _, _, graph_ll, _, predictions = self.readout(stacked, data.batch, targets=data.y)
        extras = [
            ll,
            graph_ll,
            x,
            x,
            imputed,
            missing if observed is not None else None,
            ~missing if observed is not None else None,
            weights,
            params,
            conditional,
        ]
        return predictions, stacked.reshape(x.shape[0], -1), extras
