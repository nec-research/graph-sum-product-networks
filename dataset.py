"""
Graph-Induced Sum-Product Networks

Files: dataset.py

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

# MLWiz adapters preserve graph order and return (graph, target).
import hashlib
import json
from pathlib import Path

import torch
from mlwiz.data.dataset import DatasetInterface
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset


def transform_identity(value):
    """Serialize transform configuration recursively for processed-dataset cache identity."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [transform_identity(v) for v in value]
    if isinstance(value, dict):
        return {k: transform_identity(v) for k, v in value.items()}
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return {
        "class": type(value).__module__ + "." + type(value).__qualname__,
        "args": transform_identity(vars(value)),
    }


class GraphDataset(DatasetInterface):
    def __init__(
        self,
        storage_folder,
        name=None,
        root="DATA",
        raw_dataset_folder=None,
        transform_train=None,
        transform_eval=None,
        pre_transform=None,
        seed=42,
        **kwargs,
    ):
        self.source_name = name or type(self).__name__
        self.source_root = root
        self.options = kwargs
        self.seed = seed
        # Dataset-specific preprocessing identity avoids stale cache collisions.
        identity = {
            "name": self.source_name,
            "root": str(Path(root).resolve()),
            "raw": raw_dataset_folder,
            "options": kwargs,
            "seed": seed,
            "pre_transform": transform_identity(pre_transform),
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        # Preserve the CPU Torch RNG while precomputing stochastic transforms.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            super().__init__(
                str(Path(storage_folder) / digest),
                raw_dataset_folder,
                transform_train,
                transform_eval,
                pre_transform,
            )

    @staticmethod
    def _save_dataset(dataset, dataset_filepath):
        torch.save(dataset, dataset_filepath)

    @staticmethod
    def _load_dataset(dataset_filepath):
        """Load the trusted graph cache, which contains PyG objects as well as tensors."""
        return torch.load(dataset_filepath, weights_only=False)

    def __getitem__(self, index):
        """Return clones so transforms cannot mutate the processed cache."""
        graph, target = self.dataset[index]
        return graph.clone(), target.clone()

    @property
    def dim_input_features(self):
        graph = self.dataset[0][0]
        return (
            graph.x.shape[1],
            graph.edge_attr.shape[1]
            if graph.edge_attr is not None and graph.edge_attr.ndim > 1
            else 0,
        )

    @property
    def dim_target(self):
        return self._target_dimension()

    def _target_dimension(self):
        """Default to a scalar target; concrete adapters specify task-dependent widths."""
        return 1

    def _samples(self, graphs):
        """Clone graphs in source order and attach stable sample IDs and flattened targets."""
        result = []
        for index, source in enumerate(graphs):
            graph = source.clone()
            graph.sample_id = torch.tensor([index])
            target = graph.y.clone() if graph.y is not None else torch.zeros(1)
            result.append((graph, target.reshape(-1)))
        return result

    def process_dataset(self):
        raise NotImplementedError("Use a concrete graph dataset adapter")


class TUDatasetInterface(GraphDataset):
    def process_dataset(self):
        graphs = TUDataset(
            root=self.source_root,
            name=self.source_name,
            use_node_attr=self.options.get("use_node_attr", False),
        )
        return self._samples(graphs)

    def _target_dimension(self):
        target = self.dataset[0][1]
        if target.is_floating_point():
            return target.numel()
        return int(torch.cat([target for _, target in self.dataset]).max().item()) + 1


class TUDatasetInterfaceRegression(TUDatasetInterface):
    def _target_dimension(self):
        return self.dataset[0][1].numel()


class TUDatasetInterfaceMissingData(TUDatasetInterfaceRegression):
    def __init__(self, *args, **kwargs):
        kwargs["use_node_attr"] = True
        super().__init__(*args, **kwargs)


class OGBGDatasetInterface(GraphDataset):
    def process_dataset(self):
        from ogb.graphproppred import PygGraphPropPredDataset

        graphs = PygGraphPropPredDataset(name=self.source_name, root=self.source_root)
        self.official_split = {k: v.tolist() for k, v in graphs.get_idx_split().items()}
        (self.dataset_folder / "official_splits.json").write_text(json.dumps(self.official_split))
        return self._samples(graphs)

    def _target_dimension(self):
        return self.dataset[0][1].numel()


class OGBGmolpcbaFeatureMap(OGBGDatasetInterface):
    def process_dataset(self):
        samples = super().process_dataset()
        # Preserve the original category-ID mapping and graph order.
        values = torch.cat([g.x for g, _ in samples])
        vocabulary = [torch.unique(values[:, i], sorted=True) for i in range(values.shape[1])]
        # Persist the original IDs for consistent SMILES query encoding.
        (self.dataset_folder / "categorical_vocabulary.json").write_text(
            json.dumps([v.tolist() for v in vocabulary])
        )
        for graph, _ in samples:
            graph.x = torch.stack(
                [torch.searchsorted(vocabulary[i], graph.x[:, i]) for i in range(graph.x.shape[1])],
                dim=1,
            )
        return samples


class SyntheticDataset(GraphDataset):
    def process_dataset(self):
        raw = Path(self.options.get("raw_dir") or str(self._raw_dataset_folder or "GENERATED_DATA"))
        # Keep the original raw-file selection and sample ordering.
        files = [raw / 'data_list_100.pt']
        if not all(path.exists() for path in files):
            raise FileNotFoundError(
                f"No data_list_*.pt files in {raw}; generate raw data using the notebook"
            )
        graphs = [g for path in files for g in torch.load(path, weights_only=False)]
        # Node community labels are not graph targets; batch a dummy target.
        return [(graph, torch.zeros(1)) for graph, _ in self._samples(graphs)]

    def _target_dimension(self):
        return 0


class SmokeGraphDataset(GraphDataset):
    """Small deterministic graph fixture; smoke scores are not research results."""

    def process_dataset(self):
        # Test the real MLWiz CLI without downloading scientific datasets.
        generator = torch.Generator().manual_seed(self.seed)
        graphs = []
        for i in range(30):
            x = torch.randn(4, 2, generator=generator) + (i % 2)
            mask = torch.ones_like(x, dtype=torch.bool)
            mask[i % 4, i % 2] = False
            graphs.append(
                Data(
                    x=x,
                    edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
                    y=torch.tensor([i % 2]),
                    mask=mask,
                )
            )
        return self._samples(graphs)

    def _target_dimension(self):
        return 2
