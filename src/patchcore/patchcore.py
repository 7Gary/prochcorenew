"""PatchCore anomaly detection with contrastive memory augmentation."""
from __future__ import annotations

import logging
import os
import pickle
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import torch.utils.data

import patchcore.backbones
import patchcore.common
import patchcore.sampler
import patchcore.yvmm
from patchcore.networks.contrastive import (
    ContrastiveMemoryAugmentation,
    DomainInvariantContrastiveAdapter,
)

LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    """PatchCore anomaly detection class based on CMAM and DICA."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.forward_modules = torch.nn.ModuleDict()
        self.patch_maker: Optional[PatchMaker] = None
        self._nn_method: patchcore.common.FaissNN = patchcore.common.FaissNN(False, 4)
        self._anomaly_scorer_config: Dict[str, int] = {"n_nearest_neighbours": 1}
        self.anomaly_scorer: Optional[patchcore.common.NearestNeighbourScorer] = (
            patchcore.common.NearestNeighbourScorer(
                n_nearest_neighbours=self._anomaly_scorer_config["n_nearest_neighbours"],
                nn_method=self._nn_method,
            )
        )
        self.anomaly_segmentor: Optional[patchcore.common.RescaleSegmentor] = None
        self.featuresampler = patchcore.sampler.IdentitySampler()
        self.target_embed_dimension: Optional[int] = None
        self.training_statistics: Dict[str, float] = {}
        self._dica_mmd_trace: List[float] = []
        self.memory_module: Optional[ContrastiveMemoryAugmentation] = None
        self.dica: Optional[DomainInvariantContrastiveAdapter] = None
        self.yvmm_module = None

    def load(
        self,
        backbone,
        layers_to_extract_from: Iterable[str],
        device: torch.device,
        input_shape: Tuple[int, ...],
        pretrain_embed_dimension: int,
        target_embed_dimension: int,
        patchsize: int = 3,
        patchstride: int = 1,
        anomaly_score_num_nn: int = 1,
        featuresampler: patchcore.sampler.Sampler = patchcore.sampler.IdentitySampler(),
        nn_method: patchcore.common.FaissNN = patchcore.common.FaissNN(False, 4),
        **kwargs,
    ) -> None:
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = list(layers_to_extract_from)
        self.input_shape = input_shape
        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(target_dim=target_embed_dimension)
        _ = preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        adapter_depth = kwargs.pop("adapter_depth", 2)
        contrast_temperature = kwargs.pop("contrast_temperature", 0.07)
        adapter_heads = kwargs.pop("adapter_heads", 4)
        adapter_dropout = kwargs.pop("adapter_dropout", 0.0)
        self.dica = DomainInvariantContrastiveAdapter(
            target_embed_dimension,
            depth=adapter_depth,
            num_heads=adapter_heads,
            dropout=adapter_dropout,
            temperature=contrast_temperature,
            device=self.device,
        ).to(self.device)
        self.forward_modules["dica"] = self.dica

        self._nn_method = nn_method
        self._anomaly_scorer_config = {"n_nearest_neighbours": anomaly_score_num_nn}
        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=self._anomaly_scorer_config["n_nearest_neighbours"],
            nn_method=self._nn_method,
        )

        self.featuresampler = featuresampler
        self.memory_module = ContrastiveMemoryAugmentation(
            target_embed_dimension,
            temperature=contrast_temperature,
            device=self.device,
        ).to(self.device)

        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.training_statistics = {}
        self._dica_mmd_trace = []

        yvmm_config = kwargs.pop("yvmm_config", None)
        if yvmm_config is not None:
            if isinstance(yvmm_config, dict):
                config = patchcore.yvmm.YarnVoxelConfig(**yvmm_config)
            elif isinstance(yvmm_config, patchcore.yvmm.YarnVoxelConfig):
                config = yvmm_config
            else:
                raise TypeError("yvmm_config must be a dict or YarnVoxelConfig instance")
            self.yvmm_module = patchcore.yvmm.YarnVoxelManifoldMapping(
                self.target_embed_dimension, config
            ).to(self.device)
            self.forward_modules["yvmm_module"] = self.yvmm_module

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(
        self,
        images: torch.Tensor,
        detach: bool = True,
        provide_patch_shapes: bool = False,
    ):
        """Return feature embeddings for images."""

        def _detach(tensors: torch.Tensor):
            if detach:
                return [x.detach().cpu().numpy() for x in tensors]
            return tensors

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]
        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        if self.yvmm_module is not None:
            batchsize = images.shape[0]
            features = self.yvmm_module(features, patch_shapes[0], batchsize)

        adapted_features, mmd_value = self.dica(features)
        if mmd_value is not None:
            self._dica_mmd_trace.append(mmd_value)

        if provide_patch_shapes:
            return _detach(adapted_features), patch_shapes
        return _detach(adapted_features)

    def fit(self, training_data: torch.utils.data.DataLoader) -> None:
        """Compute embeddings of the training data and fill the memory bank."""

        self._fill_memory_bank(training_data)
        if self.yvmm_module is not None and hasattr(self.yvmm_module, "last_residual_norm"):
            try:
                residual = float(self.yvmm_module.last_residual_norm.item())
                fold_energy = float(self.yvmm_module.last_fold_energy.item())
                LOGGER.info(
                    "YVMM 诊断: 残差范数 %.4f, 折叠能量 %.4f. 可通过调整 --yvmm_residual_mix / --yvmm_fold_scale 优化。",
                    residual,
                    fold_energy,
                )
            except (RuntimeError, AttributeError):
                LOGGER.debug("Failed to read YVMM diagnostics after training.")

    def _fill_memory_bank(self, input_data: torch.utils.data.DataLoader) -> None:
        """Compute and set the support features for the contrastive memory."""

        _ = self.forward_modules.eval()

        def _image_to_features(input_image: torch.Tensor):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features_per_image: List[np.ndarray] = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for batch in data_iterator:
                batch_images = batch
                if isinstance(batch, dict):
                    batch_images = batch["image"]
                batch_size = batch_images.shape[0]
                for idx in range(batch_size):
                    single_image = batch_images[idx : idx + 1]
                    single_features = np.asarray(_image_to_features(single_image))
                    features_per_image.append(single_features)

        if not features_per_image:
            raise RuntimeError("No features were computed for memory bank training.")

        flattened = [feat.reshape(-1, feat.shape[-1]) for feat in features_per_image]
        concatenated = np.concatenate(flattened, axis=0)
        sampled = self.featuresampler.run(concatenated)

        all_features = torch.from_numpy(concatenated).to(self.device)
        self.dica.update_reference(all_features)

        torch_features = torch.from_numpy(sampled).to(self.device)
        prototypes = self.memory_module.fit(torch_features)
        if not isinstance(
            self.anomaly_scorer, patchcore.common.NearestNeighbourScorer
        ):
            self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
                n_nearest_neighbours=self._anomaly_scorer_config["n_nearest_neighbours"],
                nn_method=self._nn_method,
            )
        self.anomaly_scorer.fit(detection_features=[prototypes.detach().cpu().numpy()])
        self.dica.update_reference(prototypes.detach())

        stats = self.memory_module.export_stats()
        summary = {
            "memory_bank_size": float(stats.bank_size),
            "info_nce": float(stats.info_nce),
            "temperature": float(stats.temperature),
        }
        if self._dica_mmd_trace:
            summary["dica_mmd"] = float(np.mean(self._dica_mmd_trace))
        self.training_statistics = summary
        self._dica_mmd_trace = []

        LOGGER.info(
            "CMAM 记忆库已构建: 原始特征 %d 个, 对比记忆 %d 个, InfoNCE=%.4f.",
            concatenated.shape[0],
            int(stats.bank_size),
            float(stats.info_nce),
        )

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        image_names = []
        image_paths = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for batch in data_iterator:
                batch_names = None
                batch_paths = None
                if isinstance(batch, dict):
                    labels_gt.extend(batch["is_anomaly"].numpy().tolist())
                    masks_gt.extend(batch["mask"].numpy().tolist())
                    batch_names = batch.get("image_name")
                    batch_paths = batch.get("image_path")
                    images = batch["image"]
                else:
                    images = batch

                _scores, _masks = self._predict(images)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
                batchsize = len(_scores)

                def _extend_metadata(container, values):
                    if values is None:
                        container.extend([None] * batchsize)
                    elif isinstance(values, (list, tuple)):
                        container.extend(list(values))
                    else:
                        container.append(values)

                _extend_metadata(image_names, batch_names)
                _extend_metadata(image_paths, batch_paths)

        return scores, masks, labels_gt, masks_gt, image_names, image_paths

    def _predict(self, images):
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)

            patch_scores, _ = self._compute_contrastive_scores(features)
            image_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]

    def _compute_contrastive_scores(self, features: np.ndarray):
        patch_scores, distances, _ = self.anomaly_scorer.predict([features])
        return patch_scores, distances

    @staticmethod
    def _params_file(filepath: str, prepend: str = "") -> str:
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules["preprocessing"].output_dim,
            "target_embed_dimension": self.forward_modules["preadapt_aggregator"].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
            "adapter_depth": getattr(self.dica.encoder, "num_layers", 2),
            "contrast_temperature": self.memory_module.temperature,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN,
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


class PatchMaker:
    def __init__(self, patchsize: int, stride: Optional[int] = None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features: torch.Tensor, return_spatial_info: bool = False):
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x: np.ndarray, batchsize: int):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
