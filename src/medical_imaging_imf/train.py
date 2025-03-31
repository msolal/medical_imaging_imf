from pathlib import Path

import hydra 
from omegaconf import DictConfig, OmegaConf

from medical_imaging_imf.trainer import DSBMJointTrainer 
from medical_imaging_imf.models.schrodinger_bridge import SchrodBridgeIMF

from torch.utils.data import DataLoader

from clinicadl.data.datatypes import PETLinear
from clinicadl.transforms import Transforms
from clinicadl.transforms.config import RescaleIntensityConfig, ResizeConfig
from clinicadl.transforms.extraction import Slice
from transforms import SimulateHypometabolic

from data.our_caps_dataset import OurCapsDataset

PATHOLOGY = "AD"
PERCENTAGE = 30

TENSOR_CONVERSION_NAME = ""


@hydra.main(version_base=None, config_path="configs", config_name="train")
def train(cfg: DictConfig):
    
    # TODO - create datasets in separate function
    
    transforms = Transforms(
        extraction=cfg.data.extraction,
        image_transforms=cfg.data.image_transforms,
        sample_transforms=cfg.data.sample_transforms,
    )
    
    hypo_transforms = Transforms(
        extraction=cfg.data.extraction,
        image_transforms=[
            cfg.data.image_transforms,
            SimulateHypometabolic(cfg.data.caps_dir, PATHOLOGY, PERCENTAGE),
        ],
        sample_transforms=cfg.data.sample_transforms,
    )
    
    # TODO - depending on task
    # TODO - also add validation
    # TODO - add these to the config
    train_x0_data = cfg.data.split_path / "train_cn.tsv"
    train_x1_data = cfg.data.split_path / "train_ad.tsv"
    val_x0_data = cfg.data.split_path / "validation_cn_baseline.tsv"
    val_x1_data = cfg.data.split_path / "validation_ad_baseline.tsv"
    
    train_x0_dataset = OurCapsDataset(
        tensor_conversion_name=TENSOR_CONVERSION_NAME,
        caps_directory=cfg.data.caps_dir,
        preprocessing=cfg.data.preprocessing,
        data=train_x0_data,
        transforms=transforms,
    )

    train_x1_dataset = OurCapsDataset(
        tensor_conversion_name=TENSOR_CONVERSION_NAME,
        caps_directory=cfg.data.caps_dir,
        preprocessing=cfg.data.preprocessing,
        data=train_x1_data,
        transforms=hypo_transforms,
        return_hypo=True,
    )

    val_x0_dataset = OurCapsDataset(
        tensor_conversion_name=TENSOR_CONVERSION_NAME,
        caps_directory=cfg.data.caps_dir,
        preprocessing=cfg.data.preprocessing,
        data=val_x0_data,
        transforms=transforms,
    )

    val_x1_dataset = OurCapsDataset(
        tensor_conversion_name=TENSOR_CONVERSION_NAME,
        caps_directory=cfg.data.caps_dir,
        preprocessing=cfg.data.preprocessing,
        data=val_x1_data,
        transforms=hypo_transforms,
        return_hypo=True,
    )
    
    # TODO Tensor conversion (to do only once)
    dataset.read_tensor_conversion(TENSOR_CONVERSION_NAME)
    dataset.read_tensor_conversion(TENSOR_CONVERSION_NAME)

    network_forward = None
    network_backward = None

    optimizer_forward = None
    optimizer_backward = None

    ema=None

    sb = SchrodBridgeIMF(
        network_forward=network_forward,
        network_backward=network_backward,
        sigma=cfg.model.sigma,
    )

    trainer = DSBMJointTrainer(
        model=sb,
        optimizer_forward=optimizer_forward,
        optimizer_backward=optimizer_backward,
        ema=None,
        train_config=cfg.trainer,
    )

    trainer.train(
        train_loader=DataLoader(
            # stack x0 and x1 datasets
        )
    )
    
if __name__ == "__main__":
    train()