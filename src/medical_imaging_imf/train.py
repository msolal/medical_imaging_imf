
import hydra
from clinicadl.transforms import Transforms
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from medical_imaging_imf.data.transforms import SimulateHypometabolic

from medical_imaging_imf.models.schrodinger_bridge import SchrodBridgeIMF
from medical_imaging_imf.trainer import DSBMJointTrainer

from clinicadl.data.datasets import CapsDataset, UnpairedDataset

from hydra.utils import instantiate


PATHOLOGY = "AD"
PERCENTAGE = 30

TENSOR_CONVERSION_NAME_CN = "pet_coregiso_cn"
TENSOR_CONVERSION_NAME_AD = "pet_coregiso_ad"


@hydra.main(version_base=None, config_path="configs", config_name="train")
def train(cfg: DictConfig):

    # TODO - create datasets in separate function
    
    preprocessing = instantiate(cfg.data.preprocessing)
    extraction = instantiate(cfg.data.extraction)
    image_transforms = instantiate(cfg.data.image_transforms)
    sample_transforms = instantiate(cfg.data.sample_transforms)

    transforms = Transforms(
        extraction=extraction,
        image_transforms=[image_transforms],
        sample_transforms=[sample_transforms],
    )

    # hypo_transforms = Transforms(
    #     extraction=cfg.data.extraction,
    #     image_transforms=[
    #         cfg.data.image_transforms,
    #         SimulateHypometabolic(f"{cfg.data.caps_dir}/masks", PATHOLOGY, PERCENTAGE),
    #     ],
    #     sample_transforms=cfg.data.sample_transforms,
    # )

    # TODO - depending on task
    # TODO - also add validation
    # TODO - add these to the config
    train_x0_data = f"{cfg.data.split_path}/micro-train-cn.tsv"
    train_x1_data = f"{cfg.data.split_path}/micro-train-ad.tsv"
    # val_x0_data = cfg.data.split_path / "micro-train-cn.tsv"
    # val_x1_data = cfg.data.split_path / "micro-train-ad.tsv"

    train_x0_dataset = CapsDataset(
        caps_directory=cfg.data.caps_dir,
        preprocessing=preprocessing,
        data=train_x0_data,
        transforms=transforms,
    )

    train_x1_dataset = CapsDataset(
        caps_directory=cfg.data.caps_dir,
        preprocessing=preprocessing,
        data=train_x1_data,
        transforms=transforms,
    )

    # val_x0_dataset = CapsDataset(
    #     caps_directory=cfg.data.caps_dir,
    #     preprocessing=preprocessing,
    #     data=val_x0_data,
    #     transforms=transforms,
    # )

    # val_x1_dataset = CapsDataset(
    #     caps_directory=cfg.data.caps_dir,
    #     preprocessing=preprocessing,
    #     data=val_x1_data,
    #     transforms=transforms,
    # )
    
    # train_x0_dataset.to_tensors(TENSOR_CONVERSION_NAME_CN, save_transforms=False)
    # train_x1_dataset.to_tensors(TENSOR_CONVERSION_NAME_AD, save_transforms=False)

    # TODO Tensor conversion (to do only once)
    train_x0_dataset.read_tensor_conversion(TENSOR_CONVERSION_NAME_CN)
    train_x1_dataset.read_tensor_conversion(TENSOR_CONVERSION_NAME_AD)
    # val_x0_dataset.read_tensor_conversion(TENSOR_CONVERSION_NAME_CN)
    # val_x1_dataset.read_tensor_conversion(TENSOR_CONVERSION_NAME_AD)
    
    train_dataset = UnpairedDataset([train_x0_dataset, train_x1_dataset])
    # val_dataset = UnpairedDataset([val_x0_dataset, val_x1_dataset])

    network_forward = instantiate(cfg.model.unet)
    network_backward = instantiate(cfg.model.unet)

    optimizer_forward_partial = instantiate(cfg.trainer.optimizer)
    optimizer_backward_partial = instantiate(cfg.trainer.optimizer)
    
    optimizer_forward = optimizer_forward_partial(network_forward.parameters())
    optimizer_backward = optimizer_backward_partial(network_backward.parameters())

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
        ema=ema,
        train_config=cfg.trainer.train_config,
    )

    print("Here!")
    
    partial_loader = instantiate(cfg.trainer.dataloader)
    train_loader = partial_loader(dataset=train_dataset)

    trainer.train(
        train_loader=train_loader
    )

    print("Here!")

if __name__ == "__main__":
    train()
