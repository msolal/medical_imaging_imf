from pathlib import Path

from clinicadl.data.datatypes import PETLinear
from clinicadl.transforms import Transforms
from clinicadl.transforms.config import RescaleIntensityConfig, ResizeConfig
from clinicadl.transforms.extraction import Slice
from transforms import SimulateHypometabolic

from data.our_caps_dataset import OurCapsDataset

CAPS_DIR = Path("")
SPLIT_PATH = Path("")
TENSOR_CONVERSION_NAME = "slice"

PATHOLOGY = "AD"
PERCENTAGE = 30

IMG_RES = 128
SLICE_RANGE = 10

preprocessing = PETLinear(reconstruction="coregiso", suvr_reference_region="cerebellumPons2")

# TODO - depending on task
# TODO - also add validation
train_source_data = SPLIT_PATH / "train_cn.tsv"
train_target_data = SPLIT_PATH / "train_ad.tsv"
val_source_data = SPLIT_PATH / "validation_cn_baseline.tsv"
val_target_data = SPLIT_PATH / "validation_ad_baseline.tsv"


transforms = Transforms(
    extraction=Slice(
        slices=list(range(IMG_RES // 2 - SLICE_RANGE, IMG_RES // 2 + SLICE_RANGE)), slice_direction=2
    ),  # extract slices
    image_transforms=[
        RescaleIntensityConfig(out_min_max=(-1, 1)),
    ],  # normalise images between -1 and 1
    sample_transforms=ResizeConfig((IMG_RES, IMG_RES)),  # resize images
)

hypo_transforms = Transforms(
    extraction=Slice(
        slices=list(range(IMG_RES // 2 - SLICE_RANGE, IMG_RES // 2 + SLICE_RANGE)), slice_direction=2
    ),  # extract slices
    image_transforms=[
        RescaleIntensityConfig(out_min_max=(-1, 1)),  # normalise images between -1 and 1
        SimulateHypometabolic(CAPS_DIR, PATHOLOGY, PERCENTAGE),
    ],
    sample_transforms=ResizeConfig((IMG_RES, IMG_RES)),  # resize images
)

train_source_dataset = OurCapsDataset(
    tensor_conversion_name=TENSOR_CONVERSION_NAME,
    caps_directory=CAPS_DIR,
    preprocessing=preprocessing,
    data=train_source_data,
    transforms=transforms,
)

train_target_dataset = OurCapsDataset(
    tensor_conversion_name=TENSOR_CONVERSION_NAME,
    caps_directory=CAPS_DIR,
    preprocessing=preprocessing,
    data=train_target_data,
    transforms=hypo_transforms,
    return_hypo=True,
)

val_source_dataset = OurCapsDataset(
    tensor_conversion_name=TENSOR_CONVERSION_NAME,
    caps_directory=CAPS_DIR,
    preprocessing=preprocessing,
    data=val_source_data,
    transforms=transforms,
)

val_target_dataset = OurCapsDataset(
    tensor_conversion_name=TENSOR_CONVERSION_NAME,
    caps_directory=CAPS_DIR,
    preprocessing=preprocessing,
    data=val_target_data,
    transforms=hypo_transforms,
    return_hypo=True,
)

# TODO Tensor conversion (to do only once)
train_source_dataset.read_tensor_conversion(TENSOR_CONVERSION_NAME)
train_target_dataset.to_tensors(TENSOR_CONVERSION_NAME)
