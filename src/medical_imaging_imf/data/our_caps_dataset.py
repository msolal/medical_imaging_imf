import copy

from clinicadl.data.datasets import CapsDataset
from clinicadl.transforms.extraction import Sample


class OurCapsDataset(CapsDataset):
    def __init__(
        self, tensor_conversion_name: str, caps_directory, preprocessing, data, transforms, return_hypo: bool = False
    ):
        super().__init__(
            caps_directory,
            preprocessing,
            data,
            transforms,
        )

        self.read_tensor_conversion(tensor_conversion_name)

        self.return_hypo = return_hypo

        # TODO Caps Hypometabolic Dataset
        if self.return_hypo:
            # label_transforms = image_transforms without SimulateHypometabolic
            label_transforms = []
            for transform in transforms.image_transforms:
                if not str(transform).startswith("SimulateHypometabolic"):
                    label_transforms.append(transform)

        # TODO change into Compose
        self.label_transforms = label_transforms if self.return_hypo else None

    def __getitem__(self, idx: int) -> Sample:
        X = super().__getitem__(idx)

        if self.return_hypo:
            label = copy.deepcopy(X.sample)
            if self.label_transforms:
                label = self.label_transforms(label)

        # TODO
        data = {
            "participant_id": X.participant,
            "session_id": X.session,
            "slice_idx": X.slice_position,  # TODO check correctness
            "image": X.sample,
            "label": label if self.return_hypo else None,
        }

        # for compatibility with current code
        # TODO - use new clinicadl sample class
        return data
