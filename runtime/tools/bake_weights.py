#!/usr/bin/env python3
"""Download torchvision backbone checkpoints into TORCH_HOME at image build time.

Spec 5.5: pretrained weights that libraries auto-download at model construction are baked
into the image (LeRobot ACT defaults to a pretrained ResNet18). torchvision verifies the
hash embedded in each checkpoint file name. Writes TORCH_HOME/baked-weights.json with the
sha256 of every file. Run with network access, at build time only.

  TORCH_HOME=/opt/rc/torch bake_weights.py [resnet18 resnet34 resnet50 ...]
"""
import hashlib
import json
import os
import sys
from pathlib import Path

DEFAULT = ["resnet18", "resnet34", "resnet50"]
WEIGHT_ENUMS = {"resnet18": "ResNet18_Weights", "resnet34": "ResNet34_Weights",
                "resnet50": "ResNet50_Weights"}


def main(names: list) -> int:
    import torchvision.models as tvm

    home = Path(os.environ["TORCH_HOME"])
    for name in names or DEFAULT:
        enum = getattr(tvm, WEIGHT_ENUMS[name])
        # Constructing the model with weights triggers the (hash-verified) download.
        getattr(tvm, name)(weights=enum.DEFAULT)
    ckpt = home / "hub" / "checkpoints"
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ckpt.glob("*")) if p.is_file()}
    if not files:
        print("no checkpoints downloaded", file=sys.stderr)
        return 1
    (home / "baked-weights.json").write_text(json.dumps(files, indent=2) + "\n")
    print(json.dumps(files, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
