"""Train the pretrained-encoder UNet-like segmentation model."""

from __future__ import annotations

from task1.PretrainedAutoencoder.train.common_segmentation import build_common_arg_parser, load_config_from_args, train_segmentation


def main() -> None:
    """Train the UNet-like segmentation model."""
    parser = build_common_arg_parser(
        "Train the pretrained-encoder UNet-like segmentation model."
    )
    args = parser.parse_args()
    config = load_config_from_args(args)
    train_segmentation(config)


if __name__ == "__main__":
    main()
