# TissueSegmentationandClassification-PUMA

Install dependencies once from the repository root:

```bash
cd <repo-root>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the commands below from the repository root.

**Task 1b – Pretrained Autoencoder**
Folder: `task1/PretrainedAutoencoder/`

This folder contains the full Task 1b workflow:
- tissue dataset inspection
- offline mask generation and verification
- convolutional autoencoder pretraining
- pretrained-encoder UNet-like segmentation training
- sliding-window segmentation evaluation

Expected task-local data and outputs:
- `task1/PretrainedAutoencoder/Dataset_Splits/`
- `task1/PretrainedAutoencoder/processed_masks/`
- `task1/PretrainedAutoencoder/outputs/`

Important configs:
- `task1/PretrainedAutoencoder/configs/autoencoder.yaml`
- `task1/PretrainedAutoencoder/configs/seg_unet_decoder.yaml`

These configs use task-local relative paths such as `Dataset_Splits`, `processed_masks`, and `outputs/...`. The Task 1 entrypoints resolve those paths relative to the Task 1 folder at runtime, so the repo can be copied anywhere as long as the task folder structure is preserved.

Preprocessing commands:
```bash
python -m task1.PretrainedAutoencoder.preprocessing.inspect_dataset \
  --config task1/PretrainedAutoencoder/configs/autoencoder.yaml

python -m task1.PretrainedAutoencoder.preprocessing.generate_tissue_masks \
  --config task1/PretrainedAutoencoder/configs/autoencoder.yaml

python -m task1.PretrainedAutoencoder.preprocessing.verify_masks \
  --config task1/PretrainedAutoencoder/configs/autoencoder.yaml
```

Training commands:
```bash
python -m task1.PretrainedAutoencoder.train.train_autoencoder \
  --config task1/PretrainedAutoencoder/configs/autoencoder.yaml

python -m task1.PretrainedAutoencoder.train.train_segmentation_unet_like \
  --config task1/PretrainedAutoencoder/configs/seg_unet_decoder.yaml
```

Evaluation command:
```bash
python -m task1.PretrainedAutoencoder.evaluate.evaluate_segmentation \
  --config task1/PretrainedAutoencoder/configs/seg_unet_decoder.yaml \
  --split test
```

Task 1 outputs are written under:
- `task1/PretrainedAutoencoder/outputs/autoencoder/`
- `task1/PretrainedAutoencoder/outputs/segmentation_unet/`
- `task1/PretrainedAutoencoder/outputs/comparison/`

Config notes:
- `model.pretrained_encoder_path` in the segmentation config is task-local and points to `outputs/autoencoder/checkpoints/best_encoder.pt` inside the Task 1 folder.
- Override paths from the CLI if you intentionally want to use data or outputs outside the default task-local layout.

**Task 2a – End-to-End Classifier**
Folder: `task2/ClassifierEndtoEnd/`

This folder contains the full Task 2a workflow:
- nuclei patch extraction from the Task 2 source splits
- end-to-end ConvNeXt-Tiny training
- validation and official test evaluation

Expected task-local data and outputs:
- `task2/ClassifierEndtoEnd/Dataset_Splits/`
- `task2/ClassifierEndtoEnd/Task2_Test_Set/`
- `task2/ClassifierEndtoEnd/task2_nuclei_patches/`
- `task2/ClassifierEndtoEnd/outputs/`

Important configs:
- `task2/ClassifierEndtoEnd/configs/task2a_convnext_tiny.yaml`
- `task2/ClassifierEndtoEnd/configs/task2a_convnext_tiny_tuned.yaml`

These configs use task-local relative paths such as `task2_nuclei_patches`, `Task2_Test_Set`, and `outputs/...`. The Task 2 entrypoints resolve those paths relative to the Task 2 folder at runtime, so the repo can be moved without editing the configs.

Dataset creation:
```bash
python -m task2.ClassifierEndtoEnd.preprocessing.create_nuclei_classification_dataset
```

Classifier training:
```bash
python -m task2.ClassifierEndtoEnd.train.train_nuclei_classifier \
  --config task2/ClassifierEndtoEnd/configs/task2a_convnext_tiny.yaml
```

Tuned classifier training:
```bash
python -m task2.ClassifierEndtoEnd.train.train_nuclei_classifier \
  --config task2/ClassifierEndtoEnd/configs/task2a_convnext_tiny_tuned.yaml
```

Validation evaluation:
```bash
python -m task2.ClassifierEndtoEnd.evaluate.evaluate_nuclei_classifier \
  --config task2/ClassifierEndtoEnd/configs/task2a_convnext_tiny.yaml \
  --split val
```

Official test evaluation:
```bash
python -m task2.ClassifierEndtoEnd.evaluate.evaluate_nuclei_classifier \
  --config task2/ClassifierEndtoEnd/configs/task2a_convnext_tiny.yaml \
  --split task2_test
```

Task 2 outputs are written under:
- `task2/ClassifierEndtoEnd/outputs/task2_convnext_tiny/`
- `task2/ClassifierEndtoEnd/outputs/task2_convnext_tiny_tuned/`

Config notes:
- Task 2 uses fixed label mapping `tumor=0`, `lymphocyte=1`, `histiocyte=2`.
- The dataset creation script writes extracted `100x100` patches into the task-local `task2_nuclei_patches/` folder.
- Training and evaluation resize patches to `224x224` inside the transform pipeline.
