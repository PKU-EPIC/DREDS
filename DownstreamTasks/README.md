# SwinDRNet for Downstream Tasks
PyTorch code and weights of SwinDRNet baseline for category-level pose estimation.
## Setup

Same as SwinDRNet

## Training

```bash
# An example command for training
$ python train.py --train_data_path PATH_DRED_CatKnown_TrainSplit --val_data_path PATH_DRED_CatKnown_ValSplit --val_obj_path PATH_DRED_CatKnown_CADMOEL
```

## Testing
```bash
# An example command for testing
$ python inference.py --val_data_type TYPE_OF_DATA --train_data_path PATH_DRED_CatKnown_TrainSplit --val_data_path PATH_DRED_CatKnown_TestSplit  --val_obj_path PATH_DRED_CatKnown_CADMOEL --val_depth_path PATH_VAL_DEPTH
```