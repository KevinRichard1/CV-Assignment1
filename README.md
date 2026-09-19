# Assignment 1
Repository for training a ConvNeXt on a small dataset

## Dependencies
```
pip install -r requirements.txt
```

## Training the model
```
python train.py \
    --data_dir /path/to/data \
    --ckpt_path /path/to/model.pth
```

## Evaluating the model
```
python evaluate.py \
    --data_dir /path/to/data \
    --ckpt_path /path/to/model.pth
```