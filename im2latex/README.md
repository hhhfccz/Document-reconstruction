# Attention-ocr Im2latex

## dataset

After downloading the dataset of im2latex, you should unzip it in `/dataset/`


## requirements.txt

> torch 1.6.0
> 
> opencv-python
> 
> numba
> 
> nltk

## How to use

1.  modify the path of dataset

2.  run train.py

### modify the path of dataset

in `/dataset/data_match.py` line 25: 

```python
split_road = "/home/hhhfccz/im2latex/dataset/im2latex-100k/" + split + "/"
```

should be modified by your own dataset path

then, run `python data_match.py`

### run train.py

in `/attention_ocr.pytorch/utils.py` line 30

```python
dataset_path = "/home/hhhfccz/im2latex/dataset/im2latex-100k/"
```

should be modified by your own dataset path

then, run `python train.py`

## Pretrained Model

Coming soon!
