# Attention-ocr Im2latex

## dataset

After downloading the dataset of im2latex, you should unzip it in `/dataset/`

link：https://pan.baidu.com/s/1volnYe6zMGYDz_iWuMc3VA 
pwd：gje5 

## How to use

1.  modify the path of dataset

3.  run train.py

### modify the path of dataset

in `/dataset/data_match.py` line 25: 

```python
split_road = "/home/hhhfccz/im2latex/dataset/" + split + "/"
```

should be modified by your own dataset_path

then, run `python data_match.py`

### run train.py

in `/attention_ocr.pytorch/config.py` line 27 and line 28

should be modified by your own annotations.json path

then, run `python train.py --cuda`

## Pretrained Model

Coming soon!
