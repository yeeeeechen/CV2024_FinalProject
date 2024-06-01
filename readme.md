## Prerequisites:
1. Python version at least 3.9
2. `pip install -r requirements.txt`
3. Put training/testing videos in `dataset/videos/train` or `dataset/videos/test`


### Usage: 
How to find the bounding box of door from video.
1. **Run `python main.py`**

## Misc.
### Generate video with bounding boxes:
1. Save best model in `detectDoor/checkpoint/model_best.pth`
2. Go to path `dataset`
3. **Run `python inference.py`**
### Training: 
1. Put training data in `doorSamples/train` and validation data in `doorSamples/val` (should be annotated)
2. Go to path `dataset`
3. **Run `python train.py`**