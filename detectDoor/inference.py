import os
import sys
import time
import argparse

import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torchvision.transforms import v2 as transforms

from model import FastRCNN
from tqdm import tqdm
from dataset import get_dataloader

def video_inference(VIDEO_PATH, OUTPUT_PATH, model, device, threshold=0.85):
    print(f"input: {VIDEO_PATH}")
    print(f"write to: {OUTPUT_PATH}\n")

    model.eval()
    
    # get video data
    video = cv2.VideoCapture(VIDEO_PATH)
    film_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    film_h, film_w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    film_fps = video.get(cv2.CAP_PROP_FPS)
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowriter = cv2.VideoWriter(OUTPUT_PATH, fourcc, film_fps, (film_w, film_h))

    # define transform
    transform = transforms.Compose([
        transforms.ToImage(),
        # transforms.Grayscale(),
        transforms.ToDtype(torch.float, scale=True),
        # transforms.ToPureTensor()
    ])

    pbar = tqdm(total = film_frame_count)
    while (video.isOpened()):
        ret, frame = video.read()
        if ret:  ## check whethere the frame is legal, i.e., there still exists a frame

            # convert to torch tensor
            color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_tensor = transform(color_converted)

            images = [image_tensor.to(device)]
            with torch.no_grad():
                pred = model(images)
                pred = pred[0]

            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()

            # get highest rating box
            # t = np.max(scores)
            t = threshold
            high_score_indices = (scores >= t)
            good_box = boxes[high_score_indices]
            good_score = scores[high_score_indices]


            for box, score in zip(good_box, good_score):
                # draw using opencv
                box_int = box.astype(np.uint32)
                frame = cv2.rectangle(img=frame, rec=box_int, color=(0,0,255), thickness=2)
                origin = (box_int[0], box_int[3])
                frame = cv2.putText(frame, text=f'{score:.2f}', org=origin, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,0,255),thickness=3)

            # draw to video
            videowriter.write(frame)
            pbar.update(1)
        else:
            break

    pbar.close()
    video.release()
    videowriter.release()
    cv2.destroyAllWindows()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='model path', type=str, default='./checkpoint/model_best.pth')
    parser.add_argument('--video_datadir', help='test video directory', type=str, default='../dataset/videos')
    parser.add_argument('--output_dir', help='output video path', type=str, default='./output/')
    args = parser.parse_args()

    video_datadir = args.video_datadir
    output_dir = args.output_dir
    model_path = args.model_path

    # Check if GPU is available, otherwise CPU is used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    model = FastRCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)

    inputList = ['train/01.mp4', 'train/02.mp4', 'train/03.mp4', 'test/01.mp4', 'test/03.mp4', 'test/05.mp4', 'test/07.mp4', 'test/09.mp4', 'test/03_flipped.mov','test/09_flipped.mov', 'test/05_rotated.mov']
    outputList = ['train_01.mp4', 'train_02.mp4', 'train_03.mp4', 'test_01.mp4', 'test_03.mp4', 'test_05.mp4', 'test_07.mp4', 'test_09.mp4', 'test_03_flipped.mp4', 'test_09_flipped.mp4', 'test_05_rotated.mp4']
    for input_path, output_path in zip(inputList, outputList):
        video_inference(os.path.join(video_datadir, input_path), os.path.join(output_dir, output_path), model, device, threshold=0.8)


    
if __name__ == '__main__':
    main()
