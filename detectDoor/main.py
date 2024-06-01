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

def guess_door_location(video_filename,threshold=0.9):
    print(f"guessing door location from {video_filename}...")
    # Check if GPU is available, otherwise CPU is used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # load rcnn model
    model = FastRCNN()
    model.load_state_dict(torch.load('./checkpoint/model_best.pth', map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    # define transform
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float, scale=True),
    ])

    # Open the video file
    cap = cv2.VideoCapture(video_filename)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Initialize the frame count and valid guesses
    frame_count = 0 
    good_bboxes = np.empty((6,4))
    good_guesses = 0
    good_guesses_stop = 6
    while good_guesses < good_guesses_stop:
        ret, frame = cap.read()
        if not ret:
            print('failed')
            break

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
        if np.max(scores) > threshold:
            high_score_index = np.argmax(scores)
            good_box = boxes[high_score_index]
            # good_bboxes.append(good_box)
            good_bboxes[good_guesses,:] = good_box
            good_guesses += 1

    # find the max bbox of all good bboxes, expand contour by a few pixels
    minx = np.min(good_bboxes[:,0])
    miny = np.min(good_bboxes[:,1])
    maxx = np.max(good_bboxes[:,2])
    maxy = np.max(good_bboxes[:,3])
    final_box = np.array([minx,miny,maxx,maxy])
    print(final_box)
    return final_box

    

def main():
    video_path = "../dataset/videos/test/01.mp4"
    
    bbox = guess_door_location(video_path)
    # directory = "./"  # Specify the directory to scan
    # output_filename = "algorithm_output.json"  # Output JSON file name
    # videos_info = scan_videos(directory)
    # generate_json(output_filename, videos_info)
    # print(f"Generated JSON file '{output_filename}' with video annotations.")

if __name__ == "__main__":
    main()
