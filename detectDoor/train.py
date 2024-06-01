import os
import sys
import time
import argparse
from datetime import datetime

import torch
import config as cfg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from model import FastRCNN
from dataset import get_dataloader
from utils import set_seed, write_config_log, write_result_log

def visualize_prediction(model, data_loader, device, threshold=0.7):
    
    model.eval()

    image_num = len(data_loader)
    # fig, axs = plt.subplots(1, image_num, figsize=(12, 8))
    # if image_num == 1:
    #     axs = [axs]
    
    for i, tup in enumerate(data_loader):
        images = tup[0]
        images = list(img.to(device) for img in images)
        # axs[i].imshow(image)
        # targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        # images, targets = next(iter(data_loader))
        # images = list(img.to(device) for img in images)

        with torch.no_grad():
            predictions = model(images)
        


        # Get the predictions
        prediction = predictions[0]
        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        # Filter out predictions below the threshold
        # high_score_indices = scores >= threshold
        high_score_indices = np.argmax(scores)
        box = boxes[high_score_indices]
        label = labels[high_score_indices]
        score = scores[high_score_indices]


        # for box, label, score in zip(boxes, labels, scores):
        #     xmin, ymin, xmax, ymax = box
        #     rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        #     ax.add_patch(rect)
        #     ax.text(xmin, ymin - 10, f'{label}: {score:.2f}', color='red', fontsize=12, backgroundcolor='white')
   

        # for box in prediction['boxes']:
        #     xmin, ymin, xmax, ymax = box
        #     rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, color='red', linewidth=2)
        #     ax.add_patch(rect)
        
        # visualize
        fig, ax = plt.subplots()
        image = images[0].cpu().numpy().transpose(1, 2, 0)
        ax.imshow(image)

        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin - 10, f'{label}: {score:.2f}', color='red', fontsize=12, backgroundcolor='white')
        ax.axis('off')
        plt.savefig(f"test_{i}.png")

    
    
    # for i, image in enumerate(images):
    #     image = image.cpu().numpy().transpose(1, 2, 0)
    #     axs[i].imshow(image)
        
    #     for box in predictions[i]['boxes']:
    #         xmin, ymin, xmax, ymax = box
    #         rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, color='red', linewidth=2)
    #         axs[i].add_patch(rect)
        
    #     axs[i].axis('off')
    
    # plt.show()
    # plt.savefig("test.png")


def train(model, train_loader, val_loader, logfile_dir, model_save_dir, optimizer, scheduler, device):
    for epoch in range(cfg.epochs):
        train_start_time = time.time()

        # train for one epoch
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            # print("wtf")

            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            # Forward pass 
            loss_dict = model(images, targets)
            # calculate loss
            losses = sum(loss for loss in loss_dict.values())
            
            # backprop
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # update loss
            train_loss += losses.item()
        print(losses)
            
        train_loss /= len(train_loader)
        print(f"trainloss: {train_loss}")
            
        # step onces
        scheduler.step()

        # evaluate on val
        val_loss = 0.0
        model.eval()
        for images, targets in val_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            
            with torch.no_grad():
                prediction = model(images,targets)
                # print(prediction)
                # prediction_losses = sum(loss for loss in prediction.values())
                # val_loss += prediction_losses.item()
        # val_loss /= len(val_loader)
        # print(f"valloss: {val_loss}")


    model_path = os.path.join(model_save_dir, 'model_best.pth')
    print(f"saving model to {model_path}!")
    torch.save(model.state_dict(), model_path)
    print("That's it!")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', help='dataset directory', type=str, default='../dataset/doorSamples/')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir

    # Experiment name
    exp_name = cfg.model_type + datetime.now().strftime('_%Y_%m_%d_%H_%M_%S') + '_' + cfg.exp_name

    # Write log file for config
    logfile_dir = os.path.join('./experiment', exp_name, 'log')
    os.makedirs(logfile_dir, exist_ok=True)
    write_config_log(os.path.join(logfile_dir, 'config_log.txt'))

    # Fix a random seed for reproducibility
    set_seed(9527)

    # Check if GPU is available, otherwise CPU is used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('Device:', device)

    model_save_dir = os.path.join('./experiment', exp_name, 'model')
    os.makedirs(model_save_dir, exist_ok=True)

    model = FastRCNN()
    model.to(device)

    train_loader = get_dataloader(os.path.join(dataset_dir, 'train'), batch_size=cfg.batch_size, split='train')
    val_loader   = get_dataloader(os.path.join(dataset_dir, 'val'), batch_size=cfg.batch_size, split='val')

    ##### LOSS & OPTIMIZER #####
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=1e-6)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    ##### TRAINING & VALIDATION #####
    ##### TODO: check train() in this file #####
    print("starting training...")
    train(model          = model,
          train_loader   = train_loader,
          val_loader     = val_loader,
          logfile_dir    = logfile_dir,
          model_save_dir = model_save_dir,
          optimizer      = optimizer,
          scheduler      = scheduler,
          device         = device)  
    
    visualize_prediction(model, val_loader, device)
    


if __name__ == "__main__":
    main()
