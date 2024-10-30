import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cityscapes_datasets import FacadesDataset
from Unet import FullyConvNetwork
from torch.optim.lr_scheduler import StepLR

def tensor_to_image(tensor):
    """
    Convert a pytorch tensor to a numpy array suitable for opencv.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to Numpy array
    image = tensor.cpu().detach().numpy()

    image = np.transpose(image, (1, 2, 0))

    image = (image + 1) / 2
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Curent epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatentate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, num_epochs):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (Dataloader): Dataloader for the training data.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.train()
    running_loss = 0.0

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        optimizer.zero_grad()

        outputs = model(image_rgb)

        if epoch % 10 == 0 and i == 0:
            save_images(image_rgb, image_semantic, outputs, 'train_results_1', epoch)

        loss = criterion(outputs, image_semantic)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')
        with open("test_loss_1.txt", "a") as f:
            f.write(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f} \n')

def validate(model, dataloader, criterion, device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (Dataoader): Dataloader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            outputs = model(image_rgb)

            loss = criterion(outputs, image_semantic)
            val_loss += loss.item()

            if epoch % 5 == 0 and i == 0:
                save_images(image_rgb, image_semantic, outputs, 'val_results_1', epoch)

    avg_val_loss = val_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')
    with open("val_loss_1.txt", "a") as f:
        f.write(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f} \n')

def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_dataset = FacadesDataset(list_file='train_list_1.txt')
    val_dataset = FacadesDataset(list_file='val_list_1.txt')

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=8)

    model = FullyConvNetwork().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))

    scheduler = StepLR(optimizer, step_size=200, gamma=0.2)

    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs)
        validate(model, val_loader, criterion, device, epoch, num_epochs)

        scheduler.step()

        if (epoch + 1) % 20 == 0:
            os.makedirs('checkpoints_1', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints_1/pix2pix_model_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
