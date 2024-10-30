import numpy as np
import torch
import torch.nn.functional as F
import cv2
from run_blending_gradio import create_mask_from_points, cal_laplacian_loss, blending

img_1 = cv2.imread("01.jpg")
img_2 = cv2.imread("02.jpg")

poly_state = np.array([[800, 400],
                       [800, 800],
                       [1200, 400],
                       [1200, 800]])
polygon_state = {'points': poly_state, 'closed': True}

img = blending(img_1, img_2, 0, 0, polygon_state)

cv2.imwrite('image_0.png', img)
