import numpy as np
from tqdm import tqdm
import os
import cv2
from skimage.io import imread, imsave

spoof_folder_path = r"C:\Users\sebas\Documents\FaceSpoofsDatasets\zips\CelebA_Spoof_train\Data\spoofedimages2"
live_folder_path = r"C:\Users\sebas\Documents\FaceSpoofsDatasets\zips\CelebA_Spoof_train\Data\liveimages2"
output_folder = r"C:\Users\sebas\Documents\FaceSpoofsDatasets\zips\CelebA_Spoof_train\Data\realdata"

# List to store image paths and labels
spoof_image_paths = []
live_image_paths = []

def GetSpoofImagePaths():
    for filename in tqdm(os.listdir(spoof_folder_path), desc="Loading spoof images", unit="image"):
        if filename.endswith(".png"):
            spoof_image_path = os.path.join(spoof_folder_path, filename)
            spoof_image_paths.append(spoof_image_path)
    return spoof_image_paths

def GetLiveImagePaths():
    for filename in tqdm(os.listdir(live_folder_path), desc="Loading live images", unit="image"):
        if filename.endswith(".png"):
            live_image_path = os.path.join(live_folder_path, filename)
            live_image_paths.append(live_image_path)
    return live_image_paths

def SaveImages(images, output_folder ,folder_name):
    rOutputFolder = output_folder + f"\{folder_name}"
    if not os.path.exists(rOutputFolder):
        os.makedirs(rOutputFolder)
    for i, image in enumerate(images):
        if image is not None:
            output_path = os.path.join(rOutputFolder, f"output_image_{i}.png")
            cv2.imwrite(output_path, image)


def SaveImagesScikit(images, output_folder, folder_name):
    r_output_folder = os.path.join(output_folder, folder_name)
    if not os.path.exists(r_output_folder):
        os.makedirs(r_output_folder)

    for i, image in enumerate(images):
        output_path = os.path.join(r_output_folder, f"output_image_{i}.png")
        imsave(output_path, image)
