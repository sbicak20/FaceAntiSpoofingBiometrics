from tqdm import tqdm
import os
import cv2

spoof_folder_path = r"C:\Users\sebas\Documents\FaceSpoofsDatasets\zips\CelebA_Spoof_train\Data\spoofedimages2"
live_folder_path = r"C:\Users\sebas\Documents\FaceSpoofsDatasets\zips\CelebA_Spoof_train\Data\liveimages2"
output_folder = r"C:\Users\sebas\Documents\FaceSpoofsDatasets\zips\CelebA_Spoof_train\Data\images2greyscale"

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

def SaveImages(images):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, image in enumerate(images):
        output_path = os.path.join(output_folder, f"output_image_{i}.png")
        cv2.imwrite(output_path, image)
