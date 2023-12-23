import cv2
import numpy as np


#image_path as parameter
def PreprocessImage(image, target_size=(224, 224)):
    # Load the image using OpenCV
    #image = cv2.imread(image_path)

    # grayscale
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to the target size while maintaining the aspect ratio
    height, width, _ = image.shape
    aspect_ratio = width / height

    if aspect_ratio > 1:  # landscape orientation
        new_width = int(target_size[0] * aspect_ratio)
        resized_image = cv2.resize(image, (new_width, target_size[1]))
    else:  # portrait orientation
        new_height = int(target_size[1] / aspect_ratio)
        resized_image = cv2.resize(image, (target_size[0], new_height))

    # Crop the center of the resized image to match the target size
    crop_x = max(0, int((resized_image.shape[1] - target_size[0]) / 2))
    crop_y = max(0, int((resized_image.shape[0] - target_size[1]) / 2))
    cropped_image = resized_image[crop_y:crop_y + target_size[1], crop_x:crop_x + target_size[0]]

    # Normalize pixel values to be in the range [0, 1]
    normalized_image = cropped_image.astype("float32") / 255.0

    # Expand dimensions to create a batch (if needed)
    preprocessed_image = np.expand_dims(normalized_image, axis=0)

    return preprocessed_image

def ConvertToGreyscale(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

# Example usage:
#image_path = "path/to/your/image.jpg"
#preprocessed_image = preprocess_image(image_path)
