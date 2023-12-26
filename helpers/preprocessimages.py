import cv2
from PIL import Image
import numpy as np


#image_path as parameter
def PreprocessImage(image_path, target_size=(64,64)):
        # Open the image
        original_image = Image.open(image_path)

        # Convert the image to grayscale if it's not already
        if original_image.mode != 'L':
            original_image = original_image.convert('L')

        # Resize the image to the target size
        resized_image = original_image.resize(target_size)

        # Convert the image to a NumPy array
        image_array = np.array(resized_image)

        # Normalize pixel values to the range [0, 1]
        normalized_image = image_array / 255.0

        # You can perform additional preprocessing steps here if needed

        return normalized_image

def ConvertToGreyscale(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

# Example usage:
#image_path = "path/to/your/image.jpg"
#preprocessed_image = preprocess_image(image_path)
