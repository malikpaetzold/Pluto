# Pluto
# v0.1

import numpy as np
import cv2
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

IMG_SIZE = 256

use_tesseract = False
use_easyocr = False

tesseract_path = ""

def read_image(path):
    image = cv2.imread(path)
    if type(image) is None:
        raise Exception("Image path is not valid, read object is of type None!")
    return image

def show_image(image):
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()

def merge(aimg, bimg):
    # aimg.shape[0] < bimg.shape[0] and aimg.shape[1] < bimg.shape[1]
    if aimg.shape[2] == bimg[2]

def input_image(path):
    image = read_image(path)
    imshape = image.shape
    image_dimension = max(imshape[0], imshape[1])
    # The neural network input image size is 256x256.
    bg_dimension = 0
    if image_dimension < 256: bg_dimension = 256
    elif image_dimension < 512: bg_dimension = 512
    else: bg_dimension = 768
    bg_r = np.full((bg_dimension, bg_dimension), 22)
    bg_g = np.full((bg_dimension, bg_dimension), 32)
    bg_b = np.full((bg_dimension, bg_dimension), 42)
    background = cv2.merge((bg_r, bg_g, bg_b))
    out = cv2.add(image, background)
    show_image(out)

def ocr(image, override=False, function_use_tesseract=False, function_use_easyocr=False):
    if use_tesseract:
        from pytesseract import pytesseract
        pytesseract.tesseract_cmd = tesseract_path
        text = pytesseract.image_to_string(image)
        return text
    if use_easyocr:
        import easyocr
        reader = easyocr.Reader(['en'])
        return reader.readtext(image)
    
    return None

def sgmtn_header(image, reverse_output=False, reverse_only=False):
    import tensorflow as tf
    
    new_array = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3) 
    
    model = tf.keras.models.load_model("C:/.../.../...") # Path to model
    prediction = model.predict(image)[0] * 255
    
    output_mask_blur = cv2.blur(prediction, (10, 10))
    output_mask_rough = (output_mask_blur > 0.9).astype(np.uint8) * 255
    
    output = np.zeros((IMG_SIZE, IMG_SIZE, 3)).astype(np.uint8)
    reverse = output
    
    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            if output_mask_rough[i][j] > 0.5:
                for c in range(3):
                    output[i][j][c] = image[0][i][j][c]
                    reverse[i][j][c] = 0.0
            else:
                for c in range(3):
                    output[i][j][c] = 0.0
                    reverse[i][j][c] = image[0][i][j][c]
    
    if reverse_output:
        return output, reverse
    elif reverse_only:
        return reverse
    else:
        return output

def get_header(image, pre_processed=False):
    header_subtracted = image
    if not pre_processed:
        header_subtracted = sgmtn_header(image, False, True)
    
    print(ocr(header_subtracted))
    
    show_image(header_subtracted)

def get_text(image, pre_processed=False):
    header_subtracted = image
    if not pre_processed:
        header_subtracted = sgmtn_header(image, False, True)
    
    print(ocr(header_subtracted))
    
    show_image(header_subtracted)