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

def read_image(path, no_BGR_correction=False):
    image = cv2.imread(path)
    if type(image) is None:
        raise Exception("Image path is not valid, read object is of type None!")
    if no_BGR_correction: return image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def show_image(image, BGR2RGB=False):
    import matplotlib.pyplot as plt
    if BGR2RGB: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

def merge(aimg, bimg):
    # Merges two images, aimg being the foreground and bing being the background.
    # aimg.shape[0] < bimg.shape[0] and aimg.shape[1] < bimg.shape[1] and matching dimensions for the color channel(s)
    # Returns an image with the dimensions of bimg. Only works with RGB and grayscale images.
    color_channels = 1
    if len(aimg.shape) == len(bimg.shape):
        if aimg.shape[2] == 3: color_channels = 3
        else:
            print("Pluto WARNING - Input 1 shape: ", aimg.shape, " Input 2 shape: ", bimg.shape)
            raise Exception("Images must both be RGB or grayscale. Wrong ammount of color channels detected.")
    else:
        print("Pluto WARNING - Input 1 shape: ", aimg.shape, " Input 2 shape: ", bimg.shape)
        raise Exception("Images must have matching dimensions for the color channels.")
    out = np.zeros((bimg.shape[0], bimg.shape[1], color_channels), dtype=np.uint8)
    maxi = aimg.shape[0]
    maxj = aimg.shape[1]
    for i in range(bimg.shape[0]):
        for j in range(bimg.shape[1]):
            out[i][j] = bimg[i][j]
            if i < maxi and j < maxj:
                out[i][j] = aimg[i][j]
    return out

def input_image(path):
    image = read_image(path)
    imshape = image.shape
    image_dimension = max(imshape[0], imshape[1])
    # The neural network input image size is 256x256.
    bg_dimension = 0
    if image_dimension < 256: bg_dimension = 256
    elif image_dimension < 512: bg_dimension = 512
    else:
        bg_dimension = 768
        scale_factor = 768 / image_dimension
        image = cv2.resize(image, (int(imshape[1] * scale_factor), int(imshape[0] * scale_factor)))
    bg_r = np.full((bg_dimension, bg_dimension), 22)
    bg_g = np.full((bg_dimension, bg_dimension), 32)
    bg_b = np.full((bg_dimension, bg_dimension), 42)
    background = cv2.merge((bg_r, bg_g, bg_b))
    out = merge(image, background)
    return out

def ocr(image, override=False, function_use_tesseract=False, function_use_easyocr=False):
    if use_tesseract:
        if tesseract_path == "": print("Pluto WARNING - Please check if tesseract_path has been set.")
        from pytesseract import pytesseract
        pytesseract.tesseract_cmd = tesseract_path
        text = pytesseract.image_to_string(image)
        return text
    if use_easyocr:
        import easyocr
        reader = easyocr.Reader(['en'])
        return reader.readtext(image)
    
    print("Pluto WARNING - Check if use_tesseract and use_easyocr attributes are set.")
    return None

def expand_to_rows(image):
    dimensions = image.shape
    for i in range(int(dimensions[0] / 2)):
        for j in range(dimensions[1]):
            if image[i][j] > 200:
                image[i] = [255 for k in range(dimensions[1])]
    for i in range(int(dimensions[0] / 2), dimensions[0]):
        for j in range(dimensions[1]):
            image[i][j] = 0
    return image

def sgmtn_header(image, reverse_output=False, reverse_only=False):
    # Requires RGB image. Returns masked image in original dimensions.
    original_dimensions = image.shape
    
    try:
        import tensorflow as tf
        from tensorflow.compat.v1.keras.backend import set_session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        sess = tf.compat.v1.Session(config=config)
        set_session(sess)
    except ModuleNotFoundError as e:
        print("Please make shure to install Tensorflow dependency. Original error: ", e)
    
    new_array = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image_reshape = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3) 
    
    model = tf.keras.models.load_model("D:/Codeing/Twitter_Segmentation/model_2")
    prediction = model.predict(image_reshape)[0] * 255
    
    output_mask_blur = cv2.blur(prediction, (10, 10))
    output_mask_rough = (output_mask_blur > 0.9).astype(np.uint8) * 255
    output_mask_upscale = cv2.resize(output_mask_blur, (original_dimensions[0], original_dimensions[1]))
    output_mask = expand_to_rows(output_mask_upscale)
    show_image(output_mask)
    output = np.zeros((original_dimensions[0], original_dimensions[1], 3)).astype(np.uint8)
    reverse = np.zeros((original_dimensions[0], original_dimensions[1], 3)).astype(np.uint8)
    
    for i in range(original_dimensions[0]):
        for j in range(original_dimensions[1]):
            if output_mask_upscale[i][j] > 0.5:
                for c in range(3):
                    output[i][j][c] = image[i][j][c]
                    reverse[i][j][c] = 0.0
            else:
                for c in range(3):
                    output[i][j][c] = 0.0
                    reverse[i][j][c] = image[i][j][c]
    
    if reverse_output:
        return output, reverse
    elif reverse_only:
        return reverse
    else:
        return output

def sgmtn_stats(image, reverse_output=False, reverse_only=False):
    # Requires RGB image. Returns masked image in original dimensions.
    original_dimensions = image.shape
    
    try:
        import tensorflow as tf
    except ModuleNotFoundError as e:
        print("Please make shure to install Tensorflow dependency. Original error: ", e)
    
    new_array = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image_reshape = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3) 
    
    model = tf.keras.models.load_model("D:/Codeing/Twitter_Segmentation/model_1")
    prediction = model.predict(image_reshape)[0] * 255
    
    output_mask_blur = cv2.blur(prediction, (10, 10))
    output_mask_rough = (output_mask_blur > 0.9).astype(np.uint8) * 255
    output_mask_upscale = cv2.resize(output_mask_blur, (original_dimensions[0], original_dimensions[1]))
    
    output = np.zeros((original_dimensions[0], original_dimensions[1], 3)).astype(np.uint8)
    reverse = np.zeros((original_dimensions[0], original_dimensions[1], 3)).astype(np.uint8)
    
    for i in range(original_dimensions[0]):
        for j in range(original_dimensions[1]):
            if output_mask_upscale[i][j] > 0.5:
                for c in range(3):
                    output[i][j][c] = image[i][j][c]
                    reverse[i][j][c] = 0.0
            else:
                for c in range(3):
                    output[i][j][c] = 0.0
                    reverse[i][j][c] = image[i][j][c]
    
    if reverse_output:
        return output, reverse
    elif reverse_only:
        return reverse
    else:
        return output

def get_header(image, already_segmented=False):
    text_subtracted = image
    if not already_segmented:
        text_subtracted = sgmtn_header(image)
    
    print(ocr(text_subtracted))
    
    show_image(text_subtracted)

def get_text(image, already_segmented=False):
    header_subtracted = image
    if not already_segmented:
        header_subtracted = sgmtn_header(image, False, True)
    show_image(header_subtracted)
    ocr_result = ocr(header_subtracted)
    lines = ocr_result.split("\n")
    print(lines)
    # show_image(header_subtracted)

def get_stats(image, already_segmented=False):
    stats_subtracted = image
    if not already_segmented:
        stats_subtracted = sgmtn_stats(image)
    show_image(stats_subtracted)
    ocr_result = ocr(stats_subtracted)
    lines = ocr_result.split("\n")
    print(lines)
    show_image(stats_subtracted)