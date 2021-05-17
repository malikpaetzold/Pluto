# Pluto
# v0.1

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# from tensorflow.python.ops.gen_array_ops import reverse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

IMG_SIZE = 256

use_tesseract = False
use_easyocr = False

tesseract_path = ""

def read_image(path, no_BGR_correction=False):
    image = cv2.imread(path)
    if image is None:
        raise AttributeError("Pluto ERROR in read_image() function: Image path is not valid, read object is of type None!")
    if no_BGR_correction: return image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def show_image(image, BGR2RGB=False):
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

def mask_overlay(base, mask1, mask2=None):
    black_img = np.zeros([256, 256]).astype(np.uint8)
    og_img = model_input[0].astype(np.uint8)
    over_img = (output_mask_rough0).astype(np.uint8)
    cor_overlay_image = cv2.merge((over_img, black_img, black_img))
    out = cv2.addWeighted(og_img, 0.5, cor_overlay_image, 1.0, 0)
    over_img = (output_mask_rough1).astype(np.uint8)
    cor_overlay_image = cv2.merge((black_img, black_img, over_img))
    out = cv2.addWeighted(out, 0.5, cor_overlay_image, 1.0, 0)

    plt.imshow(out)
    plt.show()

def display_masks(mask1, mask1_pp=None, mask2=None, mask2_pp=None):
    try:
        cv2.imshow("pred", mask1)
    except Exception as e:
        print("Pluto ERROR - In display_masks(), origininal error message: ", e)
        quit()
    cv2.imshow("mask", mask1_pp)
    cv2.waitKey(0)
    cv2.imshow("mask2", mask2)
    cv2.imshow("mask", mask2_pp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def nyt_sgmt(input_image):
    import tensorflow as tf
    model_input = cv2.resize(input_image, (IMG_SIZE, IMG_SIZE)) # The model expects an image of size 256x256x3
    model_input = model_input.reshape(-1, IMG_SIZE, IMG_SIZE, 3) # Tensorflow requires the dimensions to be (1, 256, 256, 3)
    model0 = tf.keras.models.load_model("models/nyt_model_0") # load both models from the "models" folder
    model1 = tf.keras.models.load_model("models/nyt_model_1")

    prediction0 = model0.predict(model_input)[0] * 255  # The model returns its predictions in the dimension (1, 256, 256, 1) with values ranging from 0-1
    prediction1 = model1.predict(model_input)[0] * 255  # But the desired output is (256, 256, 1) with values ranging from 0-255

    output_mask_blur0 = cv2.blur(prediction0, (2, 2)) # Some post-processing on the segmentation mask
    output_mask_rough0 = (output_mask_blur0 > 0.999).astype(np.uint8) * 255

    output_mask_blur1 = cv2.blur(prediction1, (2, 2))
    output_mask_rough1 = (output_mask_blur1 > 0.999).astype(np.uint8) * 255
    
    # display_masks(None)
    
    # mask_overlay(None)
    
    return output_mask_rough0, output_mask_rough1

def mask_color(img, mask, reverse2=False):
    dimensions = img.shape
    output = np.zeros((dimensions[0], dimensions[1], 3)).astype(np.uint8)
    if reverse2: reverse = np.zeros((dimensions[0], dimensions[1], 3)).astype(np.uint8)
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            if mask[i][j] > 0.5:
                for c in range(3):
                    output[i][j][c] = img[i][j][c]
                    if reverse2: reverse[i][j][c] = 0.0
            else:
                for c in range(3):
                    output[i][j][c] = 0.0
                    if reverse2: reverse[i][j][c] = img[i][j][c]
    if reverse2: return output, reverse
    return output

def nyt_extract(img, title_mask, body_mask):
    img = cv2.resize(img, (512, 512))
    title_mask = cv2.resize(title_mask, (512, 512))
    body_mask = cv2.resize(body_mask, (512, 512))
    title_insight = mask_color(img, title_mask)
    body_insight = mask_color(img, body_mask)
    use_tesseract = True
    show_image(title_insight)
    title_raw_ocr = ocr(title_insight)
    show_image(body_insight)
    body_raw_ocr = ocr(body_insight)
    return title_raw_ocr, body_raw_ocr

def nyt(img):
    title_mask, body_mask = nyt_sgmt(img)
    print(nyt_extract(img, title_mask, body_mask))