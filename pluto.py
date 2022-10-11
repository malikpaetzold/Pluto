# Pluto
# v1.0.0

# MIT License
# Copyright (c) 2022 Malik Pätzold

from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
import torch.nn.functional as F
import torch.hub as hub

import time
import webbrowser

import easyocr
reader = easyocr.Reader(['en'])

# For reproducibility
seed = 3
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# cli capabilities
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Runs Pluto on screenshots.")
    parser.add_argument("-i", "--input", type=str, metavar="", help="Path to input image. If left empty, the clipboard content will be used automatically")
    parser.add_argument("-o", "--output", type=str, metavar="", help="Path to where the output file should be saved.")
    parser.add_argument("-c", "--category", type=str, metavar="", help="Category of media. Equal to class name")
    args = parser.parse_args()

    arg_i = args.input
    arg_o = args.output
    arg_c = args.category


def read_image(path: str, no_BGR_correction=False, resz=None):  # -> np.ndarray
    """Returns an image from a path as a numpy array, resizes it if necessary
    
    Args:
        path: location of the image.
        no_BGR_correction: When True, the color space is not converted from BGR to RGB
    
    Returns:
        The read image as np.ndarray.
    
    Raises:
        AttributeError: if path is not valid, this causes image to be None
    """
    if type(path) == np.ndarray: return path
    image = cv2.imread(path)
    if resz is not None: image = cv2.resize(image, resz)
    if image is None: raise AttributeError("Pluto ERROR in read_image() function: Image path is not valid, read object is of type None!")
    if no_BGR_correction: return image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def show_image(img, BGR2RGB=False, specify_tkinter_backend=False, use_cv2=False):
    """Displays an image using Matplotlib's pyplot.imshow()
    
    Args:
        image: The image to be displayed.
        BGR2RGB: When True, the color space is converted from BGR to RGB.
        specify_tkinter_backen: may solve matplotlib related backend problems when set to True
    """
    if specify_tkinter_backend:
        import tkinter
        import matplotlib
        matplotlib.use('TkAgg')
    
    if BGR2RGB: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if use_cv2:
        cv2.imshow("Pluto show_image", img)
        cv2.waitKey()
        return
    
    plt.imshow(img)
    plt.show()

def write_image(path, img):
    """wrapper for matplotlib's imsave method
    """
    plt.imsave(path, img)

def grab_clipboard():
    from PIL import ImageGrab
    img = ImageGrab.grabclipboard().convert("RGB")
    img = np.array(img)
    return img

def avg_of_row(img: np.ndarray, row: int, ovo=False):  # -> int | float
    """Calculates the average pixel value for one row of an image
    
    Args:
        img: The screenshot as np.ndarray
        row: which row of the image should be analysed?
        ovo: output as 'one value only' instead of list?
    
    Returns:
        The average value per row, ether one value only or per color channel value
    """
    all_values_added = 0
    if img.shape[2] == 3: all_values_added = [0, 0, 0]
    length = len(img)
    for pixl in img[row]: all_values_added += pixl
    out = all_values_added / length
    if ovo: out = sum(out) / 3
    return out

def avg_of_collum(img: np.ndarray, collum: int, ovo=False):  # -> int | float
    """Calculates the average pixel value for one collum of an image
    
    Args:
        img: The screenshot as np.ndarray
        collum: which collum of the image should be analysed?
        ovo: output as 'one value only' instead of list?
    
    Returns:
        The average value per collum, ether one value only or per color channel value
    """
    all_values_added = 0
    if img.shape[2] == 3: all_values_added = [0, 0, 0]
    length = len(img[0])
    for pixl in img[:, collum]: all_values_added += pixl
    out = all_values_added / length
    if ovo: out = sum(out) / 3
    return out

def trimm_and_blur(inpt: np.ndarray, less: bool, value: int, blurs, trimm, double_down=False, invert=None, remove_color=False, remove_value=np.ndarray([0,0,0])):
    """Isolates parts of an image with a specific color or color range. Also capable of removing color and bluring the output.
    
    Args:
        inpt: The input image as np.ndarray (must have 3 color channels)
        less: Bigger / smaller trimming method
        value: Threshold for color values
        blurs: blurring kernel size
        trimm: pixel value for trimmed areas
        double_down: If True, the non-isolated areas will also receive a different pixel value
        invert: pixel value for non-isolated areas (as list of values, representing the colors of a pixel)
        remove_color: When True, all color pixel will be overridden
        remove_value: new pixel value for color pixel
    
    Returns:
        A np.ndarray with dimensions of the inpt image
    
    ---------
    Examples:
    ---------
    
    trimm_and_blur(img, True, 20, (3, 3), [255, 255, 255])
    -> The function goes through each pixel in img. If the value of any color channel of the pixel is bigger than 20, the whole pixel will be overridden with the trimm parameter.\
        So, if a pixel has the following values: [14, 21, 3] It will be overriden to : [255, 255, 255] Once that's done for every pixel, a blur will be applied to the entire image.

    trimm_and_blur(img, True, 20, (3, 3), [255, 255, 255], True, [0, 0, 0])
    -> Now a trimm is also applied to the non-isolated parts of an image. If a pixel has the values [14, 21, 3], it will be overridden to [255, 255, 255].\
        A pixel with the values [16, 10, 12] will be overridden to [0, 0, 0].
    """
    for i in range(len(inpt)):
        for j in range(len(inpt[i])):
            if remove_color:
                if np.max(inpt[i][j]) - np.min(inpt[i][j]) > 2:
                    inpt[i][j] = remove_value            
            if less:
                if inpt[i][j][0] > value or inpt[i][j][1] > value or inpt[i][j][2] > value:
                    inpt[i][j] = np.array(trimm)
                elif double_down:
                    inpt[i][j] = np.array(invert)
            else:
                if inpt[i][j][0] < value or inpt[i][j][1] < value or inpt[i][j][2] < value:
                    inpt[i][j] = np.array(trimm)
                elif double_down:
                    inpt[i][j] = np.array(invert)
    blur = cv2.blur(inpt, blurs)
    return blur

def to_grayscale(img: np.ndarray): # -> np.ndarray
    """Converts a color image to grayscale.
    Note: If the input image has dimensions of 200x200x3, the output image will have dimensions of 200x200.
    
    Args:
        img: color image with BGR channel order
    
    Returns:
        The input image as grayscale.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def iso_grayscale(img: np.ndarray, less_than, value, convert_grayscale=False, blur=(1, 1), inverse=False): # -> np.ndarray
    """Isolates image areas with a specific value
    
    Args:
        img: input image as np.ndarray
        less_than: Sets filter technique to less than / bigger than
        value: Value to filter by
        convert_to:grayscale: True if the input image is color and needs to be converted to grayscale first
        blur: optional blur kernal size
    
    Returns:
        modified input image as np.ndarray
    """
    inv = None
    
    if convert_grayscale: img = to_grayscale(img)
    if inverse: inv = img.copy()

    if less_than:
        for i in range(len(img)):
            for j in range(len(img[i])):
                if img[i][j] < value:
                    img[i][j] = 255
                    if inverse: inv[i][j] = 0
                else:
                    img[i][j] = 0
                    if inverse: inv[i][j] = 255
    else:
        for i in range(len(img)):
            for j in range(len(img[i])):
                if img[i][j] > value:
                    img[i][j] = 255
                    if inverse: inv[i][j] = 0
                else:
                    img[i][j] = 0
                    if inverse: inv[i][j] = 255
    
    if blur != (1, 1):
        img = cv2.blur(img, blur)
    
    if inverse: return img, inv
    return img

def expand_to_rows(image: np.ndarray, full=False, value=200, bigger_than=True):  # -> np.ndarray
        """If one value in a row (of a mask, for example) is above a specific threshold, the whole row is expanded to a specific value.
        
        Args:
            image: An grayscale image as np.ndarray.
        
        Returns:
            A np.ndarray of the edited image.
        """
        dimensions = image.shape
        imglen = dimensions[0]
        
        white_row = np.array([255 for k in range(dimensions[1])])
        black_row = np.array([0 for k in range(dimensions[1])])
        
        if not full: imglen = dimensions[0] / 2
        if bigger_than:
            for i in range(int(imglen)):
                for j in range(dimensions[1]):
                    if image[i][j] > value:
                        image[i] = white_row
        else:
            for i in range(int(imglen)):
                for j in range(dimensions[1]):
                    if image[i][j] < value:
                        image[i] = black_row
        for i in range(int(imglen), dimensions[0]):
            for j in range(dimensions[1]):
                image[i] = black_row
        return image

def expand_to_rows_speed(image: np.ndarray, full=False, value=200, bigger_than=True):  # -> np.ndarray
    """If one value in a row (of a mask, for example) is above a specific threshold, the whole row is expanded to a specific value.
    
    Args:
        image: An grayscale image as np.ndarray.
    
    Returns:
        A np.ndarray of the edited image.
    """
    dimensions = image.shape
    imglen = dimensions[0]
    
    t1 = time.time()
    white_row = np.array([255 for k in range(dimensions[1])])
    black_row = np.array([0 for k in range(dimensions[1])])
    t2 = time.time()
    # print("exptr: create np array:", t2 - t1)
    
    t1 = time.time()
    if not full: imglen = dimensions[0] / 2
    if bigger_than:
        for i in range(int(imglen)):
            for j in range(0, dimensions[1], 2):
                if image[i][j] > value:
                    image[i] = white_row
    else:
        for i in range(int(imglen)):
            for j in range(0, dimensions[1], 2):
                if image[i][j] < value:
                    image[i] = black_row
    t2 = time.time()
    # print("exptr: loop complex:", t2 - t1)
    t1 = time.time()
    for i in range(int(imglen), dimensions[0]):
        for j in range(dimensions[1]):
            image[i] = black_row
    t2 = time.time()
    # print("exptr: end:", t2 - t1)
    return image

def determine_device(): # -> Literal["cuda", "cpu"]
    return "cuda" if torch.cuda.is_available() else "cpu"

def google(query: str):
        """Googles a query. Opens result in browser window.
        """
        link = "https://www.google.de/search?q="
        query.replace(" ", "+")

        webbrowser.open((link + query))

import warnings
import functools

def deprecated(func):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    
    The code for this function was copied from https://stackoverflow.com/a/30253848/12834761
    """
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                        category=DeprecationWarning,
                        stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func

class PlutoObject:
    def __init__(self, img: np.ndarray):
        self.img = img

    def load_model(self, path, model, device: Literal["cuda", "cpu"]):
        """Loads the state dictionary and applies it to the model
        
        Args:
            path: The relative or absolute path to the state dict file
            model: The corresponding PyTorch model
        
        Returns:
            The inputed PyTorch model with the loaded state.
        """
        model.load_state_dict(torch.load(path))
        model.to(device)
        
        return model
    
    def load_model_from_github(self, repository="patzold/yolov5-slim", weights="models/general.pt", force_reload=False):
        """Loads a model architecture from a GitHub repository.
        
        Args:
            repository: Path to the repository -> repo_owner/repo_name with optional branch
            weights: Path to weights the model will be loaded with
            force_reload: Force a fresh download of the repo
        
        Returns:
            An instance of the loaded model class with the weights specified in the parameter
        """
        _model = torch.hub.load(repository, "custom", path=weights, source="github", force_reload=force_reload)
        return _model

    def vis_model_prediction(self, img: np.ndarray, mask: np.ndarray, display=False):  # --> np.ndarray
        """ Shows the predicted mask of a segmentation model as an overlay on the input image.
            All arrays must be np.uint8 and have a color range of 0 - 255.
        
        Args:
            img: Input image for the model. Shape: IMG_SIZE x IMG_SIZE x 3
            mask: The result of the model. Shape: IMG_SIZE x IMG_SIZE
            display: True if the return image should also be displayed.
        
        Returns:
            The visualized prediction as np.ndarray
        """
        dim = img.shape
        mask = cv2.resize(mask, (dim[1], dim[0]))
        black_img = np.zeros([dim[0], dim[1]]).astype(np.uint8)
        # print(mask.shape, black_img.shape, mask.dtype, black_img.dtype, img.shape, img.dtype)
        overlay = cv2.merge((mask, black_img, black_img))
        out = cv2.addWeighted(img, 0.5, overlay, 1.0, 0)
        if display: show_image(out)
        return out

    def to_tensor(self, arr: np.ndarray, img_size, dtype, device: Literal["cuda", "cpu"], cc=3):  # --> torch.Tensor
        """Converts an np.ndarray (which represents an image) to a PyTorch Tensor
        
        Args:
            arr: The image as a NumPy array.
            img_size: The final image size (quadratic)
            dtype: The Type for the Tensor. Recommendet is torch.float32
            device: If the Tensor should be moved to the GPU, make this "cuda"
        
        Returns:
            The input array as torch.Tensor
        """
        arr = cv2.resize(arr.copy(), (img_size, img_size)) / 255.0 # load, resize & normalize
        # show_image(arr)
        arr = arr.reshape(-1, cc, img_size, img_size)
        tensor = torch.from_numpy(arr).to(dtype).to(device) # to tensor
        return tensor

    def from_tensor(self, tensor, img_size, dtype=np.uint8):
        return tensor.cpu().numpy().reshape(img_size, img_size).astype(dtype)
    
    def visualize_detection_result(self, img: np.array, result: np.array, one_class_only=None, color=[255, 0, 0]): # -> np.array
        """Visualize result of a yolo detection model
        """
        for r in result:
            if one_class_only is not None:
                if r[5] != one_class_only: continue
            cv2.rectangle(img, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), color, 2)
        
        return np.array(img)
    
    def detection_result_to_numpy(self, detection_result):
        """Converts a detections result to a NumPy Array.
        
        Args:
            detection_result: raw yolo output
        
        Returns:
            NumPy Array of results
        """
        return np.array([elem.cpu().numpy().tolist() for elem in detection_result.xyxy][0])
    
    def remove_overlapping_detections(self, detections: list):
        """If detections are overlapping (only y-axis wise), all smaller detections are removed
        from the detections array.
        """        
        detections.sort(key=lambda elem: elem[3] - elem[1], reverse=True)
        
        start = 0
        while start < len(detections):
            current = start + 1
            while current < len(detections):
                s_row = detections[start]
                c_row = detections[current]
                
                if s_row[1] < c_row[1] and s_row[3] > c_row[3]:
                    detections.pop(current)
                else:
                    current += 1
            start += 1
        
        return detections

    def ocr(self, image=None, switch_to_tesseract=False):  # -> str
        """Preforms OCR on a given image, using EasyOCR
        
        Args:
            image: np.ndarray of the to be treated image.
            switch_to_tesseract: deprecated parameter. can be assigned any value with no impact.
        
        Returns:
            String with the raw result of the OCR library.
        
        """
        if image is None: image = self.img
        try:
            # reader = easyocr.Reader(['en'])
            ocr_raw_result = reader.readtext(image, detail=0)
        except Exception as e:
            print("Pluto WARNING - Error while performing OCR: ", e)
            ocr_raw_result = [""]
        out = ""
        for word in ocr_raw_result:
            out += " " + word
        return out

    def expand_to_rows(self, image: np.ndarray, full=False, value=200):  # -> np.ndarray
        """
        Args:
            image: An grayscale image as np.ndarray, which represents a mask.
        
        Returns:
            A np.ndarray of the edited image.
        """
        dimensions = image.shape
        imglen = dimensions[0]
        if not full: imglen = dimensions[0] / 2
        for i in range(int(imglen)):
            for j in range(dimensions[1]):
                if image[i][j] > value:
                    image[i] = [255 for k in range(dimensions[1])]
        for i in range(int(imglen), dimensions[0]):
            for j in range(dimensions[1]):
                image[i][j] = 0
        return image

    def ocr_cleanup(self, text: str):  # -> str
        """Removes unwanted characters or symbols from a text
        
        This includes \n, \x0c, and multiple ' ' 
        
        Args:
            text: The String for cleanup.
        
        Returns:
            The cleaned text as String.
        """
        out = text.replace("\n", " ")
        out = out.replace("\x0c", "")
        out = " ".join(out.split())
        
        splits = out.split(",")
        clean_splits = []
        
        for phrase in splits:
            arr = list(phrase)
            l = len(arr)
            start = 0
            end = l
            for i in range(0, l):
                if arr[i] == " ": start += 1
                else: break
            for i in range(l, 0, -1):
                if arr[i-1] == " ": end -= 1
                else: break
            clean_splits.append(phrase[start:end])
        
        out = ""
        for phrase in clean_splits:
            out += phrase
            out += ", "
        out = out[:-2]
        
        return out

    def to_json(self, data: dict):
        import json
        out = json.dumps(data)
        return out

    def load_model(self, model, path: str, device):
        """Loads the state of an model
        
        Args:
            model: PyTorch model
            path: path to state dic
        
        Returns:
            the input model with loaded state
        """
        model.load_state_dict(torch.load(path))
        return model.to(device)

    def determine_device(self): # -> Literal["cuda", "cpu"]
        return "cuda" if torch.cuda.is_available() else "cpu"

    def run_model(self, model, tnsr):
        """Runs a model with a sigmoid activation function
        """
        with torch.no_grad():
            prediction = torch.sigmoid(model(tnsr))
        return prediction * 255

    def run_segmentation_model(self, state_path, img=None): # -> np.ndarray
        """Runs a UNET segmentation model given an image and the state of the model.
        
        Args:
            state_path: path to the models's state_dict
            img: the model's input image
        
        Returns:
            The model's prediction as np.ndarray
        """
        if img is None: img = self.img
        
        device = self.determine_device()
        
        model = UNET(in_channels=3, out_channels=1)
        model = self.load_model(model, state_path, device)
        
        input_tensor = self.to_tensor(img, 256, torch.float32, device)
        
        prediction = self.run_model(model, input_tensor)
        
        output = self.from_tensor(prediction, 256)
        
        return output

    def extr_mask_img(self, mask: np.ndarray, img: np.ndarray, inverted=False):
        """Performs extend_to_rows() on the mask and returns the masked out parts of the original image.
        
        Args:
            mask: grayscale mask
            img: Original image
            inverted: if True, an inverted version of the output will also be returned
        
        Returns:
            A np.ndarray of the masked out part
        """
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        
        extr = self.expand_to_rows(mask)
        
        out = []
        invout = []
        for i in range(len(extr)):
            if extr[i][0] > 200: out.append(img[i])
            elif inverted: invout.append(img[i])
        
        if inverted: return np.array(out), np.array(invout)
        return np.array(out)

    def extr_replace_mask(self, mask: np.ndarray, img: np.ndarray, replace_value: np.ndarray, invert_replace=False):
        """Performs extend_to_rows() on the mask and returns the masked out parts of the original image.
        
        Args:
            mask: grayscale mask
            img: Original image
            inverted: if True, an inverted version of the output will also be returned
        
        Returns:
            A np.ndarray of the masked out part
        """
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        
        extr = self.expand_to_rows(mask)
        
        if invert_replace:
            for i in range(len(extr)):
                if extr[i][0] > 200:
                    for j in range(len(img[i])):
                        img[i][j] = replace_value
        else:
            for i in range(len(extr)):
                if extr[i][0] < 200:
                    for j in range(len(img[i])):
                        img[i][j] = replace_value
        
        return img

    def characters_filter_strict(self, inpt: str, allow_digits=True): # -> str
        numbers =  list(range(128))
        # only digits, uppercase, lowercase and spaces letters are valid
        allowed_ascii_values = numbers[65:91] + numbers[97:123] + [32]
        if allow_digits: allowed_ascii_values += numbers[48:58]
        
        out = ""
        
        for i in inpt:
            if ord(i) in allowed_ascii_values: out += i
        
        out = " ".join(out.split())

        return out

class Detectors:
    def __init__(self, *args):
        """Stores loaded detection models.
        
        Args:
            Name of detection model to load
        """
        self.detector_models = {}
        
        self.load(args)
    
    def __repr__(self):
        out = []
        out.append("--- Detector ---")
        out.append("models loaded:")
        
        if len(self.detector_models) > 0:
            for key in self.detector_models.keys():
                out.append(f"  -> {str(key)} - {str(type(self.detector_models[key]))}")
        else:
            out.append("  [none]")
        
        return "\n".join(out)
    
    def return_model(self, model_name):
        """Returns a loaded detection model. Will load first if not done yet.
        """
        if not model_name in self.detector_models.keys():
            self._load_model(model_name)
        
        return self.detector_models[model_name]
    
    def load(self, *args, **kwargs):
        for a in args:
            if a == "" or a == (): continue
            self._load_model(a)
    
    def _load_model(self, model_name, force_reload=False):
        if model_name in self.detector_models.keys() and not force_reload:
            return
        
        self.detector_models[model_name] = self._load_detection_model(
            weights_path=f"models/{model_name}.pt",
            force_reload=force_reload)
    
    def _load_detection_model(self, weights_path: str, force_reload=False, local_path=None):
        """Loads a YOLOv5-slim object detection model with custom weigths using PyTorch Hub.
        
        Args:
            weights_path: Path to .pt file with model weigths, can be relative or absolute
            force_reload: Force reload of model -> no loading from cache.
            local_path: Optional. For loading model from a local directory. If empty, the model will be loaded from GitHub
        
        Returns:
            The YOLOv5-slim model class
        
        Example:
            load_detection_model(weights_path="v1.0/text_row3.pt", force_reload=False, local_path="/Users/examplusr/Models/yolov5-slim")
        """
        source = "github"
        path = "patzold/yolov5-slim"
        
        if local_path:
            source = "local"
            path = local_path
        
        return hub.load(path, "custom", path=weights_path, source=source, verbose=False)

class Facebook(PlutoObject):
    def __init__(self, img: np.ndarray):
        super().__init__(img)
        self.header = None
        self.text = None
        self.insert = None
        self.engagement = None
    
    def analyse(self, img=None):
        """Main method for extraction information from the image
        """
        if img is None: img = self.img
        
        splits = self.split(img)
        self.header, self.text = self.sliceing(splits[0])
        
        self.insert, self.engagement = None, None
        
        if len(splits) > 1 and splits[1] is not None: self.insert = splits[1]
        if len(splits) > 2 and splits[2] is not None: self.engagement = splits[2]
        
        date = self.header[int(self.header.shape[0] / 2) :]
        header = self.header[: int(self.header.shape[0] / 2)]
        
        engocr = None
        headerocr = self.ocr_cleanup(self.ocr(header))
        dateocr = self.ocr_cleanup(self.ocr(date))
        textocr = self.ocr_cleanup(self.ocr(self.text))
        if self.engagement is not None:
            engocr = self.ocr_cleanup(self.ocr(self.engagement))
            engocr = self.engagement_str(engocr)
        
        return headerocr, dateocr, textocr, engocr
    
    def split(self, img):
        """Splits the screenshot at an attached image or link
        """
        if img is None: img = self.img
        
        og_img = img.copy()
        img = to_grayscale(img)
        dm = self.dark_mode(img)
        if dm: img = 255 - img
        exists1 = False
        for i in range(len(img)):
            if not dm:
                if img[i][0] < 250:
                    exists1 = True
                    break
            else:
                if img[i][0] < 215:
                    exists1 = True
                    break
        
        top, bottom = None, None
        if exists1:
            top = og_img[:i-1]
            bottom_og = og_img[i:]
            bottom = img[i:]
        # else: return img
        
        if bottom is None:
            return og_img, None, None
        exists2 = False
        for j in range(len(bottom)-1, i, -1):
            if not dm:
                if bottom[j][0] < 250:
                    exists2 = True
                    break
            else:
                if bottom[j][0] < 215:
                    exists2 = True
                    break
        
        insert, engagement = None, None
        if exists2:
            insert = bottom_og[:j]
            engagement = bottom_og[j:]
        
        if insert is not None and self.classify(insert) == 1:
            ts, ins = top.shape, insert.shape
            temp = np.zeros((ts[0] + ins[0], ts[1], ts[2]), np.uint8)
            temp[:ts[0]] = top
            temp[ts[0]:] = insert
            top = temp
            insert = None
        
        return top, insert, engagement

    def sliceing(self, img):
        """Slices the Text & header
        """
        exptr = None
        img = to_grayscale(img)
        dm = self.dark_mode(img)
        
        if not dm: exptr = (255 - img.copy())
        else: exptr = img.copy()
        
        if not dm: exptr = expand_to_rows(exptr, True, 5)
        else: exptr = expand_to_rows(exptr, True, 40)

        slices = []
        for i in range(1, len(exptr)):
            if exptr[i][0] > 250 and exptr[i-1][0] < 50 or \
            exptr[i][0] < 250 and exptr[i-1][0] > 50:
                slices.append(i)

        slc = []

        for i in range(1, len(slices), 2):
            if slices[i] - slices[i-1] < 5: continue
            slc.append(img[slices[i-1]:slices[i]])
        
        # return slc
        return img[slices[0]:slices[1]], img[slices[1]:slices[len(slices)-1]]

    def first(self, img):
        """Gets the first block from a slice
        
        Args:
            img: Slice as np.ndarray
        
        Returns:
            The first block as np.ndarray
        """
        img = np.transpose(img, (1, 0))
        exptr = img.copy() #pl.to_grayscale(img.copy())
        exptr = expand_to_rows(exptr, False, 245, False)
        # show_image(exptr)
        
        for i in range(1, len(exptr)):
            if exptr[i-1][0] > 250 and exptr[i][0] < 100: break
        
        for j in range(i+1, len(exptr)):
            if exptr[j-1][0] < 100 and exptr[j][0] > 250: break
        
        # show_image(img[i:j])
        return img[i:j]
    
    def slices(self, img=None):
        """Image to slices
        """
        if img is None: img = self.img
        
        test = (255 - to_grayscale(img))
        for i in range(len(test)):
            if test[i][0] > 10: break
        
        # show_image(test)
        test = test[:i,:]
        j = i
        
        # show_image(test)
        test = expand_to_rows(test, True, 10)
        # show_image(test)

        slices = []
        for i in range(2, len(test)):
            if test[i][0] > 250 and test[i-1][0] < 10 or \
            test[i][0] < 250 and test[i-1][0] > 10:
                slices.append(i)
        # slices.append(len(img) - 1)
        
        return slices, j
    
    def img_eng(self, indx, img=None):
        img = img[indx:]
        
        test = (255 - to_grayscale(img))
        for i in range(len(test)-1, 0, -1):
            if test[i][0] > 10: break
        image = img[:i]
        eng = img[i:]
        
        return image, eng
    
    def engagement_str(self, ocr: str):
        """Returns shares and views from engagements
        """
        try:
            comsplit = ocr.split("Comments")
            sharesplit = comsplit[1].split("Shares")
            viewsplit = sharesplit[1].split("Views")
            return str(sharesplit[0].strip()) + " Shares", str(viewsplit[0].strip()) + " Views"
        except Exception as e:
            return ocr
    
    def part(self, img, slices):
        """From slice arr to list of images
        """
        if img is None: img = self.img
        
        out = []
        full = []
        
        for i in range(1, len(slices), 2):
            if slices[i] - slices[i-1] < 5: continue
            temp = img[slices[i-1]:slices[i]]
            og = temp
            temp = temp[:,:int(temp.shape[1]/3)]
            # show_image(temp)
            out.append(temp)
            full.append(og)
        
        return out, full
    
    def classify(self, img=None):
        """Image or still part of text?
        0 == image, 1 == text
        """
        if img is None: img = self.img
        img = to_grayscale(img)
        
        net = ConvNet(1, 6, 12, 100, 20, 2)
        net.load_state_dict(torch.load("Utility Models/general_1.pt"))
        
        device = self.determine_device()
        net.to(device)
        
        tnsr = self.to_tensor(img, 224, torch.float32, device, 1)
        
        net.eval()
        with torch.no_grad():
            net_out = net(tnsr.to(device))[0]
            predicted_class = torch.argmax(net_out)
        result = predicted_class.cpu().numpy()
        
        return result
    
    def header(self, img=None):
        if img is None: img = self.img
        
        path = "FB Models/fb1.pt"
        
        result = self.run_segmentation_model(path, img)
        show_image(result)
        
        result = expand_to_rows(result)
        show_image(result)
        
        result = cv2.resize(result, (img.shape[1], img.shape[0]))
        # print(result.shape, img.shape)
        
        out = []
        for i in range(len(result)):
            if result[i][0] > 200: out.append(img[i])
        
        show_image(np.array(out))
    
    def to_json(self, img=None, path=None):
        """Extracts information from a screenshot and saves it as json
        """
        if img is None: img = self.img.copy()
        import json
        name, date, body_text, engagement_text = self.analyse(img)
        
        jasoon = {  "source": "Facebook",
                    "category": "Social Media",
                    "post": {
                        "username": name,
                        "date": date,
                        "content": body_text,
                        "engagement": engagement_text
                    }
                }
        
        if path == None: return json.dumps(jasoon)
        else:
            out = open(path, "w")
            json.dump(jasoon, out, indent=6)
            out.close()
    
    def dark_mode(self, img=None):
        """Checks if dark mode is enabled
        
        Args:
            img: input screenshot (optional)
        
        Returns:
            Dark Mode enabled? True / False
        """
        if img is None: img = self.img
        dim = img.shape
        img = img[:int(dim[0]*0.1),:int(dim[0]*0.015)]
        avg = np.average(img)
        return avg < 220

    def split_legacy(self, img=None, darkmode=None): # -> np.ndarray
        """---
        DEPRECATED
        ---
        Splits a Facebook Screenshot into a top part (that's where the header is), a 'body' part (main text) and a 'bottom' part ('engagement stats').
        
        Args:
            img: Alternative to the default self.img
        
        Returns:
            The top, body & bottom part, all as np.ndarray
        """
        if img is None: img = self.img
        og = img.copy()
        
        if darkmode is None: darkmode = self.dark_mode(img)
        # print(darkmode)
        
        if darkmode: gry = iso_grayscale(img, False, 50, True)
        else: gry = iso_grayscale(img, True, 250, True)

        gry_extr = expand_to_rows(gry, False, 100)
        
        top = []
        middle = []

        c = 0
        for i in range(len(gry_extr)):
            if gry_extr[i][0] > 100 and c < 2:
                if c == 0: c += 1
                top.append(og[i])
            elif c == 1: c += 1
            elif c == 2: middle.append(og[i])
        
        top = np.array(top)
        middle = np.array(middle)
        
        non_color = []
        for i in range(len(middle)):
            pix = middle[i][5] 
            if pix[0] > 250 or pix[1] > 250 or pix[2] > 250:
                non_color.append(middle[i])

        non_color = np.array(non_color)

        rgh = to_grayscale(non_color)

        rgh = rgh[:, int(non_color.shape[1] / 2):]
        rgh = iso_grayscale(rgh, True, 10, False, (25, 25))
        rgh = expand_to_rows(rgh, True, 5)
        
        body = []
        engagement = []
        for i in range(len(rgh)):
            if rgh[i][0] > 200: body.append(non_color[i])
            else: engagement.append(non_color[i])

        body = np.array(body)
        engagement = np.array(engagement)
        
        self.top = top
        self.body = body
        self.engagement = engagement
        
        return top, body, engagement
    
    @deprecated
    def old_topsplit(self, img=None, darkmode=None): # -> np.ndarray
        """---
        DEPRECATED
        ---
        
        Splits a Facebook Screenshot into a top part (thst's where the header is) and a 'bottom' part.
        
        Args:
            img: Alternative to the default self.img
        
        Returns:
            The top part and the bottom part, both as np.ndarray
        """
        if img is None: img = self.img
        og = img.copy()
        
        if darkmode is None: darkmode = self.dark_mode(img)
        # print(darkmode)
        
        if darkmode: gry = iso_grayscale(img, False, 50, True)
        else: gry = iso_grayscale(img, True, 250, True)

        gry_extr = expand_to_rows(gry, False, 100)
        
        top = []
        middle = []
        bottom = []

        c = 0
        for i in range(len(gry_extr)):
            if gry_extr[i][0] > 100 and c < 2:
                if c == 0: c += 1
                top.append(og[i])
            elif c == 1: c += 1
            elif c == 2: break

        c = 0
        for j in range(len(gry_extr)-1, i, -1):
            if gry_extr[j][0] > 100 and c < 2:
                if c == 0: c += 1
                bottom.insert(0, og[j])
            elif c == 1: c += 1
            elif c == 2: break
        
        for l in range(i, j, 1):
            middle.append(og[l])
        
        # print(i, j, l)
        
        top = np.array(top)
        middle = np.array(middle)
        bottom = np.array(bottom)
        
        self.top = top
        self.middle = middle
        self.bottom = bottom
        
        return top, middle, bottom
    
    def search(self, query: str):
        """Searches a query with Facebook's search function. Opens result in browser window.
        """
        link = "https://www.facebook.com/search/top/?q="
        query.replace(" ", "+")
        
        webbrowser.open((link + query))
    
    @deprecated
    def clean_top(self, img=None):
        """---
        DEPRECATED
        ---
        'Cleans' the top excerpt by removing the profile picture
        """
        if img is None: img = self.top

        og_img = img.copy()
        img2 = img[:,int(img.shape[1]*0.5):,:]
        img = img[:,:int(img.shape[1]*0.5),:]
        dim = img.shape
        img = cv2.resize(og_img[:,:int(og_img.shape[1]*0.5),:], (200, 200))
        
        prediction = self.run_segmentation_model("Utility Models/imgd_2_net_1.pt", img)
        # self.vis_model_prediction(img, prediction, True)
        
        prediction = np.transpose(prediction, (1, 0))
        prediction = cv2.resize(prediction, (dim[0], dim[1]))
        
        img = og_img[:,:int(og_img.shape[1]*0.5),:]
        # show_image(img)
        # show_image(img2)
        img = np.transpose(img, (1, 0, 2))
        # show_image(prediction)
        # show_image(img)
        vis = self.vis_model_prediction(img, prediction)
        
        out = self.extr_replace_mask(prediction, img, np.array([255, 255, 255]), True)
        out = np.transpose(img, (1, 0, 2))
        
        # show_image(out)
        out = np.concatenate((out, img2), axis=1)
        # show_image(out)
        
        self.top = out
        return out, vis

class Twitter(PlutoObject):
    def __init__(self, screenshot: np.ndarray, detectors=None):
        super().__init__(screenshot)
        
        self.screenshot = screenshot
        if screenshot is not None:
            self.visualization = self.screenshot.copy()
        else:
            self.visualization = None
        self.detectors = detectors
        
        if self.detectors is None:
            self.detectors = Detectors()
        
        self.detectors.load("profile_pic", "text_row", "words")
        
        self.detections = {}
        self.data = {}
        
        self.data["source"] = "Twitter"
        self.data["category"] = "Social Media"
        self.data["tweet"] = {}
        
        self._profile_lower_indx = 0
    
    def load_detection_model(self, weights_path: str, force_reload=False, local_path=None):
        """Loads a YOLOv5-slim object detection model with custom weigths using PyTorch Hub.
        
        Args:
            weights_path: Path to .pt file with model weigths, can be relative or absolute
            force_reload: Force reload of model -> no loading from cache.
            local_path: Optional. For loading model from a local directory. If empty, the model will be loaded from GitHub
        
        Returns:
            The YOLOv5-slim model class
        
        Example:
            load_detection_model(weights_path="models/text_row.pt", force_reload=False, local_path="/Users/examplusr/Models/yolov5-slim")
        """
        source = "github"
        path = "patzold/yolov5-slim"
        
        if local_path:
            source = "local"
            path = local_path
        
        return torch.hub.load(path, "custom", path=weights_path, source=source)
    
    def get_profile(self, screenshot=None):
        """Extracts profile information from a Twitter screenshot
        
        Args:
            screenshot: screenshot to use instead of internal one [optional]
        
        Returns:
            handle, username
        """
        if screenshot is None: screenshot = self.screenshot
        
        profile_pic_detection = self.detectors.return_model("profile_pic")
        
        result = profile_pic_detection(screenshot)
        result = self.detection_result_to_numpy(result)
        self.detections["profile_pic"] = result
        
        self.visualize_detection_result(self.visualization, result)
        
        # print(f"Twitter | Profile Pic Detection - detections: {len(result)}")
        
        if len(result) < 1:
            print(f"Twitter | WARNING: Profile Pic Detection has less than one detection!")
            return None
        
        profile_upper_indx = int(np.min(result[:, 1])) # if there are multiple detections, use the min
        profile_lower_indx = int(np.max(result[:, 3])) # and max index for upper and lower bounds
        self._profile_lower_indx = profile_lower_indx
        
        profile_img = screenshot[profile_upper_indx : profile_lower_indx]
        
        # self.show_image(profile_img)
        
        # remove the profile image for clean ocr results
        img_detection = self.detectors.return_model("img_detct")
        result = img_detection(profile_img)
        result = self.detection_result_to_numpy(result)
        self.detections["profile_pic_img"] = result
        
        if len(result) < 1:
            print(f"Twitter | WARNING: Image Detection of the profile row has less than one detection!")
        
        profile_left_indx = int(np.max(result[:, 2]))
        
        profile_img = profile_img[:, profile_left_indx : int(profile_img.shape[1]*0.9)]
        
        # perform ocr & split into handle and username
        text = self.ocr(profile_img)
        text = self.ocr_cleanup(text)

        text = text.split("@")
        
        handle = str(text[-1])
        username = "".join(text[:-1])
        
        # print(handle)
        # print(username)
        # self.show_image(profile_img)
        
        self.data["tweet"]["user"] = {}
        self.data["tweet"]["user"]["handle"] = handle
        self.data["tweet"]["user"]["username"] = username
        
        return handle, username
    
    def contains_image(self, excert: np.array): # -> np.array
        """
        """
        image_detection = self.detectors.return_model("img_detct")
        result = image_detection(excert)
        result = self.detection_result_to_numpy(result)
        result = self.remove_overlapping_detections(result.tolist())
        
        vis = self.visualize_detection_result(excert, np.array(result))
        return vis
    
    def process_text_rows(self, img):
        """
        """
        # search for text rows
        text_row_detection = self.detectors.return_model("text_row")
        
        result = text_row_detection(img)
        result = self.detection_result_to_numpy(result)
        
        # sort rows in correct order (top -> bottom)
        result = result.tolist()
        result.sort(key=lambda elem: elem[1])
        result = np.array(result)
        
        self.detections["text_row"] = result
        
        # find out what color words in each text row have
        word_detection = self.detectors.return_model("words")
        
        text_row_word_color = []
        
        for r in result:
            words_result = word_detection(img[int(r[1]) : int(r[3])])
            words_result = self.detection_result_to_numpy(words_result)
            
            text_row_word_color.append([r, words_result])
        
        self.detections["text_row_word_color"] = text_row_word_color
        return text_row_word_color
    
    def find_rows_w_color(self, text_row_word_color, color_class, img=None):
        """Find all rows from text_row_word_color that have a word with color_class
        
        color classes:
        0 -> white
        1 -> gray
        2 -> blue
        """
        out = []
        
        for row in text_row_word_color:
            if len(row) != 2 or len(row[1]) == 0: continue
            
            if 1 in row[1][:, 5]:
                out.append(row[0])
                if img is not None:
                    cv2.rectangle(img, (int(row[0][0]), int(row[0][1])),
                                (int(row[0][2]), int(row[0][3])), [255, 0, 0], 2)
        
        if img is not None:
            return out, np.array(img)
        
        return out

    def get_content(self, screenshot=None, force_redetection=False):
        """Extracts the content (Tweet text itself) from a screenshot
        """
        if screenshot is None: screenshot = self.screenshot
        
        # find out where the profile row ends
        if force_redetection or not "profile_pic" in self.detections.keys():
            profile_pic_detection = self.detectors.return_model("profile_pic")
            result = profile_pic_detection(screenshot)
            result = self.detection_result_to_numpy(result)
            self.detections["profile_pic"] = result
        
        result = self.detections["profile_pic"]
        
        profile_lower_indx = int(np.max(result[:, 3])) + 1
        self.detections["profile_lower_indx"] = profile_lower_indx
        screenshot = screenshot[profile_lower_indx:]
        self._screenshot_w_o_profile = screenshot
        vis_copy = self._screenshot_w_o_profile.copy()
        
        # self.contains_image(screenshot)
        
        # the content only includes rows with white or blue words, no gray
        text_row_word_color = self.process_text_rows(screenshot)
        gray_rows = self.find_rows_w_color(text_row_word_color, 1)
        gray_rows = np.array(self.remove_overlapping_detections(gray_rows))
        self.visualize_detection_result(vis_copy, gray_rows, color=[0, 255, 0])
        self.detections["gray_rows"] = gray_rows
        # print(gray_rows)
        
        first_gray_row_indx = np.min(gray_rows[:, 1])
        # print(first_gray_row_indx, len(vis_copy))
        
        cv2.rectangle(vis_copy, (1, 5), (len(vis_copy[0])-1, int(first_gray_row_indx)-5), [0, 150, 255], 2)
        self.visualization[profile_lower_indx:] = vis_copy
        
        # perform ocr
        text = self.ocr(screenshot[:int(first_gray_row_indx)])
        text = self.ocr_cleanup(text)
        
        self.data["tweet"]["text"] = text
        return text
    
    def get_engagement(self, screenshot, force_redetection=False):
        """Extracts the engagement from a screenshot
        
        Args:
            screenshot: Twitter screenshot
        """
        if screenshot is None: screenshot = self.screenshot
        
        screenshot = screenshot[self._profile_lower_indx:]
        
        # get gray rows (potential engagement rows)
        if force_redetection or "gray_rows" not in self.detections.keys():
            text_row_word_color = self.process_text_rows(screenshot)
            gray_rows = self.find_rows_w_color(text_row_word_color, 1)
            gray_rows = np.array(self.remove_overlapping_detections(gray_rows))
            self.detections["gray_rows"] = gray_rows
        
        gray_rows = self.detections["gray_rows"]
        
        self.data["tweet"]["engagement"] = {}
        
        # we mainly work with strings to find engagement rows
        # perform ocr
        for r in gray_rows:
            excert = screenshot[int(r[1]) : int(r[3])]
            ocr_txt = self.ocr(excert)
            ocr_txt = self.ocr_cleanup(ocr_txt).lower()
            
            words = ocr_txt.split(" ")
            # print(words)
            
            if "likes" in words:
                txt_indx = words.index("likes")
                if txt_indx < 1: continue
                self.data["tweet"]["engagement"]["likes"] = words[txt_indx-1]
            
            if "retweets" in words:
                txt_indx = words.index("retweets")
                if txt_indx < 1: continue
                self.data["tweet"]["engagement"]["retweets"] = words[txt_indx-1]
            
            if "quote" in words:
                txt_indx = words.index("quote")
                if txt_indx < 1: continue
                self.data["tweet"]["engagement"]["quoted"] = words[txt_indx-1]
    
    def analyse(self, screenshot=None):
        """Main method for extraction information from the image
        """
        if screenshot is not None:
            self.screenshot = screenshot
            self.visualization = screenshot.copy()
        
        self.data["tweet"] = {}
        self.detections = {}
        
        self.get_profile(self.screenshot)
        self.get_content(self.screenshot)
        self.get_engagement(self.screenshot)
        
        return self.data
    
    def slice(self, img=None):
        """slices the screenshot into multiple rows with different content
        
        Returns:
            list of rows (rows are np.array)
        """
        if img is None: img = self.img
        
        img_og = img.copy()
        img_bw = to_grayscale(img_og.copy())

        img = expand_to_rows(to_grayscale(img[:, :int(len(img[0]) * 0.7)]), True, 150, False) # scroll bar removed

        slices = []
        for i in range(1, len(img)):
            if img[i][0] > 250 and img[i-1][0] < 5 or \
                img[i][0] < 250 and img[i-1][0] > 5:
                slices.append(i)
        slices.append(len(img) - 1)

        parts = []

        for i in range(1, len(slices)):
            sliced = img_og[slices[i-1]:slices[i]]
            avg = np.average(sliced)
            if avg < 250:
                parts.append([img_og[slices[i-1]-3:slices[i]+3], img_bw[slices[i-1]-3:slices[i]+3]])
        
        return parts
    
    def assign(self, parts):
        """assigns each row to ether the top (header + content) or bottom (metadata) part
        
        Returns:
            top (as np.array), has bottom? (boolean), bottom (np.array, only if has_bottom is true)
        """
        top = []
        
        # check if screenshot has bottom metadata part
        has_bottom = False
        canidate = parts[-1][1]
        if np.min(canidate) > 11:
            bottom = canidate
            has_bottom = True

        for i in range(1, len(parts), 1):
            temp = parts[i][1]
            minv = np.min(temp)
            averg = np.average(temp)
            if minv < 10:
                top.append(parts[i][0])
        
        if has_bottom:
            return top, has_bottom, bottom
        else: return top, None
    
    def header_cleanup(self, header):
        """seperates the profile picture from the header information
        
        Returns:
            the profile picture & header information (both as np.array)
        """
        # remove profile picture
        row = header[0].copy()
        bwrow = header[1].copy()

        bwrow = np.transpose(bwrow, (1, 0))
        row = np.transpose(row, (1, 0, 2))
        rowexptr = expand_to_rows(bwrow, True, 240, False)

        imprts = []
        for i in range(1, len(rowexptr)):
            if rowexptr[i][0] > 250 and rowexptr[i-1][0] < 50 or \
            rowexptr[i][0] < 250 and rowexptr[i-1][0] > 50:
                imprts.append(i)

        slc = []

        for i in range(1, len(imprts), 2):
            if imprts[i] - imprts[i-1] < 5: continue
            tempelem = np.transpose(row[imprts[i-1]:imprts[i]], (1, 0, 2))
            tempocr = self.ocr(tempelem[int(tempelem.shape[0] / 2) :]).strip()
            
            if tempocr[0] == "@": break

        row = np.transpose(row, (1, 0, 2))
        profile_pic = row[:, :imprts[i-1]]
        header_info = row[:, imprts[i-1]:]
        
        return profile_pic, header_info
    
    def extract(self, header, bottom):
        """extracts data from images
        
        Returns:
            username, handle, postdate, client (all str)
        """
        if header is None: header = self.header_info
        if bottom is None: header = self.bottom
        # header data
        subheader = header[int(header.shape[0] / 2) :]
        header = header[: int(header.shape[0] / 2)]
        subheader_ocr_result = self.ocr_cleanup(self.ocr(subheader))
        header_ocr_result = self.ocr_cleanup(self.ocr(header))
        
        postdate, client = None, None
        if bottom is not None:
            bottom_ocr_result = self.ocr_cleanup(self.ocr(bottom[1]))
            bts = bottom_ocr_result.split("Twitter")
            client = "Twitter" + bts[1]
            postdate = bts[0]
        
        return header_ocr_result, subheader_ocr_result, postdate, client
    
    def to_json(self, img=None, path=None):
        """Extracts information from screenshot and saves it as json file.
        
        Args:
            img: screenshot as np.array
            path: path to where the json file should be saved
        """
        if img is None: img = self.img.copy()
        import json
        result = self.analyse(img)
        
        if path == None: return json.dumps(result)
        else:
            out = open(path, "w")
            json.dump(result, out, indent=6)
            out.close()

class NYT(PlutoObject):
    def __init__(self, img: np.ndarray):
        super().__init__(img)
        self.header_img = None
        self.headline = None
    
    def header(self, img=None, non_header=False):
        """Isolates the headline from an article, based on color
        
        Args:
            img: if the used screenshot should differ from self.img, pass it here
            non_header: True if all other parts of the screenshot, that don't belong to the headline, \
                        should also be returned seperatly (basicly a inverted return of this method)
        
        Returns:
            An isolated part of the original screenshot, which only contains the headline\
            (optional also an inverted version, when non_header is True)
        """
        if img is None: img = self.img.copy()
        img_og = img.copy()
        # print(img.shape)
        img = cv2.resize(img, (255, 255))
        img = trimm_and_blur(img, False, 30, (15, 15), [255, 255, 255], True, [0, 0, 0])
        
        extr = expand_to_rows(img[:,:,0], True, 10)
        extr = cv2.resize(extr[:,:10], (10, img_og.shape[0]))
        
        out = []
        inverse_out = []
        
        for i in range(len(extr)):
            if extr[i][0] > 200: out.append(img_og[i])
            elif non_header: inverse_out.append(img_og[i])
        
        out = np.array(out)
        
        if non_header: return out, inverse_out
        return out

    def images(self, img=None, non_images=False): # -> np.ndarary | None
        """Isolates images of an article, based on color
        
        Args:
            img: if the used screenshot should differ from self.img, pass it here
            non_images: True if all other parts of the screenshot, that aren't partof an image, \
                        should also be returned seperatly (basicly a inverted return of this method)
        
        Returns:
            An isolated part of the original screenshot, which only contains the headline\
            (optional an inverted version as well, when non_images is True)
            If no images can be found, the return is None
        
        Please make sure that img is not scaled down and *not* grayscale
        """
        if img is None: img = self.img.copy()
        
        img_og = img.copy()
        img = cv2.resize(img, (50, img_og.shape[0]))
        
        image = []
        non_image = []
        
        j = 0
        stop = False
        
        for i in range(len(img)):
            stop = False
            while j < len(img[i]) and not stop:
                pix = img[i][j]
                minpix, maxpix = np.min(pix), np.max(pix)
                difference = maxpix - minpix
                if difference > 0:
                    image.append(img_og[i])
                    stop = True
                
                j += 1
            
            if not stop and non_images: non_image.append(img_og[i])
            j = 0
        
        image = np.array(image)
        if non_images: non_image = np.array(non_image)

        if len(image) < 1:
            image = None
            print("Pluto WARNING - At least one return from images() is empty.")
        
        if non_images: return image, non_image
        return image
    
    def suber(self, img=None):
        """Isolates the headline from an article, based on color
        
        Args:
            img: if the used screenshot should differ from self.img, pass it here
            non_header: True if all other parts of the screenshot, that don't belong to the headline, \
                        should also be returned seperatly (basicly a inverted return of this method)
        
        Returns:
            An isolated part of the original screenshot, which only contains the headline\
            (optional also an inverted version, when non_header is True)
        """
        if img is None: img = self.img.copy()
        img_og = img.copy()
        
        img = cv2.resize(img, (255, 255))
        img = trimm_and_blur(img, False, 55, (15, 15), [255, 255, 255], True, [0, 0, 0])
        
        extr = expand_to_rows(img[:,:,0], True, 10)
        extr = cv2.resize(extr[:,:10], (10, img_og.shape[0]))
        # show_image(extr)
        out = []
        
        for i in range(len(extr)):
            if extr[i][0] > 200: out.append(img_og[i])
        
        out = np.array(out)
        return out
    
    def analyse(self, img=None):
        """Main method for extraction information from a screenshot of a NYT article.
        """
        analyse_img = img
        if analyse_img is None: analyse_img = self.img
        import time
        
        t1 = time.time()
        sliced_result = self.slice(analyse_img)
        t2 = time.time()
        # print("slice:", t2 - t1)
        top, color_images, bottomnp = None, None, None
        if len(sliced_result) == 1: top = sliced_result[0]
        elif len(sliced_result) == 3: top, color_images, bottomnp = sliced_result
        # show_image(color_images)
        # show_image(np.array(top))
        
        bottom = []
        if bottomnp is not None:
            for i in range(len(bottomnp)):
                bottom += (bottomnp[i]).tolist()
        
        top = np.array(top, np.uint8)
        if bottomnp is None: bottom = None
        else: bottom = np.array(bottom, np.uint8)
        
        # show_image(top)
        # show_image(color_images)
        # show_image(bottom)
        
        t1 = time.time()
        head, body = self.header(top, True)
        t2 = time.time()
        # print("header:", t2 - t1)
        # show_image(head)
        # show_image(body)
        # return head, body
        
        t1 = time.time()
        self.headline = self.ocr_cleanup(self.ocr(head))
        author = self.author(bottom)
        # print(headline)
        
        # print(type(body))
        subt = self.suber(np.array(body))
        subtitle = self.ocr_cleanup(self.ocr(subt))
        t2 = time.time()
        print("ocr:", t2 - t1)
        
        return self.headline, subtitle, author
    
    def slice(self, img=None):
        """Slices image
        
        Returns:
            top, image, bottom
        """
        if img is None: img = self.img.copy()
        
        img_og = img.copy()
        
        # show_image(img[:, :int(len(img[0]) * 0.9)])
        import time
        t1 = time.time()
        img = expand_to_rows_speed(to_grayscale(img[:, :int(len(img[0]) * 0.9)]), True, 248, False) # scroll bar removed
        t2 = time.time()
        print("sclice: exptr:", t2 - t1)
        # show_image(img)
        # quit()
        
        t1 = time.time()
        slices = []
        for i in range(1, len(img)):
            if img[i][0] > 250 and img[i-1][0] < 5 or \
                img[i][0] < 250 and img[i-1][0] > 5:
                slices.append(i)
        slices.append(len(img) - 1)
        
        parts = []
        
        for i in range(1, len(slices)):
            parts.append(img_og[slices[i-1]:slices[i]])
            # show_image(img_og[slices[i-1]:slices[i]])
        t2 = time.time()
        print("sclice: appending:", t2 - t1)
        
        t1 = time.time()
        top = []
        difflen = 0
        """
        for i in range(len(parts)):
            temp = parts[i]
            for rindx, row in enumerate(temp):
                for pindx, pix in enumerate(row):
                    minpix, maxpix = np.min(pix), np.max(pix)
                    difference = maxpix - minpix
                    if difference > 10: difflen += 1
                    if difflen > 50:
                        if self.classify(temp) == 0:
                            t2 = time.time()
                            print("sclice: end:", t2 - t1)
                            print("indx:", rindx, pindx, len(temp), len(temp[0]))
                            return top, temp, parts[i+1:]
            top += temp.tolist()
            print("none indx:", rindx, pindx, len(temp), len(temp[0]))
        t2 = time.time()
        print("sclice: end:", t2 - t1)
        """
        
        for i in range(len(parts)):
            temp = parts[i]
            for rindx, row in enumerate(temp):
                for p in range(0, len(row), 4):
                    minpix, maxpix = np.min(row[p]), np.max(row[p])
                    difference = maxpix - minpix
                    if difference > 10: difflen += 1
                    if difflen > 50:
                        if self.classify(temp) == 0:
                            t2 = time.time()
                            print("sclice: end:", t2 - t1)
                            # print("indx:", rindx, p, len(temp), len(temp[0]))
                            return top, temp, parts[i+1:]
            top += temp.tolist()
            # print("none indx:", rindx, p, len(temp), len(temp[0]))
        t2 = time.time()
        print("sclice: end:", t2 - t1)
        
        return [top]
    
    def classify(self, img=None):
        """Image or still part of text?
        """
        if img is None: img = self.img
        img = to_grayscale(img)
        
        net = ConvNet(1, 6, 12, 100, 20, 2)
        net.load_state_dict(torch.load("Utility Models/general_1.pt"))
        
        device = self.determine_device()
        net.to(device)
        
        tnsr = self.to_tensor(img, 224, torch.float32, device, 1)
        
        net.eval()
        with torch.no_grad():
            net_out = net(tnsr.to(device))[0]
            predicted_class = torch.argmax(net_out)
        result = predicted_class.cpu().numpy()
        
        return result
    
    def author(self, img=None):
        """Iso author
        
        Returns:
            top, image, bottom
        """
        if img is None: return None
        # print(len(img))
        
        img_og = img.copy()
        img = expand_to_rows(to_grayscale(img), True, 80, False)
        # show_image(img)
        
        slices = []
        for i in range(1, len(img)):
            if img[i][0] > 250 and img[i-1][0] < 5 or \
                img[i][0] < 250 and img[i-1][0] > 5:
                slices.append(i)
        slices.append(len(img) - 1)
        
        parts = []
        
        for i in range(1, len(slices)):
            parts.append(img_og[slices[i-1]:slices[i]])
        
        for p in parts:
            ocr_result = self.ocr_cleanup(self.ocr(p))
            if "By " in ocr_result: return ocr_result[3:]
        
        return None
    
    def to_json(self, img=None, path=None):
        if img is None: img = self.img.copy()
        import json
        result = self.analyse(img)
        
        jasoon = {  "source": "New York Times",
                    "category": "News Article",
                    "article": {
                        "organisation": "New York Times",
                        "headline": result[0],
                        "subtitle": result[1],
                    }
                }
        
        if path == None: return json.dumps(jasoon)
        else:
            out = open(path, "w")
            json.dump(jasoon, out, indent=6)
            out.close()
    
    def headline_using_yolo(self, img=None):
        """Localizes and returns an image excert of the article headline
        
        Args:
            img: if the used screenshot should differ from self.img, pass it here [optional]
        
        Returns:
            The image excert of the headline and the y coordinate where the excert ends
        """
        if img is None: img = self.img
        
        detect_text_block = self.load_model_from_github(weights="models/text_block.pt")
        
        result = detect_text_block(img)

        text_blocks = []

        # process yolo results
        xyxy = result.xyxy[0].cpu().numpy()
        for i in range(len(xyxy)):
            xyxyc = result.xyxy[0].cpu().numpy()[i]
            
            excert = img[int(xyxyc[1]) : int(xyxyc[3]), int(xyxyc[0]) : int(xyxyc[2])]
            y_center = int(xyxyc[3] - xyxyc[1]) / 2
            
            text_blocks.append([excert, xyxyc, y_center, np.min(excert), np.max(excert)])

        # sort detected areas from top to bottom
        text_blocks.sort(key=lambda text_block: text_block[2])
        
        # check if excert has correct color to be the headline
        headline_img = None
        headline_y_end = None

        for tb in text_blocks:
            if headline_img is None:
                if tb[3] < 25:
                    # this is the headline block, widen selected area if necessary
                    x_min = int(min(tb[1][0], img.shape[1]*0.05))
                    x_max = int(max(tb[1][3], img.shape[1]*0.95))
                    
                    headline_img = img[int(tb[1][1]) : int(tb[1][3]), x_min : x_max]
                    headline_y_end = int(tb[1][3])
        
        return headline_img, headline_y_end
    
    def search(self, query: str):
        """Searches a query with the NYT's search function. Opens result in browser window.
        """
        link = "https://www.nytimes.com/search?query="
        query.replace(" ", "+")
        
        webbrowser.open((link + query))
    
    def open_search(self):
        self.search(self.headline)

class Tagesschau(PlutoObject):
    def __init__(self, img: np.ndarray):
        super().__init__(img)
    
    def analyse(self, img=None):
        """Do Tagesschau
        """
        if img is None: img = self.img
        
        dm = self.dark_mode(img)
        img = to_grayscale(img)
        # show_image(img)
        
        s = self.remove_image(img, dm)
        if len(s) == 2:
            image = s[0]
            text = s[1]
        else: text = s
        
        s = self.slicing(text, dm)
        return self.key_lines(s, dm)
    
    def remove_image(self, screenshot=None, dm=False):
        """Removes the images from the screenshot, leaving only text
        
        Args:
            screenshot: input screenshot (grayscale)
            dm: dark mode enabled
        
        Returns:
            The non-image and text parts of the input screenshot
        """
        if screenshot is None: screenshot = self.img
        
        dim = screenshot.shape
        excerpt = screenshot[:, :int(dim[0]*0.01)]
        
        stop = False
        for i in range(len(excerpt)-1):
            for l in range(len(excerpt[i])):
                if dm:
                    if excerpt[i][l] > 25 or excerpt[i][l] < 11: stop = True
                else:
                    if excerpt[i][l] < 248: stop = True
            if stop: break
        
        stop = False
        for j in range(len(excerpt)-1, i, -1):
            for l in range(len(excerpt[j])):
                if dm:
                    if excerpt[j][l] > 25 or excerpt[j][l] < 11: stop = True
                else:
                    if excerpt[j][l] < 248: stop = True
            if stop: break
        
        image = screenshot[i:j]
        text = np.delete(screenshot, range(i, j), 0)
        
        # confirm suspected image
        net = ConvNet(1, 6, 12, 100, 20, 2)
        net.load_state_dict(torch.load("models/general_1.pt"))
        
        device = self.determine_device()
        tnsr = self.to_tensor(image, 224, torch.float32, device, 1)
        
        net.to(device).eval()
        with torch.no_grad():
            net_out = net(tnsr.to(device))[0]
            predicted_class = torch.argmax(net_out)
        result = predicted_class.cpu().numpy() # 0 == image, 1 == text
        
        if result == 0: return image, text
        else: return screenshot
    
    def slicing(self, img=None, dm=False):
        """Performs line split
        
        Args:
            img: the input image
            dm: is dark mode enabled
        
        Returns:
            A list of slices
        """
        if img is None: img = self.img.copy()
        og_img = img.copy()
        img_dim = img.shape
        if dm:
            exptr = 255 - expand_to_rows(img[:, :int(img_dim[0]*0.5)], True, 80)
        else: exptr = expand_to_rows(img[:, :int(img_dim[0]*0.5)], True, 150, False)
        
        zero = np.zeros((1, int(img_dim[0]*0.5)), dtype=np.uint8)
        full = 255 - zero
        
        for i in range(len(exptr)):
            if exptr[i][0] < 125: exptr[i] = zero
            else: exptr[i] = full
        
        # show_image(exptr)

        slices = []
        if exptr[0][0] < 50: slices.append(0)
        for i in range(1, len(exptr)):
            if exptr[i-1][0] > 250 and exptr[i][0] < 50: # or \
            # exptr[i][0] > 250 and exptr[i-1][0] < 50:
                slices.append(i)

        slc = []

        for i in range(1, len(slices), 1):
            if slices[i] - slices[i-1] < 5: continue
            slc.append(og_img[slices[i-1]:slices[i]])
        
        # for s in slc: show_image(s)
        
        return slc
    
    def key_lines(self, slices, dm):
        """Isolates the key liens of text from a slice of a screenshot
        
        Args:
            slices: list of slices
            dm: is dark mode
        
        Returns:
            Date, Category, Title, Content
        """
        slc = []
        
        for s in slices:
            if not dm: s = 255 - s
            # show_image(s)
            s_ocr = self.ocr_cleanup(self.ocr(s))
            s = s[:, :int(s.shape[1]*0.5)]
            m = np.max(s)
            slc.append([s, s_ocr, m])
            # print(m, s_ocr)
            # show_image(s)
        
        for s in range(len(slc)):
            if slc[s][2] < 200 and "Stand:" in slc[s][1]: break
        
        pubsplit = slc[s][1][7:]
        content = ""
        title = ""
        category = ""
        
        for i in range(s+1, len(slc)):
            content += slc[i][1] + " "
        
        for i in range(s):
            if slc[i][1][-2:] == "AA":
                category = slc[i][1][:-2]
                continue
            title += slc[i][1] + " "
        
        return pubsplit.strip(), category.strip(), title.strip(), content.strip()
    
    def to_json(self, img=None, path=None):
        """Extracts information from screenshot and saves it as json file.
        
        Args:
            img: screenshot as np.array
            path: path to where the json file should be saved
        """
        if img is None: img = self.img.copy()
        import json
        date, category, headline, body = self.analyse(img)
        
        jasoon = {  "source": "Tagesschau",
                    "category": "News Article",
                    "article": {
                        "created": "[Updated] " + date,
                        "organisation": "Tagesschau",
                        "headline": headline,
                        "body": body,
                        "category": category
                    }
                }
        
        if path == None: return json.dumps(jasoon)
        else:
            out = open(path, "w")
            json.dump(jasoon, out, indent=6)
            out.close()
    
    def dark_mode(self, img=None):
        """Checks if dark mode is enabled
        
        Args:
            img: input screenshot (grayscale)
        
        Returns:
            Dark Mode enabled? True / False
        """
        if img is None: img = self.img
        dim = img.shape
        img = img[:, :int(dim[0]*0.02)]
        # show_image(img)
        avg = np.average(img)
        return avg < 150
    
    # previous version
    def analyse2(self, img=None):
        """Do Tagesschau
        """
        if img is None: img = self.img
        
        self.dark_mode(img)
        head, no_head = self.header(img)
        
        info, body = self.info_split(no_head)
        
        head_ocr_result = self.ocr_cleanup(self.ocr(head))
        info_ocr_result = self.ocr_cleanup(self.ocr(info))
        body_ocr_result = self.ocr_cleanup(self.ocr(body))
        
        info_ocr_split = info_ocr_result.split("Stand:")
        print(info_ocr_split)
        if info_ocr_split[1][0] == " ": info_ocr_split[1] = info_ocr_split[1][1:]
        
        return info_ocr_split[0], head_ocr_result, body_ocr_result, info_ocr_split[1]
    
    def header(self, img=None):
        if img is None: img = self.img
        
        dim = img.shape
        
        img_og = img.copy()
        img = to_grayscale(img)
        show_image(img)
        
        # no_header, only_header = iso_grayscale(img, True, 90, False, (15, 15), True)
        # show_image(no_header)
        # show_image(only_header)
        if self.dark_mode(img):
            # no_header_exptr = expand_to_rows(no_header[:,:int(no_header.shape[1] / 4)], True, 25)
            no_header_exptr = expand_to_rows(img[:,:int(img.shape[1] / 4)], True, 25)
        show_image(no_header_exptr)
        
        slices = []
        for i in range(2, len(no_header_exptr)):
            if no_header_exptr[i][0] > 250 and no_header_exptr[i-1][0] < 100:
                slices.append(i)
        slices.append(len(img) - 1)

        parts = []

        for i in range(1, len(slices)):
            temp = img_og[slices[i-1]:slices[i]]
            # temp2 = to_grayscale(temp.copy())
            # cnt = False
            # for row in temp2:
            #     for pix in row:
            #         if pix > 253:
            #             cnt = True
            #             break
            # if not cnt: continue
            
            parts.append(temp)
        
        print(len(parts))
        for p in parts: show_image(p)
        
        return parts
    
        head = []
        no_head = []
        
        for i in range(len(img)):
            if no_header_exptr[i][0] > 100: no_head.append(img_og[i])
            else: head.append(img_og[i])
        
        head = np.array(head)
        no_head = np.array(no_head)
        
        show_image(head)
        show_image(no_head)
        
        return head, no_head
    
    def info_split(self, img=None):
        if img is None: img = self.img
        
        # img_og = img.copy()
        show_image(img)
        iso = trimm_and_blur(img[:,:int(img.shape[1] / 4),:].copy(), False, 70, (10, 10), [255, 255, 255], True, [0, 0, 0])
        show_image(iso)
        iso = expand_to_rows(iso[:,:,0], False, 5)
        show_image(iso)
        
        info = []
        body = None
        
        for i in range(len(img)):
            if iso[i][0] < 100: info.append(img[i])
            else:
                body = img[i:,:,:]
                break
        
        info = np.array(info)
        show_image(info)
        show_image(body)
        
        return info, body

class WELT(PlutoObject):
    def __init__(self, img: np.ndarray):
        super().__init__(img)
    
    def analyse(self, img=None):
        if img is None: img = self.img
        
        cat, slices = self.split(img)
        
        headline, author, date = "", "", ""
        category = self.ocr_cleanup(self.ocr(cat))
        
        for s in slices:
            ocrresult = self.ocr_cleanup(self.ocr(s))
            
            if ocrresult[:4] == "Von ": author = ocrresult[4:]
            elif ocrresult[:7] == "Stand: ": date = ocrresult[6:]
            else: headline += " " + ocrresult
        
        return headline, author, date, category
    
    def to_json(self, img=None, path=None):
        """Extracts information from screenshot and saves it as json file.
        
        Args:
            img: screenshot as np.array
            path: path to where the json file should be saved
        """
        if img is None: img = self.img
        import json
        
        headline, author, date, category = self.analyse(img)
        
        jasoon = {  "source": "WELT",
                    "category": "News Article",
                    "article": {
                        "created": date,
                        "headline": headline,
                        "category": category,
                        "author": author,
                        "author": author
                    }
                }
        
        if path == None: return json.dumps(jasoon)
        else:
            out = open(path, "w")
            json.dump(jasoon, out, indent=6)
            out.close()
    
    def split(self, img=None, display=True):
        if img is None: img = self.img
        
        img_og = img.copy()
        
        img = to_grayscale(img)
        # # show_image(img)
        
        images, img = self.images(img)
        
        img = expand_to_rows(img, True, 30, False)
        # show_image(img)
        
        category = []
        sliceindx = []
        slices = []
        
        for i in range(len(img)):
            if img[i][0] == 0: break
        category = img_og[:i]
        
        sliceindx.append(i)
        for j in range(i+1, len(img)):
            if img[j-1][0] < 100 and img[j][0] > 250: sliceindx.append(j)
        
        # print(sliceindx)
        for i in range(1, len(sliceindx), 1):
            if sliceindx[i] - sliceindx[i-1] < 5: continue
            slices.append(img_og[sliceindx[i-1]:sliceindx[i]])
        
        return category, slices
    
    def images(self, img=None):
        if img is None: img = self.img
    
        image = []
        non_image = []
        
        for i in range(len(img)):
            if img[i][0] > 250:
                non_image.append(img[i])
            else: image.append(img[i])
        
        return np.array(image), np.array(non_image)

class Discord(PlutoObject):
    def __init__(self, img: np.ndarray):
        super().__init__(img)
    
    def analyse(self, img=None):
        if img is None: img = self.img
        og_img = img.copy()
        
        img = to_grayscale(img)
        
        img = np.transpose(img, (1, 0))
        
        for i in range(len(img)):
            for j in range(len(img[i])):
                if img[i][j] > 117 and img[i][j] < 123: img[i][j] = 59
        
        # show_image(img)
        exptr = expand_to_rows(img.copy(), False, 100)
        # show_image(exptr)
        for i in range(1, len(exptr)):
            if exptr[i-1][0] > 200 and exptr[i][0] < 200: break
        
        img = img[i:]
        indx = i
        img = np.transpose(img, (1, 0))
        # show_image(img)
        
        for i in range(len(img)):
            for j in range(len(img[i])):
                if img[i][j] < 253: img[i][j] = 0
        
        # show_image(img)
        
        imgb = cv2.blur(img.copy(), (10, 10))
        
        # show_image(imgb)
        
        iso1 = expand_to_rows(img.copy(), True, 254)
        iso1 = self.fix_slices(iso1)
        # show_image(iso1)
        slices = self.slice_chat(og_img[:, indx:] , iso1)
        out = []
        
        for i in slices:
            # show_image(i)
            info, body = self.split(i)
            name, info = self.nameinfo(info)
            name_ocr = self.ocr_cleanup(self.ocr(name))
            info_ocr = self.ocr_cleanup(self.ocr(info))
            body_ocr = self.ocr_cleanup(self.ocr(body))
            # show_image(name)
            # show_image(info)
            # show_image(body)
            out.append([name_ocr, info_ocr, body_ocr])
        
        return out
    
    def fix_slices(self, img=np.ndarray):
        """Fix mini slices
        """
        new_img = img.copy()
        slices = []
        
        for i in range(2, len(img)):
            if img[i][0] > 250 and img[i-1][0] < 200:
                slices.append(i)
        # print(slices)
        
        for i in range(1, len(slices)):
            if (slices[i] - slices[i-1]) < 10:
                fix = (255 - np.zeros(((slices[i] - slices[i-1]), len(img[0])), np.uint8))
                new_img[slices[i-1]:slices[i]] = fix
        
        # show_image(new_img)
        return new_img
    
    def to_json(self, img=None, path=None):
        if img is None: img = self.img.copy()
        import json
        msg = self.analyse(img)
        
        jasoon = {  "source": "Discord",
                    "category": "Chat",
                    "messages": msg
                }
        
        if path == None: return json.dumps(jasoon)
        else:
            out = open(path, "w")
            json.dump(jasoon, out, indent=6)
            out.close()
    
    def remove_usericon(self, img=None):
        if img is None: img = self.img
        
        img = np.transpose(img, (1, 0))
        # show_image(img)
        img_exptr = expand_to_rows(img.copy(), True, 75)
        
        out = []
        
        for i in range(1, len(img_exptr)):
            if img_exptr[i-1][0] > 230 and img_exptr[i][0] < 230:
                out = img[i:]
                break
        
        out = np.transpose(out, (1, 0))
        return out
    
    def slice_chat(self, img=np.ndarray, mark=np.ndarray):
        """Slices a chat screenshot into images of one message
        """
        slices = []
        out = []
        
        for i in range(2, len(mark)):
            if mark[i][0] > 250 and mark[i-1][0] < 200:
                slices.append(i)
        slices.append(len(mark))
        # print(slices)
        
        prev = 10
        for i in range(1, len(slices)):
            if slices[i-1] - prev < 1: prev = 0
            temp = img[slices[i-1]-prev:slices[i]-prev]
            # show_image(temp)
            # try:
            #     temp = self.remove_usericon(temp)
            # except Exception as e: print(e)
            # print(type(temp))
            # show_image(temp)
            
            out.append(temp)
        
        return out
    
    def split(self, img=None):
        """Splits a chat slice into header (username, date) and content (text)
        """
        if img is None: img = self.img
        
        img_og = img.copy()
        img = to_grayscale(img)
        img = expand_to_rows(img[:,:int(img.shape[1] / 2)], False, 250)
        
        for i in range(len(img)-1, 1, -1):
            if img[i-1][0] > 230 and img[i][0] < 230:
                img = img_og[:i]
                img2 = img_og[i:]
                break
        
        # show_image(img)
        # show_image(img2)
        
        return img, img2
    
    def nameinfo(self, img=None):
        """Splits the info part of a chat slice into name and date (info) parts 
        """
        if img is None: img = self.img
        og_img = img.copy()
        img = to_grayscale(img)
        
        middle = int((img.shape[0] + 0.5) / 2)
        name = None
        info = None
        
        for i in range(len(img[0])-5, 0, -1):
            for h in range(len(img)):
                if img[h][i] > 150:
                    # print(i, h)
                    name = og_img[:,:i+5]
                    info = og_img[:,i+5:]
                    return name, info

class FBM(PlutoObject):
    def __init__(self, img: np.ndarray):
        super().__init__(img)
        self.img = img
    
    def analyse(self, img=None):
        """Main method for extractiong messages from a FB Messenger chat screenshot
        """
        if img is None: img = self.img
        
        slices = self.slice(img, self.darkmode(img))
        msg = []
        
        for slc in slices:
            io = self.io_classification(slc)
            try:
                message = self.ocr_cleanup(self.ocr(slc))
                if io == 0: msg.append(["received", message])
                else: msg.append(["send", message])
            except Exception as e: print(e)
        return msg
    
    def analyse_light(self, img=None):
        if img is None: img = self.img
        
        # show_image(cv2.resize(img, (500, 500)))
        
        dim = img.shape
        gray = to_grayscale(img.copy())
        
        slices = []
        for i in range(1, len(gray)):
            if (gray[i][int(dim[1] / 4)] < 5 and gray[i-1][int(dim[1] / 4)] > 5) or \
            (gray[i][int(dim[1] / 4)] > 5 and gray[i-1][int(dim[1] / 4)] < 5) or \
            (gray[i][int(dim[1] / 5 * 4)] > 5 and gray[i-1][int(dim[1] / 5 * 4)] < 5) or \
            (gray[i][int(dim[1] / 5 * 4)] < 5 and gray[i-1][int(dim[1] / 5 * 4)] > 5): slices.append(i)
        
        slices.append(len(gray)-1)
        print(slices)
        msg = []
        
        for i in range(1, len(slices), 1):
            the_slice = (gray[slices[i-1] : slices[i]])
            the_slice = self.row_filter_recived(the_slice)
            if the_slice is not None:
                # show_image(the_slice)
                try:
                    message = self.ocr_cleanup(self.ocr(the_slice))
                    sor = self.send_or_recived(the_slice)
                    if sor == 0: continue
                    elif sor == 1: msg.append(["send", message])
                    else: msg.append(["received", message])
                except Exception: pass
        # print(msg)
        return msg
    
    def io_classification(self, img=None):
        """Send or Recived?
        """
        if img is None: img = self.img
        
        net = ConvNet(3, 6, 12, 100, 50, 2)
        net.load_state_dict(torch.load("FBM models/fbm2.pt"))
        
        device = self.determine_device()
        net.to(device)
        
        tnsr = self.to_tensor(img, 224, torch.float32, device)
        
        net.eval()
        with torch.no_grad():
            net_out = net(tnsr.to(device))[0]
            predicted_class = torch.argmax(net_out)
        result = predicted_class.cpu().numpy()
        
        return result
    
    def slice(self, img=None, dm=False): #  --> List
        """Slices a screenshot of a chat into images of individual messages.
        Returns:
            A List of images
        """
        if img is None: img = self.img
        og_img = img.copy()
        
        if dm: img = to_grayscale(img)
        else: img = (255 - to_grayscale(img))
        test = expand_to_rows(img, True, 5)

        slices = []
        for i in range(2, len(test)):
            if test[i][0] > 250 and test[i-1][0] < 10:
                slices.append(i)
        slices.append(len(img) - 1)

        parts = []

        for i in range(1, len(slices)):
            temp = og_img[slices[i-1]:slices[i]]
            temp2 = to_grayscale(temp.copy())
            cnt = False
            for row in temp2:
                for pix in row:
                    if pix > 253:
                        cnt = True
                        break
            if not cnt: continue
            
            parts.append(temp)
        
        return parts
    
    def slice_bright(self, img=None): #  --> List
        """Slices a screenshot of a chat into images of individual messages.
        Returns:
            A List of images
        """
        if img is None: img = self.img
        
        # show_image(img)
        test = expand_to_rows(to_grayscale(img), True, 250, False)
        # show_image(test)

        slices = []
        for i in range(2, len(test)):
            if test[i][0] > 250 and test[i-1][0] < 10:
                slices.append(i)
        slices.append(len(img) - 1)

        parts = []

        for i in range(1, len(slices)):
            temp = img[slices[i-1]:slices[i]]
            # temp2 = to_grayscale(temp.copy())
            # cnt = False
            # for row in temp2:
            #     for pix in row:
            #         if pix < 5:
            #             cnt = True
            #             break
            # if not cnt: continue
            
            parts.append(temp)
        
        return parts
    
    def show_slices(self, slcs):
        for img in slcs:
            show_image(img)
    
    def row_filter_recived(self, image=None, value=250):
        """
        Args:
            image: An grayscale image as np.ndarray, which represents a mask.
        
        Returns:
            A np.ndarray of the edited image.
        """
        dimensions = image.shape
        imglen = dimensions[0]
        out = []
        for i in range(int(imglen)):
            for j in range(dimensions[1]):
                if image[i][j] > value:
                    out.append(image[i])
                    break
        out = np.array(out)
        if out.shape[0] < 1 or out.shape[1] < 1: return None
        else:
            middle = int(out.shape[0] / 2)
            for i in range(len(out[0])-1, 11, -1):
                if out[middle][i] > 10: break
            for j in range(len(out[0])):
                if out[middle][j] > 10:
                    return out[:,j:i]
        # return out
    
    def send_or_recived(self, img=None):
        """Send or Recived? 0 for invalid, 1 for send, 2 for recived
        """
        if img is None: img = self.img
        
        img_exptr = np.transpose(img.copy(), (1, 0))
        img = np.transpose(img, (1, 0))
        img_exptr = expand_to_rows(img_exptr, True, 250)
        # show_image(img_exptr)
        
        out = []
        for i in range(len(img_exptr)):
            if img_exptr[i][0] > 250: out.append(img[i])
        out = np.array(out)
        # show_image(out)
        
        out = np.reshape(out, (-1))
        avg, num = 0, 0
        for pixval in out:
            if pixval > 200: continue
            else:
                avg += pixval
                num += 1
        
        if avg / num < 48: return 0
        elif avg / num > 95: return 1
        else: return 2
    
    def darkmode(self, img):
        """Checks if dark mode is enabled
        
        Args:
            img: input screenshot (optional)
        
        Returns:
            Dark Mode enabled? True / False
        """
        if img is None: img = self.img
        dim = img.shape
        img = img[:int(dim[0]*0.1),:int(dim[0]*0.02)]
        avg = np.average(img)
        return avg < 200
    
    def to_json(self, img=None, path=None):
        if img is None: img = self.img.copy()
        import json
        msg = self.analyse(img)
        
        jasoon = {  "source": "Facebook Messenger",
                    "category": "Chat",
                    "messages": msg
                }
        print(jasoon)
        
        if path == None: return json.dumps(jasoon)
        else:
            out = open(path, "w")
            json.dump(jasoon, out, indent=6)
            out.close()

class WhatsApp(PlutoObject):
    def __init__(self, img: np.ndarray):
        super().__init__(img)
        self.img = None
    
    def analyse(self, img=None):
        """Main method for extractiong messages from a WhatsApp chat screenshot
        """
        if img is None: img = self.img
        
        slices = self.sliceit(img)
        msg = []
        
        for slc in slices:
            io = self.io_classification(slc)
            try:
                message = self.ocr_cleanup(self.ocr(slc))
                if io == 1: msg.append(["received", message])
                else: msg.append(["send", message])
            except Exception as e: print(e)
        return msg
    
    def to_json(self, img=None, path=None):
        if img is None: img = self.img.copy()
        import json
        msg = self.analyse(img)
        
        jasoon = {  "source": "WhatsApp",
                    "category": "Chat",
                    "messages": msg
                }
        
        if path == None: return json.dumps(jasoon)
        else:
            out = open(path, "w")
            json.dump(jasoon, out, indent=6)
            out.close()
    
    def sliceit(self, img=None):
        """Slices the image into messages
        """
        img = img[:, int(img.shape[1] / 50) : int(img.shape[1] - (img.shape[1] / 50))]

        exptr = expand_to_rows(to_grayscale(img.copy()), True, 45)

        slice_indx = []

        for i in range(len(exptr)):
            if exptr[i][0] > 250 and exptr[i-1][0] < 100 or \
            exptr[i][0] < 250 and exptr[i-1][0] > 100:
                slice_indx.append(i)

        slices = []

        for i in range(1, len(slice_indx), 2):
            if slice_indx[i] - slice_indx[i-1] < 5: continue
            slices.append(img[slice_indx[i-1]:slice_indx[i]])
    
        return slices
    
    def io_classification(self, img=None):
        """Send or Recived?
        """
        net = ConvNet(3, 6, 12, 300, 20, 2)
        net.load_state_dict(torch.load("models/wa1.pt"))

        util = PlutoObject(None)
        device = util.determine_device()
        net.to(device)
        
        tnsr = util.to_tensor(img, 224, torch.float32, device)
    
        net.eval()
        with torch.no_grad():
            net_out = net(tnsr.to(device))[0]
            predicted_class = torch.argmax(net_out)
        result = predicted_class.cpu().numpy()
        
        return result

# cli execution
if __name__ == "__main__":
    try:
        img = None
        if arg_i is None: img = grab_clipboard()
        else: img = read_image(arg_i)
        show_image(img)
        
        if arg_c == "NYT":
            NYT(img).to_json(img, arg_o)
        elif arg_c == "Tagesschau":
            Tagesschau(img).to_json(img, arg_o)
        elif arg_c == "WPost":
            WPost(img).to_json(img, arg_o)
        elif arg_c == "WELT":
            WELT(img).to_json(img, arg_o)
        elif arg_c == "FoxNews":
            FoxNews(img).to_json(img, arg_o)
        elif arg_c == "Discord":
            Discord(img).to_json(img, arg_o)
        elif arg_c == "Facebook":
            Facebook(img).to_json(img, arg_o)
        elif arg_c == "FBM":
            FBM(img).to_json(img, arg_o)
        elif arg_c == "WhatsApp":
            WhatsApp(img).to_json(img, arg_o)
        elif arg_c == "Discord":
            Discord(img).to_json(img, arg_o)
        elif arg_c == "WELT":
            WELT(img).to_json(img, arg_o)
        elif arg_c == "WPost":
            WPost(img).to_json(img, arg_o)
    
    except Exception: pass

class ConvStage(nn.Module):
    """Two convolutional layers with batch norm & relu
    """
    def __init__(self, in_channels, out_channels):
        super(ConvStage, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),)

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    """UNET model.
    Based on: https://arxiv.org/abs/1505.04597
    
    Args:
        in_channel: input channels, default is 3 for color images
        out_channel: segmentation mask output channels, default is 1 for grayscale mask
        features: feature dimensions for the conv stages
    
    Disclaimer:
        Parts of this class have been forked from\
        https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/model.py
        Copyright (c) 2020 Aladdin Persson
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # For each feature a conv stage is created (down part)
        for feature in features:
            self.downs.append(ConvStage(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(ConvStage(feature*2, feature))

        self.bottleneck = ConvStage(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = tf.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

class ConvNet(nn.Module):
    """Basic Convolutional Neural Network for image classification.
    2 convolutional layer + 3 linear layers
    
    Args:
        conv1_out: Output channels for the first conv layer
        conv2_out: Output channels for the second conv layer
        fc1_out: Output channels for the first fully connected (linear) layer
        fc2_out: Output channels for the second fully connected (linear) layer
        fc3_out: Output channels for the third fully connected (linear) layer, corresponding to the ammount of classes 
    
    Disclaimer:
        Parts of this class have been forked from\
        https://github.com/Patzold/Jugend-Forscht-2021-Code
    """
    def __init__(self, conv1_in: int, conv1_out: int, conv2_out: int, fc1_out: int, fc2_out: int, fc3_out: int):
        super().__init__()
        self.conv1 = nn.Conv2d(conv1_in, conv1_out, 2)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, 2)
        self.dropout = nn.Dropout(0.8)
        
        x = torch.randn(224,224,conv1_in).view(-1,conv1_in,224,224)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, fc1_out) #flattening.
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, fc3_out)

    def convs(self, x):
            c1 = self.conv1(x)
            relu1 = F.relu(c1)
            pool1 = F.max_pool2d(relu1, (2, 2))
            c2 = self.conv2(pool1)
            relu2 = F.relu(c2)
            pool2 = F.max_pool2d(relu2, (2, 2))
            
            if self._to_linear is None:
                self._to_linear = pool2[0].shape[0]*pool2[0].shape[1]*pool2[0].shape[2]
            return pool2

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x