# Pluto
# v0.9.0

# MIT License
# Copyright (c) 2021 Malik Pätzold

from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
import torch.nn.functional as F

import time
import webbrowser
import requests

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

# try:
    # print(arg_i)
    # print(arg_o)
    # print(arg_c)
# except Exception: pass

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

def show_image(image: np.ndarray, BGR2RGB=False):
    """Displays an image using Matplotlib's pyplot.imshow()
    
    Args:
        image: The image to be displayed.
        BGR2RGB: When True, the color space is converted from BGR to RGB.
    """
    if BGR2RGB: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

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

def google(query: str):
        """Googles a query. Opens result in browser window.
        """
        link = "https://www.google.de/search?q="
        query.replace(" ", "+")

        webbrowser.open((link + query))

class PlutoObject:
    def __init__(self, img: np.ndarray):
        self.img = img
        self.use_easyocr = False

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

    def ocr(self, image=None, switch_to_tesseract=False):  # -> str
        """Preforms OCR on a given image, using EasyOCR
        
        Args:
            image: np.ndarray of the to be treated image.
        
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
        
        # show_image(extr)
        
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

class FoxNews(PlutoObject):
    def __init__(self, img: np.ndarray):
        super().__init__(img)
    
    def analyse(self, display=False):
        """Extracts key information from a screenshot of a Fox News Article.
        
        Args:
            display: True if in between steps should be shown
        
        Returns:
            The extracted information as JSON
        """
        og_shape = self.img.shape
        img = cv2.resize(self.img, (512, 512))
        black = np.zeros((512, 512))

        for i in range(len(black)):
            for j in range(len(black[0])):
                temp = img[i][j]
                if (temp == [34, 34, 34]).all(): black[i][j] = 255
        blured = cv2.blur(black, (20, 20))

        for i in range(len(blured)):
            for j in range(len(blured[0])):
                if blured[i][j] < 40: blured[i][j] = 0
                else: blured[i][j] = 255

        msk = self.expand_to_rows(blured)

        og_size_msk = cv2.resize(msk, (og_shape[1], og_shape[0]))
        
        top = []
        heading = []
        bottom = []

        top_part = True
        bottom_part = False

        for i in range(len(self.img)):
            if og_size_msk[i][0] > 1:
                heading.append(self.img[i])
                if top_part:
                    top_part = False
                    bottom_part = True
            elif top_part: top.append(self.img[i])
            else: bottom.append(self.img[i])

        heading = np.array(heading)
        bottom = np.array(bottom)
        top = np.array(top)
        
        if display:
            show_image(heading)
            show_image(bottom)
            show_image(top)

        ocr_result = self.ocr(heading)
        headline = self.ocr_cleanup(ocr_result)

        cat_info_img = []
        top_len = len(top)
        for i in range(top_len, 0, -1):
            if top[i-1][0][0] > 250: cat_info_img.insert(0, top[i-1])
            else: break

        cat_info_img = np.array(cat_info_img)
        if display: show_image(cat_info_img)

        ocr_result = self.ocr(cat_info_img)
        clean_ocr = self.ocr_cleanup(ocr_result)

        dotsplit = clean_ocr.split("-")[0][:-1].lstrip(" ")
        pubsplit = clean_ocr.split("Published")[1].lstrip(" ")
        
        subinfo_bottom = []

        stoper = False
        for row in bottom:
            subinfo_bottom.append(row)
            for pix in row:
                if pix[0] > 200 and pix[0] < 240 and pix[2] < 50 and pix[1] < 50:
                    stoper = True
                    break
            if stoper: break

        subinfo_bottom = np.array(subinfo_bottom[:-3])
        if display: show_image(subinfo_bottom)
        subinfo = self.ocr_cleanup(self.ocr(subinfo_bottom))

        subsplit = subinfo.split()

        author_list = []
        subtitle_list = []
        subinfo_switcher = True

        for w in reversed(subsplit):
            if w == "By" and subinfo_switcher:
                subinfo_switcher = False
                continue
            if w == "News" or w == "Fox" or w == "|": continue
            if subinfo_switcher: author_list.insert(0, w)
            else: subtitle_list.insert(0, w)

        author = " ".join(author_list)
        subtitle = " ".join(subtitle_list)
        
        return pubsplit, headline, subtitle, author, dotsplit
    
    def to_json(self, img=None, path=None):
        if img is None: img = self.img.copy()
        import json
        pubsplit, headline, subtitle, author, dotsplit = self.analyse(img)
        
        jasoon = {  "source": "News Article",
                    "article": {
                        "created": "[Published] " + pubsplit,
                        "organisation": "Fox News",
                        "headline": headline,
                        "subtitle": subtitle,
                        "author": author,
                        "category": dotsplit
                    }
                }
        
        if path == None: return json.dumps(jasoon)
        else:
            out = open(path, "w")
            json.dump(jasoon, out, indent=6)
            out.close()

class Facebook(PlutoObject):
    def __init__(self, img: np.ndarray):
        super().__init__(img)
        self.header = None
        self.text = None
        self.insert = None
        self.engagement = None
    
    def analyse_legacy(self, img=None):
        """---
        Depricated
        ---
        """
        if img is None: img = self.img
        
        slc, indx = self.slices(img.copy())
        
        image, eng = self.img_eng(indx, img.copy())
        
        inpts, imgs = self.part(img[:indx].copy(), slc)
        
        header = []
        text = []
        
        for indx in range(len(inpts)):
            sl = imgs[indx]
            t = np.transpose((255 - to_grayscale(sl)), (1, 0))
            t = expand_to_rows(t, True, 10)
            # show_image(sl)
            # show_image(t)
            continue
            clss = self.classify(inpts[indx])
            prt = (imgs[indx]).tolist()
            if clss == 0: header += prt
            else: text += prt
        
        header = np.array(header, np.uint8)
        date = header[int(header.shape[0] / 2) :]
        header = header[: int(header.shape[0] / 2)]
        text = np.array(text, np.uint8)
        
        # show_image(header)
        # show_image(date)
        # show_image(text)
        
        header = self.ocr_cleanup(self.ocr(header))
        date = self.ocr_cleanup(self.ocr(date))
        text = self.ocr_cleanup(self.ocr(text))
        engagement = self.ocr(eng)
        
        return header, date, text, engagement
    
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
        
        # show_image(top)
        # show_image(insert)
        # show_image(engagement)
        
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

    def header(img):
        slc = slice(img)
        
        show_image(slc[0])
        show_image(slc[1])
        return
        header = []
        text = []
        
        for indx, s in enumerate(slc):
            print(type(s))
            f = first(s)
            if indx == 0:
                header.append(s)
                print(s.shape)
            else:
                text.append(s)
                print(s.shape)
        
        return np.array(header), np.array(text)

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
        """
        if img is None: img = self.img
        img = to_grayscale(img)
        
        net = ConvNet(1, 6, 12, 100, 20, 2)
        net.load_state_dict(torch.load("models/general_1.pt"))
        
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
        print(result.shape, img.shape)
        
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
    def __init__(self, img: np.ndarray):
        super().__init__(img)
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.header, self.bottom = None, None
    
    def split(self, img=None, display=False):
        img_og = img
        if img is None: img_og = self.img
        img_size = 256
        img = cv2.resize(img_og.copy(), (img_size, img_size))
        img_tensor = self.to_tensor(img, img_size, torch.float32, self.DEVICE)
        
        model = UNET(in_channels=3, out_channels=1)
        model = self.load_model(model, "D:/Codeing/Twitter_Segmentation/pytorch version/own_2_net_2.pt", self.DEVICE)
        
        with torch.no_grad():
            model_out = torch.sigmoid(model(img_tensor)) * 255
        
        mask = self.from_tensor(model_out, img_size)
        if display: show_image(mask)
        img = (img.reshape(img_size, img_size, 3) * 255).astype(np.uint8)
        mask = cv2.merge((mask, mask, mask))
        mask = trimm_and_blur(mask, False, 60, (30, 30), [0, 0, 0])[:,:,0]
        if display: show_image(mask)
        self.vis_model_prediction(img, mask, False)
        mask = expand_to_rows(mask, value=30)
        img = img_og
        mask = cv2.resize(mask[:,:10], (10, img.shape[0]))
        
        header = []
        bottom = []

        # split in header / bottom
        for i in range(len(mask[:,0])):#range(len(mask[:,0])-1, 0, -1):
            if mask[i][0] > 250: header.append(img[i,:])
            else: bottom.append(img[i,:])

        self.header = np.array(header)
        self.bottom = np.array(bottom)

        if display:
            show_image(self.header)
            show_image(self.bottom)
        
        return self.header.copy(), self.bottom.copy()
    
    def header_color_mode(self, img=None):
        """Determines whether the header is in dark mode
        
        Args:
            img: if the image should be different to self.header, pass it here
        
        Returns:
            True if the header is in dark mode.
        """
        if img is not None: self.header = img
        dim = self.header.shape
        img = self.header[:,int(dim[1]/2):,:]
        avg = np.average(img)
        return avg < 150
    
    def header_analyse(self, img=None, display=False):
        if img is None: img = self.header.copy()
        
        img = (img[::-1]).transpose(1, 0, 2)
        img_og = img.copy()
        
        if not self.dark_mode(img): img = trimm_and_blur(img, False, 30, (20, 20), [255, 255, 255], True, [0, 0, 0])
        else: img = trimm_and_blur(img, True, 245, (20, 20), [255, 255, 255], True, [0, 0, 0])
        if display: show_image(img)
        img = expand_to_rows(img[:,:,0], True, 5)
        
        out = []
        c = 0
        for i in range(len(img)-1, 0, -1):
            if img[i][0] > 100:
                out.append(img_og[i])
                if c == 0: c += 1
            elif c == 1: break
        
        out = np.flip((np.array(out).transpose(1, 0, 2)), (0, 1))
        
        if display:
            show_image(img)
            show_image(out)
        
        ocr_result = self.ocr_cleanup(self.ocr(out))
        usersplit = ocr_result.split("@")
        
        return usersplit[0][:-1], usersplit[1]
    
    def body_analyse(self, img=None, display=False): # --> str
        if img is None: img = self.bottom.copy()
        img_og = img.copy()
        
        if not self.dark_mode(img): img = trimm_and_blur(img, False, 30, (20, 20), [255, 255, 255], True, [0, 0, 0])
        else: img = trimm_and_blur(img, True, 245, (20, 20), [255, 255, 255], True, [0, 0, 0])
        if display: show_image(img)
        exptr = expand_to_rows(img[:,:,0], True, 20)[:,:10]
        if display: show_image(exptr)
        
        out = []
        
        for i in range(len(exptr)):
            if exptr[i][0] > 100: out.append(img_og[i])
        out = np.array(out)
        
        if display: show_image(out)
        
        return self.ocr_cleanup(self.ocr(out))
    
    def analyse_light(self, img=None):
        if img is None: img = self.img
        
        self.split()
        
        head_info = self.header_analyse()
        body_info = self.body_analyse()

        jasoon = {  "source": "Twitter",
                    "tweet": {
                        # "created_at": "[Published] " + pubsplit,
                        # "client": "Fox News",
                        "text": body_info,
                        "user": {
                            "name": head_info[0],
                            "handle": head_info[1]
                        }
                    }
                }
        
        return jasoon
    
    def analyse(self, img=None):
        if img is None: img = self.img
        
        self.split()
        
        head_info = self.header_analyse()
        body_info = self.body_analyse()
        
        return head_info[0], head_info[1], body_info
    
    def dark_mode(self, img=None):  # -> bool
        """Checks if the screenshot has dark mode enabled
        
        Args:
            img: if the checked image should be different from the self.img, pass it here
        
        Returns:
            Is the screenshot in dark mode? True / False
        """
        testimg = self.img
        if img is not None: testimg = img.copy()
        top_row = avg_of_row(testimg, 0, True)
        bottom_row = avg_of_row(testimg, -1, True)
        left_collum = avg_of_collum(testimg, 0, True)
        right_collum = avg_of_collum(testimg, -1, True)
        final_value = sum([top_row, bottom_row, left_collum, right_collum]) / 4
        return final_value < 125
    
    def analyse_light(self):
        result = None
        if self.dark_mode(): result = self.dark()
        else: result = self.std()
        header, body = result
        # show_image(header)
        # show_image(body)
        return self.ocr_cleanup(self.ocr(header)), self.ocr_cleanup(self.ocr(body))
    
    def std(self, img=None, display=False):
        input_img = None
        if img is not None: input_img = img.copy()
        else: input_img = self.img.copy()
        if img is not None: self.img = img.copy()
        
        blur = trimm_and_blur(input_img, True, 30, (20, 20), [255, 255, 255])
        out = trimm_and_blur(blur, False, 250, (5, 5), [0, 0, 0])

        msk_inv = (255 - out[:,:,0])

        out_exptr = self.expand_to_rows(msk_inv, True)

        header_info = []
        continue_please = True
        cnt = 0
        for i in range(len(out_exptr)):
            if continue_please:
                if out_exptr[i][0] < 250: continue
            else:
                if out_exptr[i][0] < 250: break
            header_info.append(self.img[i])
            continue_please = False
            # print("hey!")
        cnt = i

        bottom = []
        lastone = False
        for i in range(cnt+1, len(out_exptr), 1):
            if out_exptr[i][0] < 250:
                if lastone:
                    bottom.append(self.img[3])
                    lastone = False
                continue
            bottom.append(self.img[i])
            lastone = True
        
        header_info = np.array(header_info)
        bottom = np.array(bottom)
        
        if display:
            show_image(header_info)
            show_image(bottom)
        
        return header_info, bottom
    
    def dark(self, img=None, display=False):
        """Segmentates the input screenshot (dark mode enabled) into header and body.
        
        Args:
            img: if the screenshot should be different from self.img, pass it here.
            display: True if output should be displayed before return
        
        Returns:
            The two segmentated areas.
        """
        input_img = None
        if img is not None: input_img = img.copy()
        else: input_img = self.img.copy()
        if img is not None: self.img = img.copy()
        
        blur = trimm_and_blur(input_img, False, 230, (20, 20),[0, 0, 0])
        out = trimm_and_blur(blur, True, 10, (5, 5),[255, 255, 255])

        msk = out[:,:,0]

        out_exptr = self.expand_to_rows(msk, True)

        header_info = []
        continue_please = True
        cnt = 0
        for i in range(len(out_exptr)):
            if continue_please < 3:
                if out_exptr[i][0] < 250:
                    if continue_please == 1: continue_please += 1
                    if continue_please == 2: header_info.append(self.img[i])
                    continue
            else:
                if out_exptr[i][0] < 250: break
            
            if continue_please == 0: continue_please += 1
            if continue_please == 2: break
            header_info.append(self.img[i])
        cnt = i

        bottom = []
        lastone = False
        for i in range(cnt+1, len(out_exptr), 1):
            if out_exptr[i][0] < 250:
                if lastone:
                    bottom.append(self.img[3])
                    lastone = False
                continue
            bottom.append(self.img[i])
            lastone = True
        
        header_info = np.array(header_info)
        bottom = np.array(bottom)
        
        if display:
            show_image(header_info)
            show_image(bottom)
        
        return header_info, bottom
    
    def black(self):
        pass
    
    def header_segmentation(self, img=None, inverted=False):
        if img is None: img = self.img
        
        output = self.run_segmentation_model("Twitter Models/twitter_header_segmentation.pt", img)
        self.vis_model_prediction(img, output, True)
        
        output, nonheader = self.extr_mask_img(output, img, True)
        
        show_image(output)
        show_image(nonheader)
        
        if inverted: return output, nonheader
        return output

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
        
        sliced_result = self.slice(analyse_img)
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
        
        head, body = self.header(top, True)
        # show_image(head)
        # show_image(body)
        # return head, body
        
        self.headline = self.ocr_cleanup(self.ocr(head))
        author = self.author(bottom)
        # print(headline)
        
        # print(type(body))
        subt = self.suber(np.array(body))
        subtitle = self.ocr_cleanup(self.ocr(subt))
        
        return self.headline, subtitle, author
    
    def slice(self, img=None):
        """Slices image
        
        Returns:
            top, image, bottom
        """
        if img is None: img = self.img.copy()
        
        img_og = img.copy()
        
        # show_image(img[:, :int(len(img[0]) * 0.9)])
        img = expand_to_rows(to_grayscale(img[:, :int(len(img[0]) * 0.9)]), True, 248, False) # scroll bar removed
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
            # show_image(img_og[slices[i-1]:slices[i]])
        
        top = []
        difflen = 0
        
        for i in range(len(parts)):
            temp = parts[i]
            for row in temp:
                for pix in row:
                    minpix, maxpix = np.min(pix), np.max(pix)
                    difference = maxpix - minpix
                    if difference > 10: difflen += 1
                    if difflen > 50:
                        if self.classify(temp) == 0: return top, temp, parts[i+1:]
            top += temp.tolist()
        
        return [top]
    
    def classify(self, img=None):
        """Image or still part of text?
        """
        if img is None: img = self.img
        img = to_grayscale(img)
        
        net = ConvNet(1, 6, 12, 100, 20, 2)
        net.load_state_dict(torch.load("models/general_1.pt"))
        
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
    
    def search(self, query: str):
        """Searches a query with the NYT's search function. Opens result in browser window.
        """
        link = "https://www.nytimes.com/search?query="
        query.replace(" ", "+")
        
        webbrowser.open((link + query))
    
    def nyt_api_query(api_key, query):
        url = "https://api.nytimes.com/svc/search/v2/articlesearch.json?q={}&api-key={}".format(query, api_key)

        query = requests.get(url)
        return query.json()
    
    def open_search(self):
        self.search(self.headline)

class Tagesschau(PlutoObject):
    def __init__(self, img: np.ndarray):
        super().__init__(img)
    
    def analyse(self, img=None):
        """Do Tagesschau
        """
        if img is None: img = self.img
        
        head, no_head = self.header(img)
        
        info, body = self.info_split(no_head)
        
        head_ocr_result = self.ocr_cleanup(self.ocr(head))
        info_ocr_result = self.ocr_cleanup(self.ocr(info))
        body_ocr_result = self.ocr_cleanup(self.ocr(body))
        
        info_ocr_split = info_ocr_result.split("Stand:")
        if info_ocr_split[1][0] == " ": info_ocr_split[1] = info_ocr_split[1][1:]
        
        return info_ocr_split[0], head_ocr_result, body_ocr_result, info_ocr_split[1]
    
    def to_json(self, img=None, path=None):
        """Extracts information from screenshot and saves it as json file.
        
        Args:
            img: screenshot as np.array
            path: path to where the json file should be saved
        """
        if img is None: img = self.img.copy()
        import json
        date, headline, body, category = self.analyse(img)
        
        jasoon = {  "source": "Tagesschau",
                    "category": "News Article",
                    "article": {
                        "created": "[Published] " + date,
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
    
    def header(self, img=None):
        if img is None: img = self.img
        
        dim = img.shape
        
        to_grayscale(img)
        
        no_header = iso_grayscale(img, True, 90, True, (15, 15))
        no_header_exptr = expand_to_rows(no_header[:,:int(no_header.shape[1] / 4)], True, 5)
        
        head = []
        no_head = []
        
        for i in range(len(img)):
            if no_header_exptr[i][0] > 100: no_head.append(img[i])
            else: head.append(img[i])
        
        head = np.array(head)
        no_head = np.array(no_head)
        
        return head, no_head
    
    def info_split(self, img=None):
        if img is None: img = self.img
        
        # img_og = img.copy()
        
        iso = trimm_and_blur(img[:,:int(img.shape[1] / 4),:].copy(), False, 70, (10, 10), [255, 255, 255], True, [0, 0, 0])
        iso = expand_to_rows(iso[:,:,0], False, 5)
        # show_image(iso)
        
        info = []
        body = None
        
        for i in range(len(img)):
            if iso[i][0] < 100: info.append(img[i])
            else:
                body = img[i:,:,:]
                break
        
        info = np.array(info)
        # show_image(info)
        # show_image(body)
        
        return info, body

class WPost(PlutoObject):
    def __init__(self, img: np.ndarray):
        super().__init__(img)
    
    def analyse(self, img=None):
        if img is None: img = self.img
        
        category, headline, img_bottom = self.category(img)
        # show_image(img_bottom)
        
        author, body = self.author(img_bottom)
        # show_image(body)
        date, body = self.date(body)
        
        return category, headline, author, date, body
    
    def to_json(self, img=None, path=None):
        """Extracts information from screenshot and saves it as json file.
        
        Args:
            img: screenshot as np.array
            path: path to where the json file should be saved
        """
        if img is None: img = self.img.copy()
        import json
        
        category, headline, author, date, body = self.analyse(img)
        
        jasoon = {  "source": "Washington Post",
                    "category": "News Article",
                    "article": {
                        "created": "[Published] " + date,
                        "author": author,
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
    
    def category(self, img=None, do_ocr=True, display=False):
        if img is None: img = self.img
        
        if display: show_image(img)
        color, img_top, img_bottom = self.images(img, True)
        if display:
            show_image(color)
            show_image(img_top)
            show_image(img_bottom)
        
        iso_top = iso_grayscale(img_top.copy(), False, 230, True, (10, 10))
        # show_image(iso_top)
        iso_expt = expand_to_rows(iso_top[:, :int(iso_top.shape[1] / 3)], True, 20)
        # show_image(iso_expt)
        
        category = []
        headline = []
        for i in range(len(iso_expt)):
            if iso_expt[i][0] > 100: category.append(img_top[i])
            else: headline.append(img[i])
        
        category = np.array(category)
        headline = np.array(headline)
        
        # show_image(category)
        # show_image(headline)
        
        if do_ocr: return self.ocr_cleanup(self.ocr(category)), \
                        self.ocr_cleanup(self.ocr(headline)), img_bottom
        return category, headline, img_bottom
    
    def author(self, img=None, do_ocr=True, display=False):
        if img is None: img = self.img
        
        iso = iso_grayscale(img, False, 235, True, (10, 10))
        # show_image(iso)
        iso = expand_to_rows(iso, True, 10)
        # show_image(iso)
        
        out = []
        body = []
        t = None
        for i in range(len(img)-2, 0, -1):
            if iso[i+1][0] < 100 and iso[i][0] > 100:
                body = img[i:]
                t = i
            elif iso[i+1][0] > 100 and iso[i][0] < 100: break
        out = img[i:t]
        
        out = np.array(out)
        body = np.array(body)
        
        if display:
            show_image(out)
            show_image(body)
        
        if do_ocr:
            ocr_result = self.ocr_cleanup(self.ocr(out))
            ocr_result = ocr_result[3:].replace(" and", ",")
            return ocr_result, body

        return out, body
    
    def date(self, img=None, do_ocr=True, display=False):
        if img is None: img = self.img
        
        iso = iso_grayscale(img, False, 180, True, (10, 10))
        # show_image(iso)
        iso = expand_to_rows(iso, True, 10)
        # show_image(iso)
        
        body = []
        date = []
        
        for i in range(1, len(img), 1):
            if iso[i][0] < 200 and iso[i-1][0] > 200: break
        
        out = img[:i]
        body = img[i:]
        
        if display:
            show_image(out)
            show_image(body)
        
        if do_ocr: return self.ocr_cleanup(self.ocr(out)), self.ocr_cleanup(self.ocr(body))
        return out, body
    
    def images(self, img=None, non_images=False): # -> np.ndarary | None
        """Isolates images of an article, based on color (WPost version)
        
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
        non_image_top = []
        non_image_bottom = []
        
        j = 0
        stop = False
        color_start = False
        
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
            
            if stop: color_start = True
            
            if not stop and non_images:
                if color_start: non_image_bottom.append(img_og[i])
                else: non_image_top.append(img_og[i])
            j = 0
        
        image = np.array(image)
        if non_images:
            non_image_top = np.array(non_image_top)
            non_image_bottom = np.array(non_image_bottom)

        if len(image) < 1:
            image = None
            print("Pluto WARNING - At least one return from images() is empty.")
        
        if non_images: return image, non_image_top, non_image_bottom
        return image

class Bild(PlutoObject):
    def __init__(self, img: np.ndarray):
        super().__init__(img)

class Spiegel(PlutoObject):
    def __init__(self, img: np.ndarray):
        super().__init__(img)
        self.headline = False
        self.subtitle = False
    
    def analyse(self, img=None):
        if img is None: img = self.img
        
        category, header, bottom = self.split()
        
        date_img = self.bottom(bottom)
        
        headline_img, subtitle_img = self.header_split(header)
        
        headline = self.ocr_cleanup(self.ocr(headline_img))
        subtitle = self.ocr_cleanup(self.ocr(subtitle_img))
        date = self.ocr_cleanup(self.ocr(date_img))
        
        return self.ocr_cleanup(self.ocr(category)), headline, subtitle, date
    
    def to_json(self, img=None, path=None):
        """Extracts information from screenshot and saves it as json file.
        
        Args:
            img: screenshot as np.array
            path: path to where the json file should be saved
        """
        if img is None: img = self.img.copy()
        import json
        
        category, headline, subtitle, date = self.analyse(img)
        
        jasoon = {  "source": "Spiegel",
                    "category": "News Article",
                    "article": {
                        "created": "[Published] " + date,
                        "headline": headline,
                        "subtitle": subtitle,
                        "category": category
                    }
                }
        
        if path == None: return json.dumps(jasoon)
        else:
            out = open(path, "w")
            json.dump(jasoon, out, indent=6)
            out.close()
    
    def split(self, img=None, display=False):
        if img is None: img = self.img
        
        image, img = self.images(img)
        
        gray = to_grayscale(img[:, :int(img.shape[1] / 2), :])
        
        header = []
        top_header = []
        bottom_header = []
        
        gray = expand_to_rows(gray, True, 10, False)
        
        pntr = 0
        pntr2 = len(gray)-1
        while gray[pntr][0] != 0:
            pntr += 1
            top_header.append(img[pntr])
        
        while gray[pntr2][0] != 0:
            pntr2 -= 1
            bottom_header.insert(0, img[pntr2])
        
        top_header = np.array(top_header)
        header = img[pntr:pntr2]
        bottom_header = np.array(bottom_header)
        
        if display:
            show_image(top_header)
            show_image(header)
            show_image(bottom_header)
        
        return top_header, header, bottom_header
    
    def header_split(self, img=None, display = False):
        if img is None: img = self.img
        
        img_og = img.copy()
        
        img = cv2.resize(to_grayscale(img), (600, 600))
        img = cv2.blur(img, (40, 40))
        if display: show_image(img)
        
        img = expand_to_rows(img, True, 100, False)
        img = iso_grayscale(img, True, 10, False, (50, 50))
        if display: show_image(img)
        
        for i in range(len(img)-1, 0, -1):
            if img[i][0] > 5: break
        
        headline = img_og[:i+20]
        subtitle = img_og[i+20:]
        
        self.headline = headline
        self.subtitle = subtitle
        
        if display:
            show_image(headline)
            show_image(subtitle)
        
        return headline, subtitle
    
    def images(self, img=None):
        if img is None: img = self.img
        
        image = []
        non_image = []
        
        for i in range(len(img)):
            pix = img[i][0]
            if pix[0] > 250 and pix[1] > 250 and pix[2] > 250:
                non_image.append(img[i])
            else: image.append(img[i])
        
        return np.array(image), np.array(non_image)
    
    def bottom(self, img=None):
        if img is None: img = self.img
        
        img_og = img.copy()
        
        img = to_grayscale(img)
        img = expand_to_rows(img, True, 200, False)
        
        out = []
        for i in range(5, len(img)):
            if img[i][0] == 0: break
        
        for j in range(i, len(img)):
            if img[j][0] == 0:
                out.append(img_og[j])
            else: break
        
        return np.array(out)

class WELT(PlutoObject):
    def __init__(self, img: np.ndarray):
        super().__init__(img)
    
    def analyse(self, img=None):
        if img is None: img = self.img
        
        headline_img, category_img, date_img = self.split()
        
        headline = self.ocr_cleanup(self.ocr(headline_img))
        category = self.ocr_cleanup(self.ocr(category_img))
        date = self.ocr_cleanup(self.ocr(date_img))
        
        return headline, category, date
    
    def to_json(self, img=None, path=None):
        """Extracts information from screenshot and saves it as json file.
        
        Args:
            img: screenshot as np.array
            path: path to where the json file should be saved
        """
        if img is None: img = self.img.copy()
        import json
        
        headline, category, date = self.analyse(img)
        
        jasoon = {  "source": "WELT",
                    "category": "News Article",
                    "article": {
                        "created": "[Published] " + date,
                        "headline": headline,
                        "category": category
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
        show_image(img)
        
        images, img = self.images(img)
        
        img = expand_to_rows(img, True, 5, False)
        show_image(img)
        
        category = []
        date = []
        
        for i in range(len(img)):
            if img[i][0] == 0: break
            else: category.append(img_og[i])
        
        for j in range(len(img)-1, 0, -1):
            if img[j][0] == 0: break
        
        for t in range(j, 0, -1):
            if img[t][0] > 2: break
            else: date.insert(0, img_og[t])
        date.insert(0, img_og[t-1])
        
        img = img_og[i-10:t-5]
        category = np.array(category)
        date = np.array(date)
        
        if display:
            show_image(img)
            show_image(category)
            show_image(date)
        
        return img, category, date
    
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
        self.img = None
    
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
        net.load_state_dict(torch.load("models/fbm2.pt"))
        
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
        net.load_state_dict(torch.load("models/wa.pt"))
        
        device = self.determine_device()
        net.to(device)
        
        tnsr = self.to_tensor(img, 224, torch.float32, device)
        
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
        elif arg_c == "Facebook":
            Facebook(img).to_json(img, arg_o)
        elif arg_c == "FBM":
            FBM(img).to_json(img, arg_o)
        elif arg_c == "WhatsApp":
            WhatsApp(img).to_json(img, arg_o)
    
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