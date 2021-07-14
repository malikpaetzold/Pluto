# Requirements:
#
# You will need to have NumPy, Matplotlib, Cv2, and PyTesseract installed.

import pluto as pl
from pluto import FoxNews

# load and show image
path = "FoxNews_Example_1.jpg"
img = pl.read_image(path)
pl.show_image(img)

# create an pluto object for Fox News articles, analyse screenshot
foxarticle = FoxNews(img)
foxarticle.tesseract_path = "C:/Program Files/Tesseract-OCR/tesseract.exe" # This is just the default path. You may need to change it for your system.

print(foxarticle.analyse())