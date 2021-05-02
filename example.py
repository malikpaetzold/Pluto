import pluto as pl
import matplotlib.pyplot as plt
import numpy as np

# path = "E:/Datasets/2021 q1/screenshots/11.jpg"
path = "E:/Datasets/2021 q1/orig_screenshots/Screenshot_20210410-022056_Twitter.jpg"
pl.use_tesseract = True
pl.tesseract_path = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

img = pl.input_image(path)

pl.show_image(img)

pl.get_header(img)
# pl.get_text(img)
pl.get_stats(img)