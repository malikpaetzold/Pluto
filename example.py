import pluto as pl

# path = "NYT_Example_1.jpg"
path = "E:/Datasets/2021 q1/screenshots/11.jpg"
pl.use_tesseract = True
pl.tesseract_path = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

img = pl.input_image(path)

pl.show_image(img)

# pl.nyt(img)
screenshot1 = pl.Twitter(img)

username, handle = screenshot1.get_header()

screenshot1.open_account()