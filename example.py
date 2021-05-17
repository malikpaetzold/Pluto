import pluto as pl

path = "NYT_Example.jpg"
pl.use_tesseract = True
pl.tesseract_path = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

img = pl.input_image(path)

pl.show_image(img)

pl.nyt(img)