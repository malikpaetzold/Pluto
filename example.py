import pluto as pl

from pluto import FoxNews, Twitter, Facebook, NYT

# load and show image
dir = "example images/"
path = dir + "1.jpg"
img = pl.read_image(path)
pl.show_image(img)

# create an pluto object for Fox News articles, analyse screenshot
foxarticle = FoxNews(img)
foxarticle.tesseract_path = "C:/Program Files/Tesseract-OCR/tesseract.exe" # This is the default path for all classes. If it's identical to your system, this step can be skipped.

result = foxarticle.analyse() # Depending on your system, this step can take a while
print(result)

input("\nPress any key to continue to the next example   ->")

img = pl.read_image(dir + "6.jpg")
pl.show_image(img)

post = Facebook(img)
result = post.analyse() # again, this step might take a few seconds
print(result)

# we can also take a look at the different parts of the screenshot, after it has been segmentated
top = post.top
pl.show_image(top)

# let's get that in writing using EasyOCR
print(post.ocr(top, True))

input("\nPress any key to continue to the next example   ->")

# as you can see, the overall process is pretty straightforward
img = pl.read_image(dir + "NYT_Example_3.jpg")
pl.show_image(img)

result = NYT(img).analyse()
print(result)

# if we only want to segmentate the headline instead of the hole image, we can do that as well
article = pl.NYT(img)

# this gets you the headline, but unfotunitly parts of the image as well
headline = article.header()
pl.show_image(headline)

image, headline = article.images(headline, True)
pl.show_image(image)
pl.show_image(headline)