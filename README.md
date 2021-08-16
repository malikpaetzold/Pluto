# Pluto
Pluto is a Python library for working with & analyzing screenshots.

![Explainer_1](https://user-images.githubusercontent.com/60754058/119409979-62be3800-bce8-11eb-9f5c-60d623d065b3.png)

The core idea is to extract information from a screenshot and then provide that information in an orderly fashion. This opens the door to many more possibilities, both for applications and analytical pipelines.

![ezgif com-gif-maker (1)](https://user-images.githubusercontent.com/60754058/129491738-09c28f3e-0b52-49a1-8624-08f0a42bc8cd.gif)

How exactly does it all work? The text is extracted using an OCR library (optical character recognition). However, the more interesting part is the correct assignment, meaning  
for example whether the extracted text is still part of the username or already the content of a post. The assignment takes place using the help of AI-supported computer vision, 
or more 'traditional'methods in the light version.

Please note that Pluto is still in development, and many features are not built yet or only exist as early access. Pluto is supposed to be able to process screenshots from social 
media posts, chat histories, and news pages. Current features of the master branch include the Twitter, Fox News and Facebook feature, as well as an early NYT feature and many 
underlying helper functions for OCR and image transforms. Take a look at example.py to see how the current version of Pluto can be used.

# Quickstart
Download ```pluto.py``` or ```pluto_light.py``` (Pluto is not yet available via pip). When using ```pluto.py```, you will also need to download the models folder. Head to 
downloads.patzold.io and download the models.zip file. Then un-zip it and place it in the same directory as ```pluto.py```. ```example.py``` has a few code examples on how to use 
the different features.

Please make sure to have all the necessary dependencies installed. ```pluto_light.py``` requires Matplotlib, Cv2, NumPy and PyTesseract, ```pluto.py``` additionally also PyTorch 
and EasyOCR.
