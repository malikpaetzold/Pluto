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
Download ```pluto.py``` and the ```models``` folder. Make sure you put them is the same directory. After installing all dependencies, you can use Pluto as a CLI or Python Library. For example:

```python pluto.py -i NYT_Example_3.jpg -o nytout.json -c NYT```

Will run the file NYT_Example_3.jpg with the NYT class and save the output as nytout.json. Type ```python pluto.py -h``` for more details.

You can also import ```pluto.py``` as a library, and use all of Pluto's functions & methods.

In both cases I highly recommend going through ```example.ipynb``` to get a better understanding of the software.

Furthermore, you can also use Pluto with a Graphical User Interface. For this option download pluto_gui.py additionally or get a .exe file from [here](https://downloads.patzold.io/).
