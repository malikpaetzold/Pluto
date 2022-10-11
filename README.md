# Pluto
Pluto is a Python library for working with & analyzing screenshots.

![Explainer_1](https://user-images.githubusercontent.com/60754058/119409979-62be3800-bce8-11eb-9f5c-60d623d065b3.png)

The core idea is to extract usefull information from a screenshot and to return that information in a machine-searchable format like JSON. This opens the door to many more possibilities, both for applications and analytical pipelines.

How exactly does it all work? The text is extracted using an OCR library (optical character recognition). However, the more interesting part is the correct assignment, meaning, for example, whether the extracted text is still part of the username or already the content of a post. The assignment takes place using the help of AI-supported computer vision and some more specialized techniques. The latest version is based on a object-detection approach using the YOLOv5 architecture.

![0](https://user-images.githubusercontent.com/60754058/194968146-e96de683-a2f4-44b1-9e37-361b7acbb519.jpg)

Please note that Pluto is still in development, and some features are not built yet or only exist as early access (see current limitations). Pluto can process screenshots from social media posts, chat histories, and news pages. Current features of the master branch include the Twitter, Facebook, New York Times, WhatsApp, and Facebook Messenger feature, as well as many underlying helper functions for OCR and image transforms. Pluto also has some generalized detection models for use in customized applications available. Take a look at ```example.ipynb``` to see how the current version of Pluto can be used.

# Quickstart
Download ```pluto.py``` and the ```models``` folder. Make sure you put them is the same directory. After installing all dependencies, you can use Pluto as a CLI or Python Library. For example:

```python pluto.py -i NYT_Example_3.jpg -o nytout.json -c NYT```

Will run the file NYT_Example_3.jpg with the NYT class and save the output as nytout.json. Type ```python pluto.py -h``` for more details.

You can also import ```pluto.py``` as a library, and use all of Pluto's functions & methods.

In both cases I highly recommend going through ```example.ipynb``` to get a better understanding of the software.

# How to get good results & current limitations
In recent versions, Pluto was based around a classification approach. In the newest version, the v1.0.0 release, I'm introducing a new approach based around object-detection that will be the focus going forward. Pluto uses a modified YOLOv5 object-detection model called [YOLOv5-slim](https://github.com/Patzold/yolov5-slim). Currently the Twitter feature is build with this new approach, but the other features will switch to it as well in future versions. The object-detection based approach allows for more flexibility and accuracy at faster execution times.

The quality of the results is based upon two aspects: the correct attribution of different parts of the screenshot & the OCR data extraction. The latter is handled by an external library and can not really be influenced by Pluto.
However, both aspects profit greatly if the input screenshot is very clear and sharp. Therefore I highly recommend avoiding image compression as much as possible. The screenshot should be in a reasonable size, so that all elements are distinct. Once everything is nice and crisp there is no point in increasing the resolution even more, since that would only exponentially lengthen the computational time.

# Use cases & roadmap
Here are some ideas on how to use Pluto. Imagine you see a screenshot of a Tweet. If you want to interact with that Tweet (like, comment, retweet), you first have to find it based on the screenshot. Use Pluto to extract all the needed metadata for a search query automatically. We can take this concept a little further: unfortunately, sometimes screenshots of social media posts or news articles have been deliberately manipulated to spread misinformation. By combining Pluto's metadata extraction with an API for reverse search, detecting potential 'Fake News' is possible. Also, screenshots can be integrated into various analytical pipelines using Pluto to gain even deeper insights into online conversations.

So what to expect next? Pluto is actively being developed & maintained, may it be in a (very) slow manner. Since I'm a student and can only work on this project for fun in my free time, there's not much I can do about that. If you have any feature ideas or recommendations for any part of this project, please feel free to open a new issue or contact me on [Twitter](https://twitter.com/malikpaetzold). If you have read through all of this to down here, I would like to thank you for your time and interest.


PS: ich hab mal das Poster vom BND Summer of Code 2021 mit hochgeladen, falls jemand noch mehr Interesse an der ganzen Sache hat. kanns nur empfehlen :)
