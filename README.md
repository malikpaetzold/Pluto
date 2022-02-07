# Pluto
Pluto is a Python library for working with & analyzing screenshots.

![Explainer_1](https://user-images.githubusercontent.com/60754058/119409979-62be3800-bce8-11eb-9f5c-60d623d065b3.png)

The core idea is to extract information from a screenshot and then provide that information in an orderly fashion. This opens the door to many more possibilities, both for applications and analytical pipelines.

![ezgif com-gif-maker (1)](https://user-images.githubusercontent.com/60754058/129491738-09c28f3e-0b52-49a1-8624-08f0a42bc8cd.gif)

How exactly does it all work? The text is extracted using an OCR library (optical character recognition). However, the more interesting part is the correct assignment, meaning, for example, whether the extracted text is still part of the username or already the content of a post. The assignment takes place using the help of AI-supported computer vision and some more specialized techniques.

Please note that Pluto is still in development, and some features are not built yet or only exist as early access (see current limitations). Pluto is capable to process screenshots from social 
media posts, chat histories, and news pages. Current features of the master branch include the Facebook, New York Times, WhatsApp, Discord, Facebook Messenger, Tagesschau, Washiongton Post, WELT & Twitter feature, as well as many 
underlying helper functions for OCR and image transforms. Take a look at example.py to see how the current version of Pluto can be used.

# Quickstart
Download ```pluto.py``` and the ```models``` folder. Make sure you put them is the same directory. After installing all dependencies, you can use Pluto as a CLI or Python Library. For example:

```python pluto.py -i NYT_Example_3.jpg -o nytout.json -c NYT```

Will run the file NYT_Example_3.jpg with the NYT class and save the output as nytout.json. Type ```python pluto.py -h``` for more details.

You can also import ```pluto.py``` as a library, and use all of Pluto's functions & methods.

In both cases I highly recommend going through ```example.ipynb``` to get a better understanding of the software.

# How to get good results & current limitations
Eventhow a good chunk of the core components of Pluto were developed last year, there are a lot of secondary features that have yet to be built. In addition to that, some earlier parts of the code (like the Twitter Feature) are currently getting a major rewrite. This causes some aspects of the feature to be not yet finished (e.g. no dark mode support at the moment); if these parts are strictly necessary for you, please use an earlier version of the software.

The quality of the results is based upon two aspects: the correct attribution of different parts of the screenshot & the OCR data extraction. The latter is handled by an external library and can not really be influenced by Pluto.
However, both aspects profit greatly if the input screenshot is very clear and sharp. Therefore I highly recommend avoiding image compression as much as possible. The screenshot should be in a reasonable size, so that all elements are distinct. Once everything is nice and crisp there is no point in increasing the resolution even more, since that would only exponentially lengthen the computational time.

# Use cases & roadmap
Here are some ideas on how to use Pluto. Imagine you see a screenshot of a Tweet. If you want to interact with that Tweet (like, comment, retweet), you first have to find it based on the screenshot. Use Pluto to extract all the needed metadata for a search query automatically. We can take this concept a little further: unfortunately, sometimes screenshots of social media posts or news articles have been deliberately manipulated to spread misinformation. By combining Pluto's metadata extraction with an API for reverse search, detecting potential 'Fake News' is possible. Also, screenshots can be integrated into various analytical pipelines using Pluto to gain even deeper insights into online conversations.

So what to expect next? Pluto is actively being developed & maintained, may it be in a slow manner. Since I'm a student and can only work on this project for fun in my free time, there's not much I can do about that. If you have any feature ideas or recommendations for any part of this project, please feel free to open a new issue or contact me on Twitter. If you have read through all of this to down here, I would like to thank you for your time and interest.


PS: ich hab mal das Poster vom BND Summer of Code 2021 mit hochgeladen, falls jemand noch mehr Interesse an der ganzen Sache hat. kanns nur empfehlen :)