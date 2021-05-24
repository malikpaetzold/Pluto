# Pluto
Pluto is a Python library for working with &amp; analyzing screenshots.

![Explainer_1](https://user-images.githubusercontent.com/60754058/119409979-62be3800-bce8-11eb-9f5c-60d623d065b3.png)

The core idea is to extract information from a screenshot and then to provide that information in an orderly fashion. This opens the door to many more possibilities, both for applications and analytical pipelines.

![Explainer_1](https://user-images.githubusercontent.com/60754058/119410659-7e760e00-bce9-11eb-8094-9b927475f32a.png)

How exactly does it all work? The text is extracted using an OCR library (optical character recognition). However, the more interesting part is the correct assignment, meaning whether the extracted text is still part of the username or already the content of the Tweet. The assignment takes place using the help of AI-supported computer vision. This is
necessary since a pure “traditional” processing of the extracted text has not proven to be sufficient enough. Especially when processing news articles and Social media posts with a variety of comments and information, this necessity becomes quite clear. 

Please note that Pluto is still in development, and many features are not built yet or only exist as early access. Pluto is supposed to be able to process screenshots from social media posts, chat histories, and news pages. Current features of the master branch include (very) early functionalities of the NYT and Twitter features, as well as many underlying helper functions for OCR and image transforms. Take a look at example.py to see how the current version of Pluto is supposed to be used.

# Current Development Status
Right now the focus is on building up and finishing the Twitter & New York Times features, since they act as a proof-of-concept, both technically and conceptually. Below you can see some results of the NYT segmentation model, which is supposed to isolate the subtitle of an article.
![Explainer_2](https://user-images.githubusercontent.com/60754058/119413524-0fe77f00-bcee-11eb-8bb7-f036f65bf025.png)
These models are basically a neural network that identifies the different parts of an article by looking at the font and its relation to other elements. Since each part has either a different font, color, or other modifiers (like italic or bold), the NN is capable of isolating the various elements of a screenshot.
![Explainer_2](https://user-images.githubusercontent.com/60754058/119414768-7ec5d780-bcf0-11eb-9b03-5ec8157b42bb.png)
For the get_header() method of the Twitter feature, for example, the neural network uses the profile picture to find the correct position of the text. With a little bit of post-processing, the Username and Handle are masked out and can now be hand over to the OCR function.

# Upcoming
As mentioned before, the main focus is on finishing the Twitter and NYT features. This might take a while (a bit more than a month or so), but after that all other, similar features are much faster to build. The reason for that is the development and refinement of the neural networks and the dataset generation pipeline since a lot of upcoming features for other news agencies and social networks will be using the (then) already developed models.
In addition to that, Pluto is going to use more PyTorch instead of Tensorflow. More short-term improvements include important reproducibility upgrades and better code documentation.
