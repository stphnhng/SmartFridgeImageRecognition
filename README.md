# Smart Fridge Image Recognition

Application used in T-Mobile's IOT Hackathon 2017. 

### Product Definition

The product was a smart fridge which could recognize products going inside the fridge and outside. Once recognized, it could display on a basic UI the items inside a fridge. Going inside the fridge is defined as going left to right facing the camera and going out is defined as going right to left facing the camera. This was linked with a database of food items with expiration dates. The UI was also able to display which food items have expired and which haven't.

This product drew concepts from datitran's object recognition system, Google's TensorFlow Object Detection API, and OpenCV.

Datitran: https://github.com/datitran/object_detector_app
Google: [Google's TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) 
and [OpenCV](http://opencv.org/).

### Goal:

Our goal was to reduce food waste by being able to remind people when their food is going to expire and when it has already expired. Furthermore, we felt that besides this function there were many ways to use this data such as building recipes using expired foods, creating a grocery shopping list, finding dietary needs, etc.

### Collaborators:

[Christopher Sofian](https://github.com/Csofian)

Eugene Jahn

[Kevin Yuan](https://github.com/DarkChocoBar)

Valdi Widanta

### Getting Started
1. `conda env create -f environment.yml`
2. Setup Raspberry Pi
    * make sure you can ssh to Raspberry Pi
3. `python RangeFinder.py` from ssh inside Raspberry Pi
    * To start Raspberry Pi server 
4. `python client.py` from laptop
    * To start Image Recognition System on current Laptop (faux smart fridge)


### Requirements
- [Anaconda / Python 3.5](https://www.continuum.io/downloads)
- [TensorFlow 1.2](https://www.tensorflow.org/)
- [OpenCV 3.0](http://opencv.org/)

### Copyright

See [LICENSE](LICENSE) for details.
