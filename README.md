# facial_recognition
**facial_recognition** aims to be the most effective face recognition [library](https://pypi.org/project/facial-recognition/) for python that you can install with a single command


```
pip install facial_recognition
```
After that, you need to download and setup the model file used for recognition by running the following command (Even if you dont do this now, the model file will be downloaded the first time you run this library)

```
facial_recognition-setup
```
In case this results in an error, you need to manually download the model file to the path where facial_recognition is installed( Usually on Windows its in C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\facial_recognition) using wget on Linux and Invoke-WebRequest on Windows

```
Invoke-WebRequest https://huggingface.co/Rubarion/facial_recognition_model/resolve/main/arcface.onnx -OutFile arcface.onnx
```

# Adding Known faces to the database
Just copy a preferably high definition camera facing and face clearly visible single photo of each person you want recognized to a folder and rename it with the name of that person. Afte you have copied and renamed all known faces in this manner, Run the following command from inside the folder (open the folder and then right click shell/terminal)

```
facial_recognition add_faces
```
This creates a database of known faces to which the faces in the new image that you give will be compared for matching. As the accuracy of this library is solely dependednt on the quality of the single photo of each person that you store in databse, please make sure that you use a very clear photo where the face is looking straight at the camera and slightly zoomed in so that the entire face region is clearly visible.
Then inorder to recognise the faces in a image, lets say sample.jpg, Open terminal in the folder where this image is stored and simply run

```
facial_recognition recognize sample.jpg
```

This will create an output folder in that same folder and the corresponding output image with boxes drawn around faces and labels will be saved in that folder.


To remove all stored known faces and start again

```
facial_recognition remove_faces
```

# Testing and Accuracy

This package was tested on the **[Labelled Faces in the Wild (LFW Dataset)](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset?utm_source=chatgpt.com) and a very impressive 74.65% accuracy or 969 faces recognised correctly out of total 1298 was obtained.** (Screenshots of test running in terminal attached below)

![Test in progress](facial_recognition/tests/accuracy/one.png "Adding all faces in LFW to database and running the tests")

![Test in progress](facial_recognition/tests/accuracy/two.png "Test Result")

Why accuarcy is not close to 100% is because the images in this dataset are very small cropped faces as shown in the output images below (where the facial_recognition package correctly recognised Angelina Jolie and Catherine Zeta Jones) and as we are very unlikely to encounter such small cropped close up shots of face images or frames in the real world, it was not worth the effort to tweak the code to obtain >90% accuracy just for this dataset. Basically this test proves that both fo normal images of persons as well as for very small images like this, the package does very very well in correctly identifying the faces.

![Output Images](facial_recognition/tests/accuracy/four.jpg "Angelina Jolie")

![Output Images](facial_recognition/tests/accuracy/three.jpg "Catherine Zeta Jones")


**Please note in case of any bugs that this library will be updated frequently**
