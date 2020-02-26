# Face-recognition 
## 1. Project name
- Face recognition
## 2. Description
![](https://i.imgur.com/J9YX9OH.png)
- Using camera (openCV 2) to perform live face recognize and know that person's name.
## 3. Installation
- You need to install Python3, OpenCV, Scikit-learn, Tensorflow 2.0.
## 4. Process
### 1. Data Collection
![](https://i.imgur.com/cx1saM4.png)
- Camera is placed right in front of face.
- Using the Haar Cascade of OpenCV to detect human face.
- Cropping area of detection only.
- Saving image in drive.
### 2. Image processing
- Centering the face in frame
- For Each Image:
    - Convert image to pixels array, array with shape (width, height, 3).
    - Use mtcnn ( a library for facial detection) to detect faces.
![](https://i.imgur.com/xVFRAjn.png)
- Using the angle between left-eye, right-eye, **straighten** face in the image.
![](https://i.imgur.com/oTjftHM.png)
- Result
![](https://i.imgur.com/VsGeN1T.png)
- Crop close to face to eliminate as much background as possible.¶
![](https://i.imgur.com/9vFmFaX.png)
### 3. Training
- Since the application detects faces in real-time and the frame constantly changes, we are not able to use CNN & classify using the output activation function softmax because the computer may not be able to handle it.
- We are going to employ a pretrained model to extract features from each image (number of features range from 128, 256 to 512).
- Feature vetors will be compared between the image captured in real-time and the true label extracted from trained images.
- We employ SVM (Support Vector Machine) - a Machine Learning Classifer - to classify each person using their feature vector.
- We set the probability threshold to ensure real-time accuracy.
#### Model (Facenet + SVM)
![](https://i.imgur.com/1uiyr8H.png)
- Using Facenet to extract 128 features for face
- Using SVM to classify face depend on feature of face.
### 4. Summary 
- Using face detector and openCV to detect face in each frame
- Crop the face image then convert to array with dtype=np.float32 (face array have shape(width, height, 3)
- Resize the array to (160, 160, 3) which fits with FaceNet input layer
- Standardize the array and expand dimension to (1, 160, 160, 3)
- Extract its feature with FaceNet
- Normalize the 128 features in order to predict with SVM. We predict the label and confidence probability
- Use LabelEncoder to inverse_transform from name to number
- Set threshold (probability > 80%) to render name on frame
### 5. Performance
- Define variable i to track the frame, we will predict after every 10 - 20 frames in order to improve the performance ( instead of preprocess and predict every frame, that will drop down the fps from 30 to 2 - 3 fps )
