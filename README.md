Contents:
* OpenCV - https://github.com/muse-gabs/MachineLearningProgress#opencvprogramming
* Python Machine Learning Modeling - https://github.com/muse-gabs/MachineLearningProgress#python-machine-learning-model
* NLP - https://github.com/muse-gabs/MachineLearningProgress#nlp

# OpenCVProgramming
working on computer vision in my spare time

this is a compilation of coding that I'm putting together on my journey through computer vision. I had started this a while ago but my computer couldn't run the video captures without overheating. 

The initial struggle with downloading packages on a windows platform is making sure the paths are setup correctly and we're able to import the right packages, there were a lot of different ways to do this but I stumbled across a simple video to be able to get it setup correctly for Visual Studio Code. 

I used this video to help me with setting it up and installing it:
https://www.youtube.com/watch?v=d3AT9EGp4iw

So after going through and making sure it was installed correctly I ran into an issue where the imshow function was not showing the image as expected on Visual Studio Code. After a lot of researching and going through stackoverflow I managed to find this link https://stackoverflow.com/questions/49992300/how-to-show-graph-in-visual-studio-code-itself
which helped me be able to get the image to show when running the program in Visual Studio Code using Jupyter extension.

Here is how my screen looks after setting it up: 
![alt text](https://github.com/muse-gabs/OpenCVProgramming/blob/main/getting%20image%20to%20show%20in%20visual%20studio%20code.png?raw=true)

The code shown there was part of the official opencv github for deep neural networks on openpose.

https://github.com/opencv/opencv/blob/master/samples/dnn/openpose.py

Now the image of the people is not the right color, this is because when we plt.imshow(img) the image is shown in BGR format instead of RGB, to change that we convert the image to RGB by using plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)) this results in the screen below:
![alt text](https://github.com/muse-gabs/MachineLearningProgress/blob/main/change%20image%20to%20RGB.png)

I realized I was jumping the gun a little with trying to get poses of a group of people before getting poses for 1 person, after some tweaking I was able to get the model for different poses. Shown in the updated version of poseEstimation.py as well as the images that I'll add below:

# Python Machine Learning Model

Took a step back to learn how to test and train a model with a simpler data set.

Utilized UCI's Machine Learning Repository to do a Linear Regression model, see linearRegression.py for information (everything is in the comments)

Used this data set https://archive.ics.uci.edu/ml/datasets/Student+Performance to work with for a simplistic model. 

Continuing with the dataset and creating linearRegressionPart2.py I was able to plot the different relationships between the G3 which was our predicted variable and G1, G2, failures, absences, and studytime. 

The graphed results are shown below:

![alt text](https://github.com/muse-gabs/MachineLearningProgress/blob/main/G1Chart.png)

![alt text](https://github.com/muse-gabs/MachineLearningProgress/blob/main/G2Chart.png)

![alt text](https://github.com/muse-gabs/MachineLearningProgress/blob/main/failureChart.png)

![alt text](https://github.com/muse-gabs/MachineLearningProgress/blob/main/absencesChart.png)

![alt text](https://github.com/muse-gabs/MachineLearningProgress/blob/main/studytimeChart.png)

# NLP 

Still trying to fully understand Word2Vec and Gensim model, want to eventually be able to create my own model

*Big eventual goal is to be able to train a model so that we can develop sign language in real time for videos using either motion processing or subtitle processing, I finish my CS Degree in June 2021, am hoping to learn as much as I can as fast as I can within the following months while taking 2 classes and having a part-time job, it's a stretch but this is my current goal*
