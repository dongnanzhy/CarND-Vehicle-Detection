**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/hog_vehicle.png
[image3]: ./output_images/hog_non-vehicle.png
[image4]: ./output_images/heatmap.png
[image5]: ./output_images/sliding_windows.png
[image6]: ./output_images/sliding_windows_merge.png
[image7]: ./output_images/sliding_windows_v2.png
[image8]: ./output_images/sliding_windows_merge_v2.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! The project code can be found [here](vehicle_detection.ipynb)

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the `Read Training Data` and `Feature Extraction` code sections of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images. I did a bit statistics showing there are **8792** `vehicle` images and **8968** `non-vehicle` images. The dataset seems to be balanced, it is important to check training dataset balance before classification. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![car vs non-car][image1]

Then I did feature extraction (spatial, color and HOG). This cell contains a set of functions provided by Udacity's lectures to extract the features from an image. The function `extract_features` combine the other function and use the class `FeatureParameters` to hold all the parameters in a single place.

Below shows an example for HOG for `vehicle` and `non-vehicle`:

![hog vehicle][image2]

![hog non-vehicle][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried parameters as follow.
```python
# HOG parameters
self.cspace = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
self.orient = 8 # HOG orientations
self.pix_per_cell = 8 # HOG pixels per cell
self.cell_per_block = 2  # HOG cells per block
self.hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
# Spatial binning dimensions
self.size = (16, 16)
# Histogram parameters
self.hist_bins = 32  # Number of histogram bins
self.hist_range = (0, 256)
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Data was split to 80% for training and 20% for test.

I used `LinearSVC` model with `StandardScaler` on data pre-process. I also did SVC tunning by playing with the following model hyper-parameters
- svc = LinearSVC() - Based on class recommendation
- svc = LinearSVC(C = 5E-5) - `C` controls training accuracy which can tune to avoid overfitting
- svc = svm.SVC(kernel= 'poly') - To try non-linear fits I tried polynomial SVC

I had similar results with `L1` regularization or by tweeking `C`. I decided to go with the simpler model of `LinearSVC()`. The model gives me around 99% accuracy across both training and test set.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

My first approach to implement sliding windows was to calculate all the windows and then apply the feature extraction to each one of them to find the one containing a car. The code is at the `Test Images` code section of Ipython Notebook. The scales and overlap parameter where found by experimenting on them until a successful result was found. The following image shows the results of this experimentation on the test images:

![sliding windows][image5]

To combine the boxes found there and eliminate some false positives, a (1) heat map as implemented with a threshold and (2) function `scipy.ndimage.measurements.label` was used to find where the cars are. These codes can be found at `Heatmap and Labels` code section of Ipython Notebook. Heatmap results on test images sliding windows can be found below:

![heatmap results][image4]

Final results on test images after merging bounding boxes can be found below:

![sliding windows merge][image6]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The performance of the method calculating HOG on each particular window was slow. To improve the processing performance, a HOG sub-sampling was implemented as suggested on Udacity's lectures. The implementation of this method could be found on `Improving performance with HOG sub-sampling` code section of Ipython Notebook. The following images show the results applied to the test images.

![sliding windows optimized][image7]

Then the same heatmap and threshold procedure was applied. I tuned the `threshold` for applying heatmap and set it to be **2** in the end. Final bounding box results can be found below.

![sliding windows merged optimized][image8]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result(project_video)](./output_images/project_video.mp4)

Here's a [link to my video result(test_video)](./output_images/test_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

Smoothing and Averaging of frames: to make the result more stable and reduce false positive I tried following approach:
- averaging: I averaged the heatmap over three consecutive frames. Results became more jittery with smaller number of frames, but with too many frames a false positive could last on the images for a longer period of time. The images in the previous section included the heapmap across the test images.

The implementation could be found on `Video Pipeline` code section in Ipython Notebook

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
1. When reviewed output project videos, I found there was one frame, the pipeline labeled one `traffic sign` as `vehicle`. Although this can be easily soleved by tuning the averaging frames in video pipleline, this made in think in real situations, how should we get bounding boxes of traffic signs, vehicles, padestrians at the same time. If we have multiple binary models and apply sliding windows for each model, it will run pretty slow. If we have one multi-classification model, then the performance would decrease I guess, with a lot of false alarms. I'm qurious how engineers solve this in real situations.
2. In this project, I actually played the YOLO (as ref in course lectures) for a long time. I found
    - The model we applied has advantage in domain explanation. Every feature we utilized can help finding a car, thus easy to tune feautres.
    - But the model we applied runs slow at inferrence compared to YOLO. Evan after optimizing hog feature generation, it still take me 15 mins to process `project video`, while YOLO can actually tag vehicles on time.
    - Also, the moel we applied has redundent features, and requires a lot of computer vision knowledge to do feature engineering. While in YOLO, the model extract features automatically.
    - Lastly, I found in YOLO we applied IOU to merge bounding boxes. In this assignment, we use heatmap and `scipy.ndimage.measurements.label()`, I'm quite qurious about the differences of these two approaches.
    - Overall, I think YOLO might be a better algorithm doing vehicle detection in real situations.
