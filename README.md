## Advanced lane finding

![Alt Text](./output_images/output_project.gif)

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image that marks lane lines in an image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/undistort_output2.png "Road Transformed"
[image3]: ./output_images/straight_lines2.jpg "Binary example"
[image4]: ./output_images/straight_lines1.jpg "Birdseye view of road, top right"
[image5]: ./output_images/test1.jpg "Processed image"
[image6]: ./output_images/test2.jpg "Processed image - good performance"
[image7]: ./output_images/test5.jpg "Processed image - poor performance"
[image8]: ./output_images/test6.jpg "Processed image - poor performance"
[image9]: ./output_images/test4.jpg "Processed image - good performance"
[video1]: ./project_video_dog.mp4 "Video"


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in `./camera_calibration.ipynb`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. I assume the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. `objp` will therefore be a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time we detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I load in the images in the `./camera_cal/` directory in a for-loop, and use OpenCV's `findChessboardCorners()` to extract the corners in the chessboard in the image. If a chessboard is found, I apply OpenCV's `cornerSubPix()` function which refines the position of the corner to subpixel accuracy.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

I serialize the computed camera matrix and the distortion coefficients into a pickle file that I can load in when I run the lanefinding pipeline. This saves me from having to run the calibration code (extract corners, compute the camera matrix and distortion coefficients) everytime I run the pipeline.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

In "./lane_finder.ipynb" I start by loading in the camera matrix and distortion coefficents that we saved earlier in the pickle file. I also define the perspective transfrom (more on this later). All of these are passed to the constructor of my instatiation of a `LaneFinder` object, `l`. The LaneFinder object is defined in `./lanefinder.py` and contains all methods used in the pipeline, the main one being `findLanes()`, defined on line 105. This method takes an input frame, extracts the lane lines from that frame, computes their parameters, and outputs a frame illustrating the intermediary steps in the pipeline, as well as the estimated curvature of the lane lines extracted, and the offset of the car from the center oof the lane.

The first step in the `findLanes()` method is correcting the raw input image from camera distortion using openCV's `undistort()` function, which takes the camera matrix and the distortion coefficients which were passed to the constructor of the `LaneFinder` object. The result is shown below.

![alt text][image2]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The second step in the `findLanes()` method is to compute a "birdseye" view of the road using OpenCV's `warpPerspective()` method. See line 112 in `lanefinder.py`. The perspective tranform matrix used is passed to the constructor of the `LaneFinder` object. 

In our case this matrix is created by manually defining a set of points in the undistorted image. Each point was picked such that the top side and bottom sides of the resulting shape are horizontal, and each point is on the outside of a lane line on the road. The top side points are positioned such that the lane lines are on the verge of being difficult to distingish around that distance, and the bottom side points are picked such that the bottom side is at the bottom of the original image. 

The destination points where selected such that: 1) the resulting shape was perfectly rectangular, 2) there were no "undefined"/dark sections in the resulting view and 3) the car bonnet was not visible.

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 578, 460      | 280, 0        | 
| 702, 460      | 1000, 0       |
| 1100, 720     | 1000, 748     |
| 180, 720      | 280, 748      |

You can see this in line 23-25 in the first cell of `./lane_finder.ipynb`.

I verified that my perspective transform was working as expected by checking that the resulting lane lines in the transformed image were parallel (see top right for birdseye view image tranformed from the original "driver seat" view):

![alt text][image4]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The next step in the `findLanes()` method is to compute a lane segmented binary image. See line 116 of `./lanefinder.py`, which calls the method `laneLineSegmentationLaplace()`.

I experimented with a number of techniques to effectively segment lane lines. The one  which performed best can be found in rows 205-222 of `./lanefinder.py`, `laneLineSegmentationLaplace()`.

This method takes the birdseye view image of the road, and creates a weighted image from the S-channel and grayscale version of the input frame. This accentuates both the yellow and the white lane lines in the resulting grayscale image well. See the image in the top left in the figure below for the resulting output.

Since the width of the lane lines are known, I apply a Laplacian filter with a kernel size which best corresponds to the lane line width (i.e. maximizes it's response when applied to a lane line). To avoid having to use a very large kernel (computationally expensive), I downsample the input images to a tenth of its original size. This allows me to use a much smaller kernel and get the same effect. 

The magnitude of the Laplacian filter output is sensitive to lighting conditions. To mitigate for that when thresholding, I linearly scale the output such that the minimum of the output is set to 0, and the maximum of the output is 255. I assume that if a road lane is present, this will likely result in the minimum output in the image. See the second image from the top left in the figure below for the resulting Laplacian filter output.

I then apply a threshold to the scaled Laplacian filter output to produce a binary image (in our case each pixel is either 0 or 255). See the third image from the top left in the figure below for the resulting output.

![alt text][image3]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The next step in the `findLanes()` method is to find the actual lane pixels in the binary image produced in the previous step. See lines 120-127 in `lanefinder.py`.

Depending on whether the lane line was confidently found in the prior frame, we search around the previously best fit polynomial using `search_around_poly()`, if not, we use sliding windows `find_lane_pixels_sliding_window_one_side()` to identify lane pixels from the thresholded image. These two methods are defined in lines 388-452 and 308-386 respectively in `lanefinder.py`. They return the identified lane line pixels and an image marking the effective lane pixel search areas.

The identified laneline pixels are passed to the `updateFit()` method belonging to the Line objects (instantiated as `leftLane` and `rightLane` belonging to the `LaneFinder` object).

This method can be found on lines 35-95 in `lanefinder.py`. This function first fits a second order polynomial to the the identified lane pixels.

If the number of identified lane line pixels are too few, if the radius of curvature of the interpolated points is lower than 60 meters, and if the difference between the previous best fit is too great, we reject the fit and mark the line as currently undetected.

If the detection is accepted, we update the best fit coefficients. The best fit coefficients are computed as the average of the last 5 accepted fits.

![alt text][image8]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

After updating the best fit coefficients in `updateFit()`, the radius of curvature and offset of the lane line from the vehicle centre is computed. See lines 87-93 in `lanefinder.py` The position of the vehicle relative to the center of the lane bounded by the lane lines is computed from the offset estimate of the left and right lane line relative to the center of the car on line 142 in `lanefinder.py`.

These metrics can all be seen in the image text overlay above, and the code to draw this overlay is defined on lines 137-145.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

As part of `updateFit()` that was run earlier, we store the points of the curve of the best fit second order polynomial the `Line` objects. The next step in the `findLanes()` method is to draw the lane line curves and the lane polygon on the birdseye view image. This is done on lines 150-154 and 166 in `lanefinder.py`. The result is the top right image seen in the figure below.

On line 155 we use the inverse perspective transform which we computed in the constructor of the LaneFinder object to warp the lane lines and lane overlays to how they look from the "driver view". On line 166, we blend the road-warped color overlay with the undistorted "driver view" image, resulting in the bottom image seen in the figure below.

![alt text][image6]

---

### Pipeline (video)


Here's a [link to my video result](./output_project_video_dog.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The lane line segmentation of the current pipeline does not perform very well when the contrast between road and lane line marking is small, when the lane line is occluded or missing from the frame, when part of the road is in shadow, there are speckles on the road, or the color of the road changes - see below for examples where the pipeline fails or does not perform well:

![alt text][image5]
![alt text][image9]
![alt text][image7]

If I were to take this project futher, I would focus on improving the lane line segmentation part of the pipeline, likely exploring training and applying a convolutional neural net for the lane line segmentation task, which is the state of the art at the time of writing: https://paperswithcode.com/paper/learning-lightweight-lane-detection-cnns-by  

In order for the lane line finder to be useful for autonomous driving, it will need to be running with a sufficiently high frame rate throughput - the current framerate throughput of the implementation is likely too low for practical application. Improving the performance of the pipeline would be the second thing I would focus on.
