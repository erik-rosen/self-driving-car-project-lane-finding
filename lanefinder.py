import numpy as np
import cv2

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, side):
        # was the line detected in the last iteration?
        self.side = side
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #points of the best fitted line over the last n iterations
        self.bestpts = np.int32( [ np.array([np.transpose(np.vstack(([], [])))])])    
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = np.array([0,0,0], dtype='float')  
        #polynomial coefficients for the most recent fit
        self.current_fit = np.array([0,0,0], dtype='float') 
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        #number of good fits currently averaged over - will never exceed self.average_over_n_fits
        self.number_of_fits = 0
        self.average_over_n_fits = 5
        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
    def updateFit(self, allx, ally):
        self.allx = allx
        self.ally = ally
        
        # Initial sanity check - if the number of points found are too few, skip
        if allx.size<3: 
            self.detected = False 
            return None
        
        # Fit a second order polynomial to the points
        self.current_fit = np.polyfit(ally,allx,2)
        # also fit one that is scaled to meters
        current_fit_meters = np.polyfit(ally*self.ym_per_pix, allx*self.xm_per_pix, 2)
        
        y_eval=720*self.ym_per_pix
        curverad = (1+(2*current_fit_meters[0]*y_eval+current_fit_meters[1])**2)**1.5 / (2*abs(current_fit_meters[0]))
        if self.number_of_fits > 0:
            self.diffs = np.absolute(self.current_fit - self.best_fit)
        
        # Sanity checks
        if (curverad<60):
            self.detected = False
            return None
            
        if (np.sum(self.diffs)>1000):
            self.detected = False
            return None
        
        self.detected = True
        
        if (self.number_of_fits < self.average_over_n_fits):
            self.number_of_fits=self.number_of_fits + 1
        
        #Average over n last fits - if there were none, adjust accordingly
        weight_current_fit = 1 / self.number_of_fits
        weight_of_historical_fit = 1 - weight_current_fit
        
        self.best_fit = [self.current_fit[0]*weight_current_fit + self.best_fit[0]*weight_of_historical_fit,
                    self.current_fit[1]*weight_current_fit + self.best_fit[1]*weight_of_historical_fit,
                    self.current_fit[2]*weight_current_fit + self.best_fit[2]*weight_of_historical_fit]
        
        # Generate x values for plotting
        ploty = np.linspace(0, 720-1, 720 )
        bestx = self.best_fit[0]*ploty**2 + self.best_fit[1]*ploty + self.best_fit[2]
        
        # Transform into points for plotting
        if self.side == 'left':
            self.bestpts = np.int32( [ np.array([np.transpose(np.vstack((bestx, ploty)))])])
        else:
            self.bestpts = np.int32( [ np.array( [ np.flipud( np.transpose( np.vstack( [bestx, ploty] ) ) ) ] ) ] )
            
        # Update radius of curvature
        best_fit_meters = np.polyfit(ploty*self.ym_per_pix, bestx*self.xm_per_pix, 2)
        y_eval=720*self.ym_per_pix
        self.radius_of_curvature = (1+(2*best_fit_meters[0]*y_eval+best_fit_meters[1])**2)**1.5 / (2*abs(best_fit_meters[0]))
        
        # Update base offset from centre - TODO: Clean up with parameters
        self.line_base_pos = (self.best_fit[0]*720**2 + self.best_fit[1]*720 + self.best_fit[2] - 1280/2) * self.xm_per_pix
        
        return None
        
class LaneFinder():
    def __init__(self,cameraMatrix,distCoeffs,perspectiveTransformMatrix):
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.perspectiveTransformMatrix = perspectiveTransformMatrix
        self.inversePerspectiveTransform = np.linalg.inv(perspectiveTransformMatrix) 
        self.leftLane = Line('left')
        self.rightLane = Line('right')
        
    def findLanes(self, frame):
        frameSize = (frame.shape[1],frame.shape[0])
        
        #Undistort image
        frame = cv2.undistort(frame, self.cameraMatrix, self.distCoeffs, None, self.cameraMatrix)
        
        #Get the birdseye view:
        birdsEyeImage = cv2.warpPerspective(frame, self.perspectiveTransformMatrix, frameSize, flags=cv2.INTER_LINEAR)
        
        #Perform Lane Segmentation:
        #grayscale, abs_sobelx, thresh_sobel, extracted_lines = self.laneLineSegmentation(birdsEyeImage)
        grayscale, laplacian, extracted_lines = self.laneLineSegmentationLaplace(birdsEyeImage)
        
        #Find lane pixels
        #leftx, lefty, rightx, righty, extracted_lane_pixels = self.find_lane_pixels_sliding_window(extracted_lines)
        if (self.leftLane.detected):
            leftx, lefty, left_img = self.search_around_poly(extracted_lines.copy(),self.leftLane)
        else: 
            leftx, lefty, left_img = self.find_lane_pixels_sliding_window_one_side(extracted_lines.copy(),self.leftLane)
        if (self.rightLane.detected):
            rightx, righty, right_img = self.search_around_poly(extracted_lines.copy(),self.rightLane)
        else: 
            rightx, righty, right_img = self.find_lane_pixels_sliding_window_one_side(extracted_lines.copy(),self.rightLane)
        
        
        extracted_lane_pixels = np.hstack((left_img, right_img))
        
        #Update lines
        self.leftLane.updateFit(leftx,lefty)
        self.rightLane.updateFit(rightx,righty)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,'L curv: '+ str(self.leftLane.radius_of_curvature) ,(900,30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame,'R curv: '+ str(self.rightLane.radius_of_curvature) ,(900,60), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame,'L offset: '+ str(self.leftLane.line_base_pos) ,(900,90), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame,'R offset: '+ str(self.rightLane.line_base_pos) ,(900,120), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        if (self.leftLane.number_of_fits>0 and self.rightLane.number_of_fits>0 ):
            center_offset = (self.leftLane.line_base_pos+self.rightLane.line_base_pos)*0.5
        else:
            center_offset = 0
        cv2.putText(frame,'Offset in Lane: '+ str(center_offset) ,(900,150), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        #cv2.putText(frame,'L diffs: '+ str(np.sum(self.leftLane.diffs)) ,(900,180), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        #cv2.putText(frame,'R diffs: '+ str(np.sum(self.rightLane.diffs)) ,(900,210), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        #Draw best averaged best fit lanes
        poly = np.hstack((self.leftLane.bestpts[0], self.rightLane.bestpts[0]))
        color_overlay = np.zeros_like(frame)
        cv2.polylines(color_overlay, self.leftLane.bestpts, False, color=[255,0,0], thickness=30, lineType=cv2.LINE_AA)
        cv2.polylines(color_overlay, self.rightLane.bestpts, False, color=[0,0,255], thickness=30, lineType=cv2.LINE_AA)
        cv2.fillPoly(color_overlay, np.array([poly], dtype=np.int32), (0,255, 0))
        warped_color_overlay = cv2.warpPerspective(color_overlay, self.inversePerspectiveTransform, (frame.shape[1],frame.shape[0]), flags=cv2.INTER_NEAREST)
        
        
        #Stitch together images
        processingrow = np.hstack((
                                   cv2.cvtColor(grayscale,cv2.COLOR_GRAY2RGB), 
                                   #cv2.cvtColor(abs_sobelx,cv2.COLOR_GRAY2RGB), 
                                   #cv2.cvtColor(thresh_sobel,cv2.COLOR_GRAY2RGB),
                                   cv2.cvtColor(laplacian,cv2.COLOR_GRAY2RGB),
                                   cv2.cvtColor(np.uint8(extracted_lines),cv2.COLOR_GRAY2RGB),
                                   extracted_lane_pixels,
                                   cv2.addWeighted(birdsEyeImage, 1, color_overlay, 0.3, 0,-1)                            
                                  ))
        processingrow = cv2.resize(processingrow, (frameSize[0],frameSize[1]//4), interpolation = cv2.INTER_AREA)
        result = np.vstack((processingrow,cv2.addWeighted(frame, 1, warped_color_overlay, 0.3, 0,-1)))
        return result
    
    def laneLineSegmentation(self, birdsEyeImage):
        # S-channel is good at finding yellow lines and is largely invariant to lighting conditions
        # S-channel isn't great at finding the white lines though, so meld it together with
        # a greyscale image to find both white and yellow lines
        sChannel = cv2.cvtColor(birdsEyeImage,cv2.COLOR_RGB2HSV)[:,:,1]
        gray = cv2.cvtColor(birdsEyeImage,cv2.COLOR_RGB2GRAY)
        weighted = cv2.addWeighted(sChannel, 1, gray, 1, 0,-1)
        
        # Extract edges of the vertical lines
        sobelx = cv2.Sobel(weighted, cv2.CV_32F, 1, 0, 15)
        # Take the absolute value of the x gradients
        abs_sobelx = np.absolute(sobelx)
        
        # Threshold gradients
        thresh_sobel = np.zeros_like(abs_sobelx)
        thresh_sobel[(abs_sobelx > 40)] = 1
       
        # We know that lies are 20-40 pixels wide. By dilating the thresholded edges, we join the edges together
        # We then erode the dilated lines by more than that such that any edges that are not within a certain 
        # distance of another edge will be removed.
        #dilation_size = 40
        #erosion_size = 70
        # Create structure elements
        #dilationStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_size, 1))
        #erosionStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion_size, 1))
        # Apply morphology operations
        #dilated = cv2.dilate(thresh_sobel, dilationStructure)
        #eroded = cv2.erode(dilated, erosionStructure)
        
        #extracted_lines = eroded * 255
        #return weighted, abs_sobelx, thresh_sobel*255, extracted_lines
        return weighted, abs_sobelx, thresh_sobel*255, thresh_sobel*255
    
    
    def laneLineSegmentationLaplace(self, birdsEyeImage):
        size = (birdsEyeImage.shape[1],birdsEyeImage.shape[0])
        downsample = cv2.resize(birdsEyeImage, (birdsEyeImage.shape[0]//10,birdsEyeImage.shape[1]//10), interpolation = cv2.INTER_AREA)
        sChannel = cv2.cvtColor(downsample,cv2.COLOR_RGB2HSV)[:,:,1]
        gray = cv2.cvtColor(downsample,cv2.COLOR_RGB2GRAY)
        weighted = cv2.addWeighted(sChannel, 1, gray, 1, 0,-1)
        #laplacian = cv2.filter2D(weighted,cv2.CV_32F,kernel)
        
        
        laplacian = cv2.Laplacian( weighted, ddepth = cv2.CV_32F, ksize=7 )
        l_min = np.amin(laplacian)
        l_max = np.amax(laplacian)
        laplacian = np.uint8((laplacian - l_min) * 255 / (l_max - l_min))
        thresh_laplacian = np.zeros_like(laplacian)
        thresh_laplacian[(laplacian < 100)] = 255
        # TODO: Implement
      
        return cv2.resize(weighted,size,interpolation = cv2.INTER_NEAREST), cv2.resize(laplacian,size,interpolation = cv2.INTER_NEAREST), cv2.resize(thresh_laplacian,(size),interpolation = cv2.INTER_NEAREST)
    
    
    def find_lane_pixels_sliding_window(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 120
        # Set minimum number of pixels found to recenter window
        minpix = 30
    
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base
    
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
    
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            ### Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - margin  # Update this
            win_xleft_high = leftx_current + margin  # Update this
            win_xright_low = rightx_current - margin  # Update this
            win_xright_high = rightx_current + margin  # Update this
            
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 
            
            ### Identify the nonzero pixels in x and y within the window ###
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            ### If you found > minpix pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position ###
            if len(good_left_inds)>minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds)>minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass
    
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
    
        return leftx, lefty, rightx, righty, out_img
    
    def find_lane_pixels_sliding_window_one_side(self, binary_warped, line):
        # Split the image in half
        midpoint = np.int(binary_warped.shape[1]//2)
        
        if(line.side=='left'):
            binary_warped= binary_warped[:, :midpoint]
        else:
            binary_warped= binary_warped[:, midpoint:]
        
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        x_base = np.argmax(histogram)
    
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 120
        # Set minimum number of pixels found to recenter window
        minpix = 30
    
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        x_current = x_base
    
        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []
    
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            ### Find the four below boundaries of the window ###
            win_x_low = x_current - margin  # Update this
            win_x_high = x_current + margin  # Update this
            
            
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_x_low,win_y_low),
            (win_x_high,win_y_high),(0,255,0), 2) 
            
            ### Identify the nonzero pixels in x and y within the window ###
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]
            
            # Append these indices to the lists
            lane_inds.append(good_inds)
            
            ### If you found > minpix pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position ###
            if len(good_inds)>minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))
    
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            lane_inds = np.concatenate(lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass
    
        # Extract left and right line pixel positions
        if(line.side=='right'):
            x = nonzerox[lane_inds] + 640
        else:
            x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds] 
    
        return x, y, out_img
    
    def search_around_poly(self, binary_warped, line):
        # Split the image in half
        midpoint = np.int(binary_warped.shape[1]//2)
        
        if(line.side=='left'):
            binary_warped = binary_warped[:, :midpoint]
        else:
            binary_warped = binary_warped[:, midpoint:]
        
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        margin = 100
    
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        
        if (line.side=='right'):
            fit_x = line.best_fit[0]*(nonzeroy**2) + line.best_fit[1]*nonzeroy + line.best_fit[2] - midpoint
        else:
            fit_x = line.best_fit[0]*(nonzeroy**2) + line.best_fit[1]*nonzeroy + line.best_fit[2]
        lane_inds = ((nonzerox > (fit_x - margin)) & (nonzerox < (fit_x + margin)))
        
        
        # Again, extract left and right line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds] 
    
        # Fit new polynomials
        #fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
        
        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = [255, 0, 0]
        
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        fitx = line.best_fit[0] * ploty**2 + line.best_fit[1] * ploty + line.best_fit[2]
        if (line.side=='right'):
            fitx = fitx - midpoint
    
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        line_window1 = np.array([np.transpose(np.vstack([fitx-margin, ploty]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx+margin, 
                                  ploty])))])
        line_pts = np.hstack((line_window1, line_window2))
    
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        if(line.side=='right'):
            x = x + midpoint
        
        return x, y, result