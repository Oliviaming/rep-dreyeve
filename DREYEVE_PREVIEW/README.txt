  _____   _____    __                 __ __      __ ______      _____          _                     _   
 |  __ \ |  __ \  / /                 \ \\ \    / /|  ____|    |  __ \        | |                   | |  
 | |  | || |__) || |  ___  _   _   ___ | |\ \  / / | |__       | |  | |  __ _ | |_  __ _  ___   ___ | |_ 
 | |  | ||  _  / | | / _ \| | | | / _ \| | \ \/ /  |  __|      | |  | | / _` || __|/ _` |/ __| / _ \| __|
 | |__| || | \ \ | ||  __/| |_| ||  __/| |  \  /   | |____     | |__| || (_| || |_| (_| |\__ \|  __/| |_ 
 |_____/ |_|  \_\| | \___| \__, | \___|| |   \/    |______|    |_____/  \__,_| \__|\__,_||___/ \___| \__|
                  \_\       __/ |     /_/                                                                
                           |___/					   
by the DR(eye)VE team
(Stefano Alletto, Andrea Palazzi, Davide Abati, Francesco Solera,
Simone Calderara and Rita Cucchiara)					   



Hi! We're glad you're interested in our dataset!
This README file is organized as follows:
	1) DREYEVE_DATA.zip content
	2) DREYEVE_PREVIEW.zip content
	3) NOTES
	4) TERMS OF USE

---------------------------------------------------------------------------------------------------------------
| 1 DREYEVE_DATA.zip content                  																  |
---------------------------------------------------------------------------------------------------------------

The DR(eye)VE Dataset is composed by 74 driving sessions in different scenarios.
Each subfolder you see in the root directory corresponds to a different run and contains:

	video_garmin.avi
		Video recorded from the roof-mounted camera. 7500 frames, 1080p, 25fps.
		
	video_saliency.avi
		Ground truth for the current run, computed as explained in the paper.
		Synchronized with the roof-mounted camera, 7500 frames.
	
	video_etg.avi
		Video recorded from the Eye-Tracking Glasses (ETG). 9000 frames, 720p, 30fps.
		
	etg_samples.txt
		This file contains the raw fixations of the driver, recorded from the eye-tracking glasses at 60Hz.
		The file is organized as follows:
			#frame_etg | #frame_gar | Xpx | Ypx | event_type (| code)
		The first column gives the index of the video_etg frame corresponding to the current gaze data.
		The second column gives the index of the video_garmin frame corresponding to the current gaze data.
		X and Y columns are the raw gaze coordinates in the video_etg frame.
		The fifth column indicates the type of observation, i.e. Fixation or Saccade.
		(Last column is an internal timestamp, residuum of the data export process. It should be of no use.)

	speed_course_coord.txt
		This file contains information on speed, course, latitude, longitude for the current run.
		The file is organized as follows:
			#frame | speed | course | lat | lon
		Speed (km/h) and Course (degree w.r.t North) information are available for each frame of the Garmin video (7500 rows = 5min * 25hz).
		Lat and Lon information are available approximately every 25 frames (e.g. once per second).
		
	mean_frame.png
		Pre-computed mean frame of the garmin video for the current run
		
	mean_gt.png
		Pre-computed mean ground truth image for the current run
		
		
Other than folders for each run, we also provide a subsequences.txt file where annotations about attentive/inattentive driving can be found.
Each line of this file encodes the run number, the starting and ending frame and the subsequence type. The type of events annotated are:
	- inattentive (i)
	- attentive and not trivial (k)
	- attentive but uninteresting (u)
	- errors in the measurement process (e)

As an example, the line:
1	2175	2188	i
denotes an inattentive driving event from frame 2175 to 2188 in run 01.


All frames where no gaze map could be computed are listed in the missing_gt.txt file.


The file dr(eye)ve_design.txt lists the characteristics of each run. It's organized as follows:
	#run | light_condition | weather | landscape | driver_id | set
where:
	#run = {01, 02, ..., 74}
	light_condition = {Morning, Evening, Night}
	weather = {Sunny, Cloudy, Rainy}
	landscape = {Downtown, Countryside, Highway}
	driver_id = {D1, D2, ..., D8}
	set = {training_set, test_set}
		
---------------------------------------------------------------------------------------------------------------
| 2 DREYEVE_PREVIEW.zip content                  															  |
---------------------------------------------------------------------------------------------------------------

Together with the main data (Sec. 1), we provide superimposition of frames and ground truth gaze maps. This archive
saves you the trouble of blending frames and ground truth and allows you to visually inspect the data. 
Additional 15 GB of data.

---------------------------------------------------------------------------------------------------------------
| 3 NOTES                        															                  |
---------------------------------------------------------------------------------------------------------------

In this section we enlist shortcomings encountered during the dataset creation.
We manually corrected errors in order to have all videos at the same resolution and frame rate.
Nevertheless, for clarity, we report the following:
- runs 11, 36, 42, 67 have been recorded at 25fps instead of 30fps and gaze captured at 30Hz instead of 60Hz due to measuring tools error.
- runs 01, 08, 15, 29, 38, 45, 57, 58, 66 have been recorded at 720p instead of 1080p from the roof-mounted camera.

---------------------------------------------------------------------------------------------------------------
| 4 TERMS OF USE                															                  |
---------------------------------------------------------------------------------------------------------------

This dataset is free for academic and non-commercial use, for commercial use please contact the authors. If you use it in your work, please cite the respective publication:

@inproceedings{dreyeve2016,
  title={DR(eye)VE: a Dataset for Attention-Based Tasks with Applications to Autonomous and Assisted Driving},
  author={Alletto, Stefano and Palazzi, Andrea and Solera, Francesco and Calderara, Simone and Cucchiara, Rita},
  booktitle={IEEE Internation Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2016}
}

@article{palazzi2017focus,
  title={Predicting the Driver's Focus of Attention: the DR(eye)VE Project},
  author={Palazzi, Andrea and Abati, Davide and Calderara, Simone and Solera, Francesco and Cucchiara, Rita},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2018}
}

For questions and comments, please email the DR(eye)VE team http://imagelab.unimore.it/dreyeve

The dataset provided is distributed "as is", we do not offer a warranty for the content or use of these files nor do 
we guarantee their quality, accuracy, fitness for a particular purpose or safety - either expressed or implied.
