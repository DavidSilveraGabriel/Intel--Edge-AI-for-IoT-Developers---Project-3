# Computer Pointer Controller

Computer Pointer Controller app is used to controll the movement of mouse pointer by the direction of eyes and also estimated pose of head. This app takes video as input and then app estimates eye-direction and head-pose and based on that estimation it move the mouse pointer.

## Project Set Up and Installation

### Step 1 is install openvino in you sistem operative 

linux --> https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html
windows --> https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html

### Step 2 inicialice the openvino enviroment

 source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5

### Step 3 download the following models by using openVINO model downloader:-

use the next commands in the a terminal in the main folder (Mateusz-pointer-controler):

--- 1. Face Detection Model

python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"

--- 2. Facial Landmarks Detection Model

python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"

--- 3. Head Pose Estimation Model

python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"

--- 4. Gaze Estimation Model

python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"



## Demo

For run the demo use the next comand in the folder where you have the main file:

python3 main.py -f intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -fl intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml -hp intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -g intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i bin/demo.mp4

## Documentation
Mateusz-pointer-controler
    -bin 
       -gitkeep
       -demo.mp4
       -bench_1.jpeg
       -bench_2.jpeg
       -bench_3.jpeg
    -Intel (Note: this folder will appear after step 3)
       -face-detection-adas-binary-0001
          -face-detection-adas-binary-0001.bin
          -face-detection-adas-binary-0001.xml
       -gaze-estimation-adas-0002
          -FP16
              -gaze-estimation-adas-0002.bin
              -gaze-estimation-adas-0002.xml
          -FP16-INT8
              -gaze-estimation-adas-0002.bin
              -gaze-estimation-adas-0002.xml
          -FP32
              -gaze-estimation-adas-0002.bin
              -gaze-estimation-adas-0002.xml
       -head-pose-estimation-adas-0001
          -FP16
              -head-pose-estimation-adas-0001.bin
              -head-pose-estimation-adas-0001.xml
          -FP16-INT8
              -head-pose-estimation-adas-0001.bin
              -head-pose-estimation-adas-0001.xml
          -FP32
              -head-pose-estimation-adas-0001.bin
              -head-pose-estimation-adas-0001.xml
       -landmarks-regression-retail-0009
          -FP16
              -landmarks-regression-retail-0009.bin
              -landmarks-regression-retail-0009.xml
          -FP16-INT8
              -landmarks-regression-retail-0009.bin
              -landmarks-regression-retail-0009.xml
          -FP32
              -landmarks-regression-retail-0009.bin
              -landmarks-regression-retail-0009.xml
    -face_detection.py
    -facial_landmarks_detection.py
    -gaze_estimation.py
    -head_pose_estimation.py
    -imput_feeder.py
    -main.py
    -mouse_controller.py
    -README.md
    -requirements.txt

--- [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html) : Face detector for driver monitoring and similar scenarios. The network features a pruned MobileNet backbone that includes depth-wise convolutions to reduce the amount of computation for the 3x3 convolution block. Also some 1x1 convolutions are binary that can be implemented using effective binary XNOR+POPCOUNT approach

--- [Facial Landmarks Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html) : This is a lightweight landmarks regressor for the Smart Classroom scenario. It has a classic convolutional design: stacked 3x3 convolutions, batch normalizations, PReLU activations, and poolings. Final regression is done by the global depthwise pooling head and FullyConnected layers. The model predicts five facial landmarks: two eyes, nose, and two lip corners.


--- [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html) : Head pose estimation network based on simple, handmade CNN architecture. Angle regression layers are convolutions + ReLU + batch norm + fully connected with one output.


--- [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html) : This is a custom VGG-like convolutional neural network for gaze direction estimation.

arguments: 
  -h --"help" 

  -f --"face detection model Path to .xml file of face detection model."
 
  -fl --"facial landmark model Path to .xml file of Facial Landmark Detection model."
 
  -hp --"head pose model Path to .xml file of Head Pose Estimation model."
 
  -g --"gaze estimation model Path to .xml file of Gaze Estimation model."
 
  -i --"input Path to video file or enter cam for webcam"
 
  -l --"cpu_extension path of extensions if any layers is incompatible with hardware"
  
  -prob --"prob_threshold Probability threshold for model to identify the face."

  -d --"device Specify the target device to run on: CPU, GPU, FPGA or
                        MYRIAD is acceptable. Sample will look for a suitable
                        plugin for device (CPU by default)"


## Benchmarks

comands used for benchmarking
FP32   
python3 main.py -f intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -fl intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml -hp intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -g intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i bin/demo.mp4


FP16
python3 main.py -f intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -fl intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml -hp intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -g intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -i bin/demo.mp4

INT8
python3 main.py -f intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -fl intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009.xml -hp intel/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001.xml -g intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002.xml -i bin/demo.mp4

## Results

Intel(R) Core(TM) i7-7700HQ CPU @2.80GHz 


### FP32

![fp32](bin/bench_2.jpeg)

### FP16

![fp32](bin/bench_1.jpeg)


### INT8

![fp32](bin/bench_3.jpeg)



As we can see from the previous results on the Intel (R) Core (TM) i7-7700HQ @ 2.80GHz CPU, the models with higher precision give us a better fps time, but strangely the results are not as expected. Because the lower precision models use less memory and are less computationally expensive, so they should give better results than the higher precision models, but this does not happen in these tests.


## Stand Out Suggestions



### Edge Cases
If for some reason model can not detect the face then it prints unable to detect the face and read another frame till it detects the face or user closes the window.

If there are more than one face detected in the frame then model takes the first detected face for control the mouse  pointer. 