# Computer Pointer Controller

Computer Pointer Controller app estimate pose of head and is used to controll the movement of mouse pointer by the direction of eyes and also . This app takes video as input and then app estimates eye-direction and head-pose and based on that estimation it move the mouse pointer.

## Project Set Up and Installation

### Step 1 is install openvino in you operative sistem  

[Install openvino](https://docs.openvinotoolkit.org/latest/index.html)

### Step 2 inicialice the openvino enviroment

```
 source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5

```

### Step 3 download the following models by using openVINO model downloader:-

use the next commands in the a terminal in the main folder:

* Face Detection Model

```python
    python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"
```
* Facial Landmarks Detection Model

```python
    python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"
```
* Head Pose Estimation Model

```python
    python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"
```

* Gaze Estimation Model

```python
    python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"
```


## Demo

- Run the demo using the next comand in the folder where you have the main file:

```python
    python3 main.py -f intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -fl intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml -hp intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -g intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i bin/demo.mp4
```
## Documentation

### Structure
```python 

Intel Edge AI for IoT Developers Project-3
   -- bin 
        -gitkeep
        -demo.mp4
   -- Intel (Note: this folder will appear after step 3)
       --face-detection-adas-binary-0001
           #face-detection-adas-binary-0001.bin
           #face-detection-adas-binary-0001.xml
       --gaze-estimation-adas-0002
           --FP16
                #gaze-estimation-adas-0002.bin
                #gaze-estimation-adas-0002.xml
           --FP16-INT8
                #gaze-estimation-adas-0002.bin
                #gaze-estimation-adas-0002.xml
           --FP32
                #gaze-estimation-adas-0002.bin
                #gaze-estimation-adas-0002.xml
       --head-pose-estimation-adas-0001
           --FP16
                #head-pose-estimation-adas-0001.bin
                #head-pose-estimation-adas-0001.xml
           --FP16-INT8
                #head-pose-estimation-adas-0001.bin
                #head-pose-estimation-adas-0001.xml
           --FP32
                #head-pose-estimation-adas-0001.bin
                #head-pose-estimation-adas-0001.xml
       --landmarks-regression-retail-0009
           --FP16
                #landmarks-regression-retail-0009.bin
                #landmarks-regression-retail-0009.xml
           --FP16-INT8
                #landmarks-regression-retail-0009.bin
                #landmarks-regression-retail-0009.xml
           --FP32
                #landmarks-regression-retail-0009.bin
                #landmarks-regression-retail-0009.xml
    #face_detection.py
    #facial_landmarks_detection.py
    #gaze_estimation.py
    #head_pose_estimation.py
    #imput_feeder.py
    #main.py
    #mouse_controller.py
    #README.md
    #requirements.txt
```

 [Face Detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html) : Face detector for driver monitoring and similar scenarios. The network features a pruned MobileNet backbone that includes depth-wise convolutions to reduce the amount of computation for the 3x3 convolution block. Also some 1x1 convolutions are binary that can be implemented using effective binary XNOR+POPCOUNT approach

 [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html) : This is a lightweight landmarks regressor for the Smart Classroom scenario. It has a classic convolutional design: stacked 3x3 convolutions, batch normalizations, PReLU activations, and poolings. Final regression is done by the global depthwise pooling head and FullyConnected layers. The model predicts five facial landmarks: two eyes, nose, and two lip corners.


 [Head Pose Estimation ](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html) : Head pose estimation network based on simple, handmade CNN architecture. Angle regression layers are convolutions + ReLU + batch norm + fully connected with one output.


 [Gaze Estimation ](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html) : This is a custom VGG-like convolutional neural network for gaze direction estimation.

```python

    "--help" ---> "the comand if you need help" 
    "-f" ---> "face_detection path to .xml file"
    "-fl" ---> "facial_landmark path to .xml file"
    "-hp" ---> "head_pose path to .xml file "
    "-g" ---> "gaze_estimation path to .xml file"
    "-i" ---> "input path to video file or enter cam for webcam"
    "-flags" ---> "Flags" "Specify the flags from face_d, face_landmark_d, head_pose_e, gaze_e"
                          "example --flags fd hp fld ge"
                          "for see the visualization of different model outputs of each frame," 
                          "face_d for Face Detection,"
                          "face_landmark_d for Facial Landmark Detection,"
                          "head_pose_e for Head Pose Estimation"
                          "gaze_e for Gaze Estimation." 
    "-l" ---> "cpu_extension path of extensions of cpu"
    "-prob" ---> "prob_threshold Probability threshold."
    "-d" ---> "device"  "Specify the target device to run on: "
                        "only CPU, GPU, FPGA or MYRIAD is acceptable. "
                        "(CPU by default)"
```


## Results

The experiment was runing in a Intel i3-7020U CPU @ 2.30GHz

### FP32 

```python
    python3 main.py -f intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -fl intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml -hp intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -g intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i bin/demo.mp4

Total loading time of the models: 0.5712590217590332
total inference time 3.569594621658325
fps 16.528487476426776
```

### FP16

```python
    python3 main.py -f intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -fl intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml -hp intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -g intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -i bin/demo.mp4

Total loading time of the models: 0.6020069122314453
total inference time 3.450314521789551
fps 17.09989035127119
```

### INT8

```python
    python3 main.py -f intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -fl intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009.xml -hp intel/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001.xml -g intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002.xml -i bin/demo.mp4

Total loading time of the models: 0.8587613105773926
total inference time 3.5212674140930176
fps 16.755330698221563
```

## Conclusions 

#### The best

```
    The best loading time is FP32 with 0.5712590217590332
    The best total inference time is FP16 with 3.450314521789551
    The best fps is FP16 with 17.09989035127119
```

Strangely, the FP32 and FP16 precision models performed better in both load time and fps than the precision INT8 model.
the results were not as expected since the lower precision model (INT8) should have a lower level of precision, lower loading speed and more fps, but it was found that both the FP32 and FP16 precision performed better both in time of load as in fps, this may be because the type of precision FP32-INT1 and FP16 are better adapted to the type of inference in videos


### Edge Cases

If there are more than one face detected in the frame then model takes the first detected face for control the mouse  pointer. 
If for some reason model can not detect the face then it prints unable to detect the face and read another frame till it detects the face or user closes the window.
