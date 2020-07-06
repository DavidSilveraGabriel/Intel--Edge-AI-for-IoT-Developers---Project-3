# Thanks to mdfazal
# https://github.com/mdfazal/computer-pointer-controller-1
# this git hub was a big help for me, 

import os
import time
from argparse import ArgumentParser
import cv2
import numpy as np

from face_detection import Face_detect
from facial_landmarks_detection import Facial_landmarks
from gaze_estimation import Gaze_estimation
from head_pose_estimation import Head_pose
from input_feeder import InputFeeder
from mouse_controller import MouseController


def build_argparser():
    parser = ArgumentParser()

    parser.add_argument("-f", "--face_detection", required=True, type=str,
                        help=" Path to .xml file of Face Detection model.")
    parser.add_argument("-fl", "--facial_landmark", required=True, type=str,
                        help=" Path to .xml file of Facial Landmark Detection model.")
    parser.add_argument("-hp", "--head_pose", required=True, type=str,
                        help=" Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-g", "--gaze_estimation", required=True, type=str,
                        help=" Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help=" Path to video file or enter cam for webcam")
    parser.add_argument("-flags", "--Flags", required=False, nargs='+',
                        default=[],
                        help="Specify the flags from face_d, face_landmark_d, head_pose_e, gaze_e"
                             "example --flags fd hp fld ge"
                             "for see the visualization of different model outputs of each frame," 
                             "face_d for Face Detection,"
                             "face_landmark_d for Facial Landmark Detection,"
                             "head_pose_e for Head Pose Estimation"
                             "gaze_e for Gaze Estimation." )
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="path of extensions of cpu")
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to run on: "
                             "only CPU, GPU, FPGA or MYRIAD is acceptable. "
                             "(CPU by default)")
    
    return parser

def face_d(Flags, croppedFace, face_coords):
    if len(Flags) != 1:
        preview_window = croppedFace
    else:
        cv2.rectangle(preview_window,
                    (face_coords[0], face_coords[1]),
                    (face_coords[2], face_coords[3]),
                    (0, 150, 0), 3)

def face_landmark_d(Flags, croppedFace, eye_coords):
    if not 'face_d' in Flags:
        preview_window = croppedFace.copy()
    cv2.rectangle(preview_window,
                 (eye_coords[0][0] - 10, eye_coords[0][1] - 10),
                 (eye_coords[0][2] + 10, eye_coords[0][3] + 10),
                 (0,255,0), 3)
    cv2.rectangle(preview_window,
                 (eye_coords[1][0] - 10, eye_coords[1][1] - 10),
                 (eye_coords[1][2] + 10, eye_coords[1][3] + 10),
                 (0,255,0), 3)

def head_pose_e(Flags, preview_window, hp_out):
    cv2.putText(preview_window,
             "Pose Angles: yaw:{} | pitch:{} | roll:{}".format(hp_out[0],
                                                               hp_out[1],
                                                               hp_out[2]),
            (50, 50),cv2.FONT_HERSHEY_COMPLEX,1,(0, 255, 0),1,cv2.LINE_AA)

def gaze_e(Flags, croppedFace, gaze_vector, left_eye, right_eye,eye_coords):
    if not 'face_d' in Flags:
        preview_window = croppedFace.copy()
    x, y, w = int(gaze_vector[0] * 12), int(gaze_vector[1] * 12), 160
    le = cv2.line(left_eye.copy(),
                  (x - w, y - w),
                  (x + w, y + w),
                  (255, 0, 255), 2)
    cv2.line(le, (x - w, y + w), (x + w, y - w), (255, 0, 255), 2)
    re = cv2.line(right_eye.copy(),
                 (x - w, y - w),
                 (x + w, y + w),
                 (255, 0, 255), 2)
    cv2.line(re, (x - w, y + w), (x + w, y - w), (255, 0, 255), 2)

    preview_window[eye_coords[0][1]:eye_coords[0][3],
                   eye_coords[0][0]:eye_coords[0][2]] = le

    preview_window[eye_coords[1][1]:eye_coords[1][3],
                   eye_coords[1][0]:eye_coords[1][2]] = re
   
   
def cam_or_video(inputs,inputFeeder):
    if inputs.lower()=="cam":
        inputFeeder=InputFeeder("cam")
    if not os.path.isfile(inputs):
        print("Unable to find input file")
        exit(1)
    inputFeeder = InputFeeder("video",inputs)
    return inputFeeder

def main():

    # args
    args = build_argparser().parse_args()
    arg_face_detect = args.face_detection
    arg_landmark = args.facial_landmark
    arg_gaze = args.gaze_estimation
    arg_head_pose = args.head_pose
    arg_cpu_extention = args.cpu_extension
    arg_device = args.device
    inputs = args.input
    Flags = args.Flags

    # imput cam or video file ? 
    inputFeeder = None
    feeder = cam_or_video(inputs,inputFeeder)

    # start timer
    start_loading = time.time()

    # load models and data
    face_det = Face_detect() 
    facial_l = Facial_landmarks() 
    gaze_e = Gaze_estimation() 
    head_p = Head_pose() 
    mouse_c = MouseController('medium','fast')
    feeder.load_data()

    face_det.load_model(arg_face_detect, arg_device, arg_cpu_extention)
    facial_l.load_model(arg_landmark, arg_device, arg_cpu_extention)
    gaze_e.load_model(arg_gaze, arg_device, arg_cpu_extention)
    head_p.load_model(arg_head_pose, arg_device, arg_cpu_extention)
    
    # loading timer 
    model_loading_time = time.time() - start_loading

    # counters 
    counter = 0
    frame_count = 0
    inference_time = 0
    start_inf_time = time.time()

    for ret, frame in feeder.next_batch():
        if not ret:
            break;
        if frame is not None:
            frame_count += 1
            if frame_count%5 == 0:
                cv2.imshow('video', cv2.resize(frame, (500, 500)))
            key = cv2.waitKey(60)
            start_inference = time.time()
            croppedFace, face_coords = face_det.predict(frame.copy(), args.prob_threshold)
            if type(croppedFace) == int:
                print("No face detected.")
                if key == 27:
                    break
                continue
            hp_out = head_p.predict(croppedFace.copy())
            left_eye, right_eye, eye_coords = facial_l.predict(croppedFace.copy())
            new_mouse_coord, gaze_vector = gaze_e.predict(left_eye, right_eye, hp_out)
            stop_inference = time.time()
            inference_time = inference_time + stop_inference - start_inference
            counter = counter + 1
            if (not len(Flags) == 0):
                preview_window = frame.copy()
                if 'face_d' in Flags: 
                    face_d(Flags,croppedFace,face_coords)
                if 'face_landmark_d' in Flags:
                    face_landmark_d(Flags,croppedFace,eye_coords)
                if 'head_pose_e' in Flags:
                    head_pose_e(Flags,preview_window,hp_out)
                if 'gaze_e' in Flags:
                    gaze_e(Flags,croppedFace,gaze_vector,left_eye,right_eye,eye_coords)
            if len(Flags) != 0:
                img_hor = np.hstack((cv2.resize(frame, (500, 500)),
                                     cv2.resize(preview_window, (500, 500))))
            else:
                img_hor = cv2.resize(frame, (500, 500))
            cv2.imshow('Visualization', img_hor)
            if frame_count%5 == 0:
                mouse_c.move(new_mouse_coord[0], new_mouse_coord[1]) 


    fps = frame_count / inference_time

    print("Benchmark: ")
    print("Total loading time of the models: {}".format(model_loading_time))
    print("total inference time {}".format(inference_time))
    print("fps {}".format(fps))
    cv2.destroyAllWindows()
    feeder.close()

if __name__ == '__main__':
    main()
