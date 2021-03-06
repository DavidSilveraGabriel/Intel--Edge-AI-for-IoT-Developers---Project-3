'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
import numpy as np
from openvino.inference_engine import IECore
import math


class Gaze_estimation:
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self):
        self.plugin = None
        self.network = None
        self.exec_net = None
        self.input_name = None
        self.input_shape = None
        self.output_name = None
        self.output_shape = None

    def load_model(self, model_name, device='CPU', extensions=None):
        '''
        TODO: You will need to complete this method
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.model_structure = self.model_name
        self.model_weights = self.model_name.split('.')[0]+'.bin'
        self.plugin = IECore()
        self.network = self.plugin.read_network(model = self.model_structure,
                                               weights = self.model_weights)
        self.supported_layers = self.plugin.query_network(network = self.network,
                                                     device_name = self.device)
        self.unsupported_layers = [l for l in self.network.layers.keys() if l not in self.supported_layers]

        if len(self.unsupported_layers) != 0 and self.device == 'CPU':
            print("unsupported layers found:{}".format(self.unsupported_layers))
            if not self.extensions == None:
                print("Adding cpu_extension")
                self.plugin.add_extension(self.extensions,
                                          self.device)
                self.supported_layers = self.plugin.query_network(network=self.network,
                                                                 device_name=self.device)
                self.unsupported_layers = [l for l in self.network.layers.keys() if l not in self.supported_layers]
                if len(unsupported_layers)!=0:
                    print("Issue still exists")
                    exit(1)
                print("Issue resolved after adding extensions")
            else:
                print("provide path of cpu extension")
                exit(1)
        #exec_net
        self.exec_net = self.plugin.load_network(network=self.network,
                                                 device_name=self.device,
                                                 num_requests=1)
        #input 
        self.input_name = [i for i in self.network.inputs.keys()]
        self.input_shape = self.network.inputs[self.input_name[1]].shape
        #output
        self.output_name = [i for i in self.network.outputs.keys()]

    def predict(self, left_eye_image, right_eye_image, head_pose_angle):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        le_img_processed, re_img_processed = self.preprocess_input(left_eye_image.copy(),
                                                                   right_eye_image.copy())
        outputs = self.exec_net.infer({'head_pose_angles':head_pose_angle,
                                       'left_eye_image':le_img_processed,
                                       'right_eye_image':re_img_processed})
        mouse_coords, gaze_vec = self.preprocess_output(outputs, head_pose_angle)
        return mouse_coords, gaze_vec


    def preprocess_input(self, left_eye_img, right_eye_img):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        imp_shape = (self.input_shape[3],
                     self.input_shape[2])
        ax = (0, 3, 1, 2)
        l_image_resized = cv2.resize(left_eye_img, imp_shape)
        l_img_processed = np.transpose(np.expand_dims(l_image_resized, axis=0), ax)

        r_image_resized = cv2.resize(right_eye_img, imp_shape)
        r_img_processed = np.transpose(np.expand_dims(r_image_resized, axis=0), ax)

        return l_img_processed,r_img_processed
        
    def preprocess_output(self, outputs, head_pose_angle):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        gaze_vec = outputs[self.output_name[0]].tolist()[0]
        angle_r_fc = head_pose_angle[2]
        cosine = math.cos(angle_r_fc*math.pi/180.0)
        sine = math.sin(angle_r_fc*math.pi/180.0)

        x_val = gaze_vec[0] * cosine + gaze_vec[1] * sine
        y_val = -gaze_vec[0] *  sine+ gaze_vec[1] * cosine
                
        return (x_val, y_val), gaze_vec
