'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
import numpy as np
from openvino.inference_engine import IECore


class Facial_landmarks:
    '''
    Class for the Facial Landmark Detection Model.
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
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        #output
        self.output_name = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_name].shape

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        copy_img = image.copy()
        img_processed = self.preprocess_input(copy_img)
        outputs = self.exec_net.infer({self.input_name : img_processed})
        coords = self.preprocess_output(outputs) 
        #                          //  width        //   height   //
        coords = coords * np.array([image.shape[1], image.shape[0],
                                    image.shape[1], image.shape[0]])
        coords = coords.astype(np.int32)
        
        l_xmin = coords[0]-10
        l_ymin = coords[1]-10
        l_xmax = coords[0]+10
        l_ymax = coords[1]+10
        l = image[l_ymin:l_ymax,
                  l_xmin:l_xmax]

        r_xmin = coords[2]-10
        r_ymin = coords[3]-10
        r_xmax = coords[2]+10
        r_ymax = coords[3]+10
        r = image[r_ymin:r_ymax,
                  r_xmin:r_xmax]
        
        eye_coords = [[l_xmin, l_ymin, l_xmax, l_ymax],
                      [r_xmin, r_ymin, r_xmax, r_ymax]]
        
        return l, r, eye_coords


    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imp_shape = (self.input_shape[3], self.input_shape[2])
        image_resized = cv2.resize(image_cvt, imp_shape)
        ax = (0, 3, 1, 2)
        img_processed = np.transpose(np.expand_dims(image_resized, axis=0), ax)

        return img_processed
        
    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        outs = outputs[self.output_name][0]
        l_eye_x = outs[0].tolist()[0][0]
        l_eye_y = outs[1].tolist()[0][0]

        r_eye_x = outs[2].tolist()[0][0]
        r_eye_y = outs[3].tolist()[0][0]
        
        return (l_eye_x, l_eye_y, r_eye_x, r_eye_y)
