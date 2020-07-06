'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
import numpy as np
from openvino.inference_engine import IECore


class Head_pose:
    '''
    Class for the Head Pose Estimation Model.
    '''
    def __init__(self):
        self.input_name = None
        self.input_shape = None
        self.output_name = None
        self.output_shape = None
        self.plugin = None
        self.network = None
        self.exec_net = None
        
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
        #imput
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
        img_processed = self.preprocess_input(image.copy())
        imput_name = {self.input_name : img_processed}
        outputs = self.exec_net.infer(imput_name)
        
        result = self.preprocess_output(outputs)
        
        return result


    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        imp_shape = (self.input_shape[3], self.input_shape[2])
        image_resized = cv2.resize(image, imp_shape)
        dims = np.expand_dims(image_resized, axis=0)
        ax = (0,3,1,2)
        img_processed = np.transpose(dims, ax)
        
        return img_processed
        
    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        output = []

        output.append(outputs['angle_y_fc'].tolist()[0][0])

        output.append(outputs['angle_p_fc'].tolist()[0][0])

        output.append(outputs['angle_r_fc'].tolist()[0][0])

        return output
