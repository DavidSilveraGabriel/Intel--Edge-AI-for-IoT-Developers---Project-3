'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
import numpy as np
from openvino.inference_engine import IECore


class Face_detect:
    '''
    Class for the Face Detection Model.
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
        self.exec_net = self.plugin.load_network(network = self.network,
                                                 device_name = self.device,
                                                 num_requests = 1)
        #imput
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        #output
        self.output_name = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_name].shape

    def predict(self, image, prob_threshold):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        img_processed = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.input_name : img_processed})
        coords = self.preprocess_output(outputs, prob_threshold)
        if (len(coords) == 0):
            return 0, 0
        coords = coords[0] 
        #                          //  width        //   height   //
        coords = coords * np.array([image.shape[1], image.shape[0],
                                    image.shape[1], image.shape[0]])
        coords = coords.astype(np.int32)

        cropped = image[coords[1]:coords[3],
                        coords[0]:coords[2]]
        return cropped, coords

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        imp = (self.input_shape[3], self.input_shape[2])
        image_resized = cv2.resize(image, imp)
        ax = (0, 3, 1, 2)
        img_processed = np.transpose(np.expand_dims(image_resized, axis = 0), ax)
        
        return img_processed
        
    def preprocess_output(self, outputs, prob_threshold):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        coords = []
        outputs = outputs[self.output_name][0][0]
        for output in outputs:
            confidence = output[2]
            if confidence >= prob_threshold:
                xmin,ymin,xmax,ymax = output[3],output[4],output[5],output[6]
                coords.append([xmin, ymin, xmax, ymax])
        
        return coords


