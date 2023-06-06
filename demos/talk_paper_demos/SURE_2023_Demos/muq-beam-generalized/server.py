import umbridge
import time
import os

class TestModel(umbridge.Model):

    def __init__(self):
        super().__init__("forward")

    def get_input_sizes(self, config):
        return [config["d"]] #default 3

    def get_output_sizes(self, config):
        return [config["d_output"]] #default 31

    def __call__(self, parameters, config):
        return [self.get_output_sizes(config)]
        #calculate integrand
    

    def supports_evaluate(self):
        return True

    def gradient(self,out_wrt, in_wrt, parameters, sens, config):
        return [2*sens[0]]

    def supports_gradient(self):
        return True

testmodel = TestModel()

umbridge.serve_models([testmodel], 4242)
