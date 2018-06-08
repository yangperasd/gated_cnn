#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function
# *************************************************
# > File Name: extraLayer.py
# > Author: yang 
# > Mail: yangperasd@163.com 
# > Created Time: Tue 15 May 2018 09:49:10 AM CST
# *************************************************
import coloredlogs, logging,verboselogs;
verboselogs.install();
coloredlogs.install(level='DEBUG',fmt='%(levelname)s %(message)s');
logger=logging.getLogger(__doc__);

from keras import backend as K 
from keras import activations
from keras.layers import Wrapper

class GatedConvBlock(Wrapper):
#{{{
    def __init__(self, conv_layer,
                 conv_num=3,
                 gate_activation='sigmoid',
                 **kwargs):
#{{{
        super(GatedConvBlock, self).__init__(conv_layer,**kwargs)
        self.conv_num= conv_num
        self.gate_activation = activations.get(gate_activation)
        self.conv_layers = []
        self.input_spec = conv_layer.input_spec 
        self.rank = conv_layer.rank
        if conv_layer.padding != 'same':
            raise ValueError("The padding mode of this layer must be `same`"
                            ", But found `{}`".format(self.padding))
        
        self.filters = conv_layer.filters//2
        #create conv layers 
        import copy
        for i in range(self.conv_num):
            new_conv_layer = copy.deepcopy(conv_layer)
            new_conv_layer.name = 'GatedConvBlock_{}_{}'.format(conv_layer.name, i)
            self.conv_layers.append(new_conv_layer) 

    #}}}
    def build(self, input_shape):
#{{{
        #ensure conv_layer's filter == input_shape[-1]*2
        if self.conv_layers[0].filters != input_shape[-1]*2:
            raise ValueError("For efficient, the sub-conv-layer's filters must be the twice of input_shape[-1].\nBut found filters={},input_shape[-1]={}".format(self.conv_layers[0].filters, input_shape[-1]))

        # call sublayer build 
        input_shape_current = input_shape
        for layer in self.conv_layers:
            with K.name_scope(layer.name):
                layer.build(input_shape_current)
                
            input_shape_current = input_shape
        self.built = True            
        
        pass;
#}}}
    def compute_output_shape(self, input_shape):
#{{{
        input_shape_current = input_shape
        for layer in self.conv_layers:
            input_shape_current = layer.compute_output_shape(input_shape_current)
            output_shape = list(input_shape_current)
            output_shape[-1] = int(output_shape[-1]/2)
            input_shape_current = output_shape   
        return tuple(input_shape_current)
#}}}
    def half_slice(self, x):
#{{{
        ndim = self.rank +2
        if ndim ==3:
            linear_output = x[:,:,:self.filters]
            gated_output = x[:,:,self.filters:]
        elif ndim ==4:
            linear_output = x[:,:,:,:self.filters]
            gated_output = x[:,:,:,self.filters:]
        elif ndim ==5:
            linear_output = x[:,:,:,:,:self.filters]
            gated_output = x[:,:,:,:,self.filters:]
        else:
            raise ValueError("This class only support for 1D, 2D, 3D conv, but the input's ndim={}".format(ndim))

        return linear_output, gated_output 
#}}}
    def call(self, inputs):
#{{{
        #from keras.layers import Lambda
        input_current = inputs  
        for i,layer in enumerate(self.conv_layers):
            output_current = layer(inputs= input_current)     
            #the output_current is 3D tensor 
            #use Lambda layer to ensure output be a keras tensor after slicing 
            linear_output, gated_output = self.half_slice(output_current)
            #linear_output = Lambda(lambda x:x[:,:,:self.filters],
                        #output_shape=
                            #lambda x:(x[:-1])+(self.filters,))(output_current)
            #gated_output = Lambda(lambda x:x[:,:,self.filters:],
                        #output_shape=
                            #lambda x:(x[:-1])+(self.filters,))(output_current)
            input_current = linear_output*self.gate_activation(gated_output)
            input_current._keras_shape = K.int_shape(linear_output)
            
        
        #residual connection
        output = input_current + inputs

        return output 
#}}}
    def get_weights(self):
#{{{
        weights = None 
        for layer in self.conv_layers:
            weights += layer.get_weights()
        return weights
#}}}
    def set_weights(self, weights):
#{{{
        for layer in self.conv_layers:
            layer.set_weights(weights)
        pass
#}}}
    @property
    def trainable_weights(self):
#{{{
        weights = []
        for layer in self.conv_layers:
            if hasattr(layer, 'trainable_weights'):
                weights += layer.trainable_weights
        return weights
        pass
#}}}
    @property
    def non_trainable_weights(self):
#{{{
        weights = []
        for layer in self.conv_layers:
            if hasattr(layer, 'non_trainable_weights'):
                weights += layer.non_trainable_weights
        return weights
        pass;
#}}}
    @property
    def updates(self):
#{{{
        updates_ = []
        for layer in self.conv_layers:
            if hasattr(layer, 'updates'):
                updates_ += layer.upates
        return updates_
        pass;
#}}}
    @property
    def losses(self):
#{{{
        losses_ = []
        for layer in self.conv_layers:
            if hasattr(layer, 'losses'):
                losses_ += layer.losses
        return losses_
        pass
#}}}
    @property
    def constraints(self):
#{{{
        constraints_ = {}
        for layer in self.conv_layers:
            if hasattr(layer, 'constraints'):
                constraints_.update(layer.constraints)
        return constraints_ 
#}}}
#}}}

