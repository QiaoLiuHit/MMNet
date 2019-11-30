classdef GlobalPooling < dagnn.Filter
  properties
    method = 'avg'
    poolSize=[1 1]
    opts = {'cuDNN'}
  end

  methods
    function outputs = forward(self, inputs, params)
%       [self.poolSize(1),self.poolSize(2),~]=size(inputs{1});
%       outputs{1} = vl_nnglobalpool(inputs{1},self.poolSize,...
%           'pad',self.pad, 'stride',self.stride, 'method', self.method, self.opts{:});
      outputs{1} = vl_nnglobalpool(inputs{1}, 'method', self.method) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      
%         [self.poolSize(1),self.poolSize(2),~]=size(inputs{1});
%         derInputs{1} = vl_nnglobalpool(inputs{1},self.poolSize, derOutputs{1},...
%         'pad',self.pad, 'stride',self.stride,  'method', self.method,self.opts{:}) ; 
      derInputs{1} = vl_nnglobalpool(inputs{1}, derOutputs{1}, 'method', self.method) ;
      derParams = {} ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.poolSize ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
    %  outputSizes = [1 1 inputSizes{1}(3) inputSizes{1}(4)];
     outputSizes{1}(1) = 1;outputSizes{1}(2) = 1;
     outputSizes{1}(3) = inputSizes{1}(3) ;
    end

    function obj = GlobalPooling(varargin)
      obj.load(varargin) ;
    end
  end
end
