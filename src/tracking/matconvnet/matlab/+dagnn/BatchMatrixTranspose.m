classdef BatchMatrixTranspose < dagnn.Layer
% Z= permute(Z,[2 1 3 4]), Z is a four demensions matrix [H x W x C x B]

  methods
    function outputs = forward(obj, inputs, params)
        outputs{1}= permute(inputs{1},[2 1 3 4]);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)

      derInputs{1} = derOutputs{1};  
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = [inputSizes{1}(2) inputSizes{1}(1) inputSizes{1}(3) inputSizes{1}(4)]; 
    end

    function obj = BatchMatrixTranspose(varargin)
      obj.load(varargin) ;
    end
  end
end
