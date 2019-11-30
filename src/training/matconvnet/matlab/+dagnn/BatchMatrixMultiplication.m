classdef BatchMatrixMultiplication < dagnn.Layer
% Z= X*Y , X Y and Z are the four demensions matrix [H x W x C x B], but
% only the first two demension are used for multipying
% C is channel number; B is batch number
% implemented by qiao

  methods
    function outputs = forward(obj, inputs, params) 
      batch = size(inputs{1},4);
      channel = size(inputs{1},3);
      for i=1:batch
          for j=1:channel
            outputs{1}(:,:,j,i) = inputs{1}(:,:,j,i) * inputs{2}(:,:,j,i) ;
          end
      end
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      batch = size(inputs{1},4);
      channel = size(inputs{1},3);
      for i=1:batch
          for j=1:channel
              derInputs{1}(:,:,j,i) = derOutputs{1}(:,:,j,i) * inputs{2}(:,:,j,i)' ; 
              derInputs{2}(:,:,j,i) = inputs{1}(:,:,j,i)' * derOutputs{1}(:,:,j,i) ;
          end
      end
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = [inputSizes{1}(1) inputSizes{2}(2) inputSizes{1}(3) inputSizes{1}(4)]; 
    end

    function obj = BatchMatrixMultiplication(varargin)
      obj.load(varargin) ;
    end
  end
end
