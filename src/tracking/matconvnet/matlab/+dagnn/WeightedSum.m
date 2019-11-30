classdef WeightedSum < dagnn.ElementWise
% weighted sum
% X= a.* A + b.* B +....
% [a b ...] is the parameter weights

  properties (Transient)
    numInputs
  end

  methods
    function outputs = forward(obj, inputs, params)
      obj.numInputs = numel(inputs) ;
      weights = params{1};
      outputs{1} = inputs{1}.*weights(1) ;
      for k = 2:obj.numInputs
        outputs{1} = outputs{1} + inputs{k}.*weights(k) ;
      end
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      weights = params{1};
      for k = 1:obj.numInputs
        derInputs{k} = derOutputs{1}.*weights(k) ;
        temp = derOutputs{1}.*inputs{k};
        derWeights(k) = sum( temp(:) );
      end
      derParams{1} = derWeights ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = inputSizes{1} ;
      for k = 2:numel(inputSizes)
        if all(~isnan(inputSizes{k})) && all(~isnan(outputSizes{1}))
          if ~isequal(inputSizes{k}, outputSizes{1})
            warning('Sum layer: the dimensions of the input variables is not the same.') ;
          end
        end
      end
    end

    function rfs = getReceptiveFields(obj)
      obj.numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
      rfs.size = [1 1] ;
      rfs.stride = [1 1] ;
      rfs.offset = [1 1] ;
      rfs = repmat(rfs, obj.numInputs, 1) ;
    end

    function params = initParams(obj)
        temp = single(1 * ones(1,obj.numInputs));
        params{1} = temp;
    end
    
    function obj = WeightedSum(varargin)
      obj.load(varargin) ;
    end
  end
end