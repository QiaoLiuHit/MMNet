classdef Scale < dagnn.ElementWise
  properties
    size
    hasBias = true
  end

  methods

    function outputs = forward(obj, inputs, params)
      args = horzcat(inputs, params) ;
      outputs{1} = bsxfun(@times, args{1}, args{2}) ;
      if obj.hasBias
        outputs{1} = bsxfun(@plus, outputs{1}, args{3}) ;
      end
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      args = horzcat(inputs, params) ;
      
     
      sz_inputs2 = [size(args{2}) 1 1 1 1] ;
      sz_inputs2 = sz_inputs2(1:4) ;
      dargs{1} = bsxfun(@times, derOutputs{1}, args{2}) ;
      dargs{2} = derOutputs{1} .* args{1} ;
      for k = find(sz_inputs2 == 1)
        dargs{2} = sum(dargs{2}, k) ;
      end
      
      %added by lq--------------------
      sz_inputs1 = [size(args{1}) 1 1 1 1] ;
      sz_inputs1 = sz_inputs1(1:4) ;
      for k = find(sz_inputs1 == 1)
        dargs{1} = sum(dargs{1}, k) ;
      end
      %--------------------------------------
      
      
      if obj.hasBias
        dargs{3} = derOutputs{1} ;
        for k = find(sz_inputs2 == 1)
          dargs{3} = sum(dargs{3}, k) ;
        end
      end
      derInputs = dargs(1:numel(inputs)) ;
      derParams = dargs(numel(inputs)+(1:numel(params))) ;
    end
    
     function outputSizes = getOutputSizes(obj, inputSizes) %first parameter is the input feature
        outputSizes =  {max(inputSizes{1},inputSizes{2})};
     end
    

    function obj = Scale(varargin)
      obj.load(varargin) ;
    end
  end
end
