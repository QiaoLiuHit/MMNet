%DiConv
% Implements the dilated convolution operation with a dilate mask.
% Example:
%      DiConv('size', [fw,fh,channel,1], 'mask', mask, 'hasBias', true, 'pad',[rh,rh,rw,rw], 'stride', [1,1]);
%      mask should be the same size with the filter size and the 0's in the
%      mask indicate the dilate area of the filter.
% Xin Li, 2018
% -------------------------------------------------------------------------------------------------------------------------
classdef DiConv < dagnn.Layer

    properties
        opts = {'cuDNN'}
        hasBias=true
        mask = []
        size=[0 0 0 0]
        pad=[0 0 0 0]
        stride=[1 1]     
    end
     
 
    methods
         function outputs = forward(obj, inputs, params)
              if ~obj.hasBias, params{2} = [] ; end
              if isempty(obj.mask), error('The dilate mask field is not given! Please use help DiConv for detailed information. ' );end
              params{1}=params{1}.*obj.mask;  % set the dilate part as 0
              
              outputs{1} = vl_nnconv(...
                inputs{1}, params{1}, params{2}, ...
                'pad', obj.pad, ...
                'stride', obj.stride, ...
                obj.opts{:}) ;
         end

         function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
              if ~obj.hasBias, params{2} = [] ; end
              
              params{1} = params{1}.*obj.mask;
              
              [derInputs{1}, derParams{1}, derParams{2}] = vl_nnconv(...
                inputs{1}, params{1}, params{2}, derOutputs{1}, ...
                'pad', obj.pad, ...
                'stride', obj.stride, ...
                obj.opts{:}) ;
            
            derParams{1} = derParams{1}.*obj.mask;
         end

        function outputSizes = getOutputSizes(obj, inputSizes)
            z_sz = inputSizes{1};
            x_sz = inputSizes{2};
            y_sz = [x_sz(1:2) - z_sz(1:2) + 1, 1, z_sz(4)];
            outputSizes = {y_sz};
        end

        function rfs = getReceptiveFields(obj)
            rfs(1,1).size = [inf inf]; % could be anything
            rfs(1,1).stride = [1 1];
            rfs(1,1).offset = 1;
            rfs(2,1).size = [inf inf];
            rfs(2,1).stride = [1 1];
            rfs(2,1).offset = 1;
        end

        function obj = DiConv(varargin)
            obj.load(varargin);
            
            obj.mask = obj.mask;
            obj.mask(~(obj.mask==0))=1;     % make sure the elements are either 1 or 0
           
        end

    end

end
