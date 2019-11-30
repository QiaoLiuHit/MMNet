function [bboxes, speed] = tracker(varargin)
%% default hyper-params for MMNet.
p.join.method = 'corrfilt';
p.net =  '';
p.net_gray = '';
p.numScale = 3;
p.scaleStep = 1.0375;
p.scalePenalty = 0.9745;
p.scaleLR = 0.59;
p.responseUp = 8;
p.wInfluence = 0.2625; % influence of cosine window for displacement penalty
p.minSFactor = 0.2;
p.maxSFactor = 5;
p.zLR = 0.005; % update rate of the exemplar for the rolling avg (use very low values <0.015)
p.video = '';
p.visualization = true;
p.visualzation_feature = false; % used for visualizing feature map.
p.gpus = 1;
p.track_lost = [];
p.startFrame = 1;
p.fout = -1;
p.imgFiles = [];
p.targetPosition = [];
p.targetSize = [];
p.track_lost = [];
p.ground_truth = [];

%% params from the network architecture params
% they have to be consistent with the training
p.scoreSize = 33;
p.totalStride = 4;
p.contextAmount = 0.5; % context amount for the exemplar
p.subMean = false;
% prefix and ids
p.prefix_z = 'br1_'; % used to identify the layers of the exemplar
p.prefix_x = 'br2_'; % used to identify the layers of the instance
p.id_score = 'score';
p.trim_z_branch = {'br1_'};
p.trim_x_branch = {'br2_','join_xcorr','fin_adjust'};
p.init_gpu = true;
% Get environment-specific default paths.
p.paths = struct();
p.paths = env_paths_tracking(p.paths);
p = vl_argparse(p, varargin);

% network surgeries depend on the architecture
switch p.join.method
    case 'xcorr'
        p.trim_x_branch = {'br2_','join_xcorr','fin_'};
        p.trim_z_branch = {'br1_'};
        p.exemplarSize = 127;
        p.instanceSize = 255;
    case 'corrfilt'
        p.trim_x_branch = {'br2_','join_xcorr','fin_adjust','br1_globalpooling','br1_fc1'};
        p.trim_z_branch = {'br1_','join_cf','join_crop_z','join_conv2_dimred','join_conv_dimred2','join_a_deconv3',...
            'join_b_deconv3','join_conv_relation','join_reshape1','join_reshape2','join_reshape3',...
            'join_transpose','join_mm1','join_softmax1','join_mm2','join_reshape4','join_sum_SPR'};
        p.exemplarSize = 255;
        p.instanceSize = 255;
    otherwise
        error('network type unspecified');
end

% Load ImageNet Video statistics
stats = load(p.paths.stats);
im = single(p.imgFiles{p.startFrame});
if(size(im, 3)==1)
    if ~isempty(p.net_gray)
        p.net = p.net_gray;
        p.grayscale = true;
    end
end

%% Load pre-trained network
if ischar(p.net)
    % network has been passed as string
    net_path = [p.net];
    net_z = load(net_path,'net');
else
    % network has been passed as object
    net_z = p.net;
end

net_x = net_z;
% Load a second copy of the network for the second branch
net_z = dagnn.DagNN.loadobj(net_z.net);

% Sanity check
switch p.join.method
    case 'xcorr'
        assert(~find_layers_from_prefix(net_z, 'join_cf'), 'Check your join.method');
    case 'corrfilt'
        assert(find_layers_from_prefix(net_z, 'join_cf'), 'Check your join.method');
end
% create a full copy of the network, not just of the handle
net_x = dagnn.DagNN.loadobj(net_x.net);

net_z = init_net(net_z, p.gpus, p.init_gpu);
net_x = init_net(net_x, p.gpus, false);

nImgs = numel(p.imgFiles);

%% Divide the net in 2
% exemplar branch (only once per video) computes features for the target
for i=1:numel(p.trim_x_branch)
    remove_layers_from_prefix(net_z, p.trim_x_branch{i});
end

% instance branch computes features for search region and cross-correlates with z features
for i=1:numel(p.trim_z_branch)
    remove_layers_from_prefix(net_x, p.trim_z_branch{i});
end

z_out_id = net_z.getOutputs();
%%
if ~isempty(p.gpus)
    im = gpuArray(im);
end
% if grayscale repeat one channel to match filters size
if(size(im, 3)==1)
    im = repmat(im, [1 1 3]);
end
% Init visualization
videoPlayer = [];
if p.visualization && isToolboxAvailable('Computer Vision System Toolbox')
    videoPlayer = vision.VideoPlayer('Position', [100 100 [size(im,2), size(im,1)]+30]);
end

avgChans = gather([mean(mean(im(:,:,1))) mean(mean(im(:,:,2))) mean(mean(im(:,:,3)))]);

wc_z = p.targetSize(2) + p.contextAmount*sum(p.targetSize);
hc_z = p.targetSize(1) + p.contextAmount*sum(p.targetSize);
s_z = sqrt(wc_z*hc_z);
s_x = p.instanceSize/p.exemplarSize * s_z;
scales = (p.scaleStep .^ ((ceil(p.numScale/2)-p.numScale) : floor(p.numScale/2)));

scaledExemplar = s_z .* scales;
% initialize the exemplar
[z_crop, ~] = make_scale_pyramid(im, p.targetPosition, scaledExemplar, p.exemplarSize, avgChans, stats, p);
z_crop = z_crop(:,:,:,ceil(p.numScale/2));

if p.subMean
    z_crop = bsxfun(@minus, z_crop, reshape(stats.z.rgbMean, [1 1 3]));
end

net_z.eval({'exemplar', z_crop});
get_vars = @(net, ids) cellfun(@(id) net.getVar(id).value, ids, 'UniformOutput', false);
z_out_val = get_vars(net_z, z_out_id);

net_visual = net_z;

min_s_x = p.minSFactor*s_x;
max_s_x = p.maxSFactor*s_x;
min_s_z = p.minSFactor*s_z;
max_s_z = p.maxSFactor*s_z;

bboxes = zeros(nImgs, 4);
tot_eval_time = 0;
tot_z_time = 0;

switch p.join.method
    case 'corrfilt'
        p.id_score = 'join_out';
        p.id_score2 = 'join_out3';
        % Extract scores for join_out (pre fin_adjust to have them in the range 0-1)
        % net_x.vars(end-1).precious = true;
        net_x.vars(net_x.getVarIndex(p.id_score)).precious = true;
        net_x.vars(net_x.getVarIndex(p.id_score2)).precious = true;
end

% windowing to penalize large displacements
window = single(hann(p.scoreSize*p.responseUp) * hann(p.scoreSize*p.responseUp)');
window = window / sum(window(:));

%% Tracker main loop
scoreId = net_x.getVarIndex(p.id_score);
scoreId(2) = net_x.getVarIndex(p.id_score2);
overall_tic = tic;
for i = p.startFrame:nImgs
    if i>p.startFrame
        im = single(p.imgFiles{i});
        if ~isempty(p.gpus)
            im = gpuArray(im);
        end
        % if grayscale repeat one channel to match filters size
        if(size(im, 3)==1), im = repmat(im, [1 1 3]); end
        scaledInstance = s_x .* scales;
        % update instance with crop at new frame and previous position
        [x_crops, pad_masks_x] = make_scale_pyramid(im, p.targetPosition, scaledInstance, p.instanceSize, avgChans, stats, p);
        copy = @(v, n) cellfun(@(x) repmat(x, [1 1 1 n]), v, 'UniformOutput', false);
        z_out = interleave(z_out_id, copy(z_out_val, p.numScale));
        [newTargetPosition, newScale] = tracker_step(net_x, s_x, s_z, scoreId, z_out, x_crops, pad_masks_x, p.targetPosition, window, p);
        s_x = max(min_s_x, min(max_s_x, (1-p.scaleLR)*s_x + p.scaleLR*scaledInstance(newScale)));
        
        % update target position
        p.targetPosition = gather(newTargetPosition);
        
        % update the exemplar with crop at new frame and new position
        if p.zLR > 0
            scaledExemplar = s_z .* scales;
            [z_crop, ~] = make_scale_pyramid(im, p.targetPosition, scaledExemplar, p.exemplarSize, avgChans, stats, p);
            z_crop = z_crop(:,:,:,ceil(p.numScale/2));
            if p.subMean,   z_crop = bsxfun(@minu, z_crop, reshape(stats.z.rgbMean, [1 1 3])); end
            eval_z_tic = tic;
            net_z.eval({'exemplar', z_crop});
            eval_z_time = toc(eval_z_tic);
            tot_z_time = tot_z_time+eval_z_time;
            z_out_val_new = get_vars(net_z, z_out_id);
            
            % template update with rolling average
            update = @(curr, next) (1-p.zLR) * curr + p.zLR * next;
            z_out_val = arrayfun(@(i) update(z_out_val{i}, z_out_val_new{i}), ...
                1:numel(z_out_id), 'UniformOutput', false);
            
            s_z = max(min_s_z, min(max_s_z, (1-p.scaleLR)*s_z + p.scaleLR*scaledExemplar(newScale)));
        end
        
        % update target bbox
        scaledTarget = [p.targetSize(1) .* scales; p.targetSize(2) .* scales];
        p.targetSize = (1-p.scaleLR)*p.targetSize + p.scaleLR*[scaledTarget(1,newScale) scaledTarget(2,newScale)];
        
        % visualized feature map
        if p.visualzation_feature && i>1
            pathSave = '/your/setting/path';
            net_visual.conserveMemory = 0;
            net_visual.eval({'exemplar', x_crops(:,:,:,1)});
            feature_out_id{1,1} = 'br1_out';% TIR-specific discriminative feature (conv5)
            feature_out_id{1,2} = 'join_conv3_dimred'; % low-level feature (conv3)
            feature_out_id{1,3} = 'join_br1_conv3_WWfeat'; %fine-grained correlation feature (see Figure 3 in the paper)
            feature_maps = get_vars(net_visual, feature_out_id);
            fig_handle = figure(1);
            set(fig_handle, 'Position', [200, 200, 1000, 150]);
            figure(fig_handle);
            img = uint8(gather(x_crops(:,:,:,1)));
            subplot('Position',[0.1 0 0.15 1]),imagesc(img);
            axis off;
            for j=1:numel(feature_maps)
                % axis on;
                subplot('Position',[0.1+0.152*j 0 0.15 1]),imagesc(img);
                h = imagesc(imresize(sum(gather(feature_maps{1,j}),3),[255 255]));
                alpha(h,1);
                axis off;
                % hold off;
            end
            %hold off;
            %close(fig_handle);
            idxx = sprintf('%04d', i);
            saveas(gca,[pathSave idxx],'pdf');
            saveas(gca,[pathSave idxx],'png');
            saveas(gca,[pathSave idxx],'fig');
        end
        
    else
        % at the first frame output position and size passed as input (ground truth)
    end
    
    rectPosition = [p.targetPosition([2,1]) - p.targetSize([2,1])/2, p.targetSize([2,1])];
    %% output bbox in the original frame coordinates
    oTargetPosition = p.targetPosition;
    oTargetSize = p.targetSize;
    bboxes(i, :) = [oTargetPosition([2,1]) - oTargetSize([2,1])/2, oTargetSize([2,1])];
    
    if p.visualization
        if isempty(videoPlayer)
            figure(1), imshow(im/255);
            figure(1), rectangle('Position', rectPosition, 'LineWidth', 4, 'EdgeColor', 'y');
            drawnow
            fprintf('Frame %d\n', p.startFrame+i);
        else
            im = gather(im)/255;
            im = insertShape(im, 'Rectangle', rectPosition, 'LineWidth', 4, 'Color', 'yellow');
            % Display the annotated video frame using the video player object.
            step(videoPlayer, im);
        end
    end
    
    %stop the tracker on track loss (if a 'track_lost' function is specified)
    if ~isempty(p.track_lost) && p.track_lost(i, bboxes(i,:))
        break
    end
end
overall_time = toc(overall_tic);
n_frames_ontrack = sum(sum(bboxes==0,2)~=4);
if isempty(p.track_lost)
    speed = (nImgs-p.startFrame+1) / overall_time;
else
    speed = n_frames_ontrack/overall_time;
end
end
