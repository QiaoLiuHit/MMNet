function net = make_siamese_3tasks(stream1, stream2, join, join2, final, final2, inputs, output, varargin)
% Constructs a multi-task Siamese network of two stream nets joined by the join and
% then followed by the final net.
% The stream and final nets can be simple or DAG.
% They should have one input and one output.
% The two streams should be identical except for parameter values.
% The join net is a DAG with 2 inputs and 1 output.
% The inputs must be named 'in1' and 'in2'.
% The inputs and output parameters are the variable names for the resulting net.
% The input names is a cell-array of 2 strings, the output names is a string.

opts.share_all = true;
% List of params to share, or layers whose params should be shared.
% Ignored if share_all is true.
opts.share_params = [];
opts.share_layers = [];
opts = vl_argparse(opts, varargin) ;

stream_outputs = {'br1_out', 'br2_out'};
join_output = 'join_out';


stream_outputs2 = {'br1_x9', 'br2_conv3_dimred_WWfeat'};% conv3 layer output
join_output2 = 'join_out3';


% Assume that indices of layers are preserved for share_layers.
if ~isa(stream1, 'dagnn.DagNN')
    stream1 = dagnn.DagNN.fromSimpleNN(stream1);
end
if ~isa(stream2, 'dagnn.DagNN')
    stream2 = dagnn.DagNN.fromSimpleNN(stream2);
end
if ~isempty(final)  && ~isempty(final2)
    if ~isa(final, 'dagnn.DagNN')  && ~isa(final2, 'dagnn.DagNN')
        final = dagnn.DagNN.fromSimpleNN(final);
        final2 = dagnn.DagNN.fromSimpleNN(final2);
    end
end
if isempty(final)
    join_output = output;
end

if opts.share_all
    opts.share_params = 1:numel(stream1.params);
else
    % Find all params that belong to share_layers.
    opts.share_params = union(opts.share_params, ...
                              params_of_layers(stream1, opts.share_layers));
end

net = dagnn.DagNN();
add_branches(net, stream1, stream2, inputs, stream_outputs, opts.share_params); %net name changed

%add reduce dimension layer on branch 2
net = addreduce_dimension(net,'br2_x9','br2_conv3_dimred');


%% used for ablation study
%holistic correlation module
%net= HC_br2(net, 'br2_conv3_dimred', 'br2_conv3_dimred_Wfeat');  

%pixel-level correlation module
%net = PC_br2(net,'br2_conv3_dimred','br2_conv3_dimred_WWfeat');
%%

net = FANet(net,'br2_conv3_dimred','br2_conv3_dimred_Wfeat1','br2_conv3_dimred_Wfeat2','br2_conv3_dimred_WWfeat');

add_join(net, join, {'in1', 'in2'}, stream_outputs, join_output);

add_join(net, join2, {'in1', 'in2'}, stream_outputs2, join_output2);

if ~isempty(final) && ~isempty(final2)
    add_final(net, final, join_output, output);
     add_final(net, final2, join_output2, 'score2');
end


%add class branch
add_classbranch(net, 'br1_out', 'br1_class_out');

end

function add_branches(net, stream1, stream2, inputs, outputs, share_inds)
% share_inds is a list of params to share.

    % Assume that both streams have the same names.
    orig_input = only(stream1.getInputs());
    orig_output = only(stream1.getOutputs());
    % Convert param indices to names.
    share_names = arrayfun(@(l) l.name, stream1.params(share_inds), ...
                           'UniformOutput', false);

    rename_unique1 = @(s) ['br1_', s];
    rename_unique2 = @(s) ['br2_', s];
    rename_common = @(s) ['br_', s];
    rename1 = struct(...
        'layer', rename_unique1, ...
        'var', rename_unique1, ...
        'param', @(s) rename_pred(s, @(x) ismember(x, share_names), ...
                                  rename_common, rename_unique1));
    rename2 = struct(...
        'layer', rename_unique2, ...
        'var', rename_unique2, ...
        'param', @(s) rename_pred(s, @(x) ismember(x, share_names), ...
                                  rename_common, rename_unique2));

    add_dag_to_dag(net, stream1, rename1);
    % Values of shared params will be taken from stream2
    % since add_dag_to_dag over-writes existing parameters.
    add_dag_to_dag(net, stream2, rename2);
    net.renameVar(rename_unique1(orig_input), inputs{1});
    net.renameVar(rename_unique2(orig_input), inputs{2});
    net.renameVar(rename_unique1(orig_output), outputs{1});
    net.renameVar(rename_unique2(orig_output), outputs{2});
end

function r = rename_pred(s, pred, rename_true, rename_false)
    if pred(s)
        r = rename_true(s);
    else
        r = rename_false(s);
    end
end

function add_join(net, join, orig_inputs, inputs, output)
    % assert(numel(join.getInputs()) == 2);
    orig_output = only(join.getOutputs());

    rename_join = @(s) ['join_', s];
    add_dag_to_dag(net, join, rename_join);
    for i = 1:2
        net.renameVar(rename_join(orig_inputs{i}), inputs{i});
    end
    net.renameVar(rename_join(orig_output), output);
end

function add_final(net, final, input, output)
    orig_inputs = final.getInputs();
    orig_outputs = final.getOutputs();
    assert(numel(orig_inputs) == 1);
    assert(numel(orig_outputs) == 1);
    orig_input = orig_inputs{1};
    orig_output = orig_outputs{1};

    rename_final = @(s) ['fin_', s];
    add_dag_to_dag(net, final, rename_final);
    net.renameVar(rename_final(orig_input), input);
    net.renameVar(rename_final(orig_output), output);
end

function param_inds = params_of_layers(net, layer_inds)
    layer_params = arrayfun(@(l) l.params, net.layers(layer_inds), ...
                            'UniformOutput', false);
    param_names = cat(2, {}, layer_params{:});
    param_inds = cellfun(@(s) net.getParamIndex(s), param_names);
    param_inds = unique(param_inds);
end

% reduce dimension
function net = addreduce_dimension(net,inputlayer,outputlayer)
    optss.scale =1; optss.weightInitMethod='xavierimproved';
   %reduce dimension br2_conv3  
    net.addLayer('br2_conv2_dimred', dagnn.Conv('size',[1 1 384 64],'pad',0,'stride',1,'hasBias',true), {inputlayer}, {outputlayer}, {'join_br_conv3f_dimred','join_br_conv3b_dimred'});
    net.params(net.getParamIndex('join_br_conv3f_dimred')).value =init_weight(optss, 1, 1, 384, 64, 'single');  %--->
    net.params(net.getParamIndex('join_br_conv3b_dimred')).value=zeros(64, 1, 'single');   
    
    
end

%add class branch
function add_classbranch(net, inputlayer, outputlayer)
    optss.scale = 1 ;
    optss.weightInitMethod = 'xavierimproved';
      
    net.addLayer('br1_globalpooling', dagnn.GlobalPooling(), inputlayer, 'br1_class_pool');
   
    %fc1 layer to suit 30 clas
    net.addLayer('br1_fc1', dagnn.Conv('size',[1 1 64 30],'pad',0,'stride',1,'hasBias',true), {'br1_class_pool'}, {outputlayer}, {'br1_convf_dimred','br1_convb_dimred'});
    net.params(net.getParamIndex('br1_convf_dimred')).value =init_weight(optss, 1, 1, 64, 30, 'single');  %--->
    net.params(net.getParamIndex('br1_convb_dimred')).value=zeros(30, 1, 'single');

end

%%%%------------------------------------used for ablation study---
% holistic correlation module (Fine-Hc), see Figure 2 of the paper
function net = HC_br2(net, inputlayer, outputlayer) 
channel=64; r=4;
net.addLayer('br2_conv11', dagnn.Conv('size', [7,7,channel,channel/r],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [2,2]), inputlayer, 'br2_conv_11', {'join_br1_conv11_f', 'join_br1_conv11_b'});

f = net.getParamIndex('join_br1_conv11_f') ;
net.params(f).value=single(randn(7,7,channel,channel/r) /sqrt(1*1*channel))/1e8;
net.params(f).learningRate=1;
net.params(f).weightDecay=10;

f = net.getParamIndex('join_br1_conv11_b') ;
net.params(f).value=single(zeros(channel/r,1));
net.params(f).learningRate=2;
net.params(f).weightDecay=10;

net.addLayer('br2_SA_relu1', dagnn.ReLU(),'br2_conv_11','br2_SA_relu_1');

net.addLayer('br2_conv12', dagnn.Conv('size', [5,5,channel/r,1],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'br2_SA_relu_1', 'br2_conv_12', {'join_br1_conv12_f', 'join_br1_conv12_b'});

f = net.getParamIndex('join_br1_conv12_f') ;
net.params(f).value=single(randn(5,5,channel/r,1) /sqrt(1*1*channel/r))/1e8;
net.params(f).learningRate=1;
net.params(f).weightDecay=10;

f = net.getParamIndex('join_br1_conv12_b') ;
net.params(f).value=single(zeros(1,1));
net.params(f).learningRate=2;
net.params(f).weightDecay=10;

net.addLayer('br2_SA_relu2', dagnn.ReLU(),'br2_conv_12','br2_SA_relu_2');

%add deconv layer for a_conv3
deconvblock=dagnn.ConvTranspose('size', [5,5,1,1], 'upsample', 1);
net.addLayer('br2_a_deconv3',deconvblock, {'br2_SA_relu_2'},{'br2_deconv3'},{'join_deconv3_f','join_deconv3_b'});
    
f = net.getParamIndex('join_deconv3_f') ;
net.params(f).value=single(randn(5,5,1,1) /sqrt(1*1*1))/1e8;
net.params(f).learningRate=1;
net.params(f).weightDecay=10;

f = net.getParamIndex('join_deconv3_b') ;
net.params(f).value=single(zeros(1,1));
net.params(f).learningRate=2;
net.params(f).weightDecay=10;

net.addLayer('br2_SA_relu3', dagnn.ReLU(),'br2_deconv3','br2_SA_relu_3');

%add deconv2
deconvblock=dagnn.ConvTranspose('size', [7,7,1,1], 'upsample', 2);
net.addLayer('br2_b_deconv3',deconvblock, {'br2_SA_relu_3'},{'br2_b_deconv3'},{'join_b_deconv3_f','join_b_deconv3_b'});
    
f = net.getParamIndex('join_b_deconv3_f') ;
net.params(f).value=single(randn(7,7,1,1) /sqrt(1*1*1))/1e8;
net.params(f).learningRate=1;
net.params(f).weightDecay=10;

f = net.getParamIndex('join_b_deconv3_b') ;
net.params(f).value=single(zeros(1,1));
net.params(f).learningRate=2;
net.params(f).weightDecay=10;

net.addLayer('br2_sigmoid1', dagnn.Sigmoid(),'br2_b_deconv3','br2_sigmoid_1');
net.addLayer('br2_scale1',dagnn.Scale('hasBias',0),{inputlayer,'br2_sigmoid_1'},outputlayer);
end

% pixel-lelvel correlation module (Fine-Pc), see Figure 2 of the paper
function net = PC_br2(net,inputlayer,outputlayer)
optss.scale =1; optss.weightInitMethod='xavierimproved';
channel=64; H =53; W= 53;

%conv1
net.addLayer('br2_conv1_SPR', dagnn.Conv('size', [1,1,channel,channel],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), inputlayer, 'br2_conv1_SPR', {'join_br1_conv1_SPR_f', 'join_br1_conv1_SPR_b'});

net.params(net.getParamIndex('join_br1_conv1_SPR_f')).value =init_weight(optss, 1, 1, channel, channel, 'single');  %--->
net.params(net.getParamIndex('join_br1_conv1_SPR_b')).value=zeros(channel, 1, 'single');

net.addLayer('br2_BN_SPR', dagnn.BatchNorm('numChannels', channel), {'br2_conv1_SPR'}, {'br2_Norm_SPR'},{'join_br1_bn_SPR_f', 'join_br1_bn_SPR_b', 'join_br1_bn_SPR_m'});
net.params(net.getParamIndex('join_br1_bn_SPR_f')).value= ones(channel, 1, 'single');
net.params(net.getParamIndex('join_br1_bn_SPR_b')).value=zeros(channel, 1, 'single');
net.params(net.getParamIndex('join_br1_bn_SPR_m')).value=zeros(channel, 2, 'single');
net.addLayer('br2_SPR_relu', dagnn.ReLU(), 'br2_Norm_SPR', 'br2_SPR_relu_out');

net.addLayer('br2_reshape1',dagnn.Reshape('shape',{H*W channel 1}),'br2_SPR_relu_out','br2_conv1_SPR_reshape');

%conv2
net.addLayer('br2_conv2_SPR', dagnn.Conv('size', [1,1,channel,channel],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), inputlayer, 'br2_conv2_SPR', {'join_br1_conv2_SPR_f', 'join_br1_conv2_SPR_b'});

net.params(net.getParamIndex('join_br1_conv2_SPR_f')).value =init_weight(optss, 1, 1, channel, channel, 'single');  %--->
net.params(net.getParamIndex('join_br1_conv2_SPR_b')).value=zeros(channel, 1, 'single');

net.addLayer('br2_BN2_SPR', dagnn.BatchNorm('numChannels', channel), {'br2_conv2_SPR'}, {'br2_Norm2_SPR'},{'join_br1_bn2_SPR_f', 'join_br1_bn2_SPR_b', 'join_br1_bn2_SPR_m'});
net.params(net.getParamIndex('join_br1_bn2_SPR_f')).value= ones(channel, 1, 'single');
net.params(net.getParamIndex('join_br1_bn2_SPR_b')).value=zeros(channel, 1, 'single');
net.params(net.getParamIndex('join_br1_bn2_SPR_m')).value=zeros(channel, 2, 'single');
net.addLayer('br2_SPR_relu2', dagnn.ReLU(), 'br2_Norm2_SPR', 'br2_SPR_relu2_out');

net.addLayer('br2_reshape2',dagnn.Reshape('shape',{H*W channel 1}),'br2_SPR_relu2_out','br2_conv2_SPR_reshape');
net.addLayer('br2_transpose',dagnn.BatchMatrixTranspose(),'br2_conv2_SPR_reshape', 'br2_conv2_SPR_reshape_transpose');

%matrix multiply 1
net.addLayer('br2_mm1',dagnn.BatchMatrixMultiplication(),{'br2_conv1_SPR_reshape','br2_conv2_SPR_reshape_transpose'},'br2_MM1_out');

net.addLayer('br2_softmax1',dagnn.SoftMax(),'br2_MM1_out','br2_MM1_out_softmax'); %size is (H*W)x(H*W)


%conv3
net.addLayer('br2_conv3_SPR', dagnn.Conv('size', [1,1,channel,channel],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), inputlayer, 'br2_conv3_SPR', {'join_br1_conv3_SPR_f', 'join_br1_conv3_SPR_b'});

net.params(net.getParamIndex('join_br1_conv3_SPR_f')).value =init_weight(optss, 1, 1, channel, channel, 'single');  %--->
net.params(net.getParamIndex('join_br1_conv3_SPR_b')).value=zeros(channel, 1, 'single');

net.addLayer('br2_BN3_SPR', dagnn.BatchNorm('numChannels', channel), {'br2_conv3_SPR'}, {'br2_Norm3_SPR'},{'join_br1_bn3_SPR_f', 'join_br1_bn3_SPR_b', 'join_br1_bn3_SPR_m'});
net.params(net.getParamIndex('join_br1_bn3_SPR_f')).value= ones(channel, 1, 'single');
net.params(net.getParamIndex('join_br1_bn3_SPR_b')).value=zeros(channel, 1, 'single');
net.params(net.getParamIndex('join_br1_bn3_SPR_m')).value=zeros(channel, 2, 'single');
net.addLayer('br2_SPR_relu3', dagnn.ReLU(), 'br2_Norm3_SPR', 'br2_SPR_relu3_out');

net.addLayer('br2_reshape3',dagnn.Reshape('shape',{H*W channel 1}),'br2_SPR_relu3_out','br2_conv3_SPR_reshape');



%matrix multiply 2
net.addLayer('br2_mm2',dagnn.BatchMatrixMultiplication(),{'br2_MM1_out_softmax','br2_conv3_SPR_reshape'},'br2_MM2_out'); %size  is (HW*channel )

net.addLayer('br2_reshape4',dagnn.Reshape('shape',{H W channel}),'br2_MM2_out','br2_MM2_out_reshape');% size is ( H x W x channel)

%1x1conv
net.addLayer('br2_conv4_SPR', dagnn.Conv('size', [1,1,channel,channel],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'br2_MM2_out_reshape', 'br2_MM2_out_reshape_rescale', {'join_br1_conv4_SPR_f', 'join_br1_conv4_SPR_b'});
net.params(net.getParamIndex('join_br1_conv4_SPR_f')).value =init_weight(optss, 1, 1, channel, channel, 'single');  %--->
net.params(net.getParamIndex('join_br1_conv4_SPR_b')).value=zeros(channel, 1, 'single');

%sum
net.addLayer('br2_sum_SPR',dagnn.Sum(),{inputlayer,'br2_MM2_out_reshape_rescale'},outputlayer);
end
%%%%------------------------------------------------------------------------

% Fine-grained aware network (FANet, equal to Figure 2)
function net = FANet(net,inputlayer,outputlayer_spr,outputlayer_sa,outputlayer_sum)

optss.scale =1; optss.weightInitMethod='xavierimproved';
channel=64; H =53; W= 53; r=4;

%conv1
net.addLayer('br2_conv1_SPR', dagnn.Conv('size', [1,1,channel,channel],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), inputlayer, 'br2_conv1_SPR', {'join_br1_conv1_SPR_f', 'join_br1_conv1_SPR_b'});

net.params(net.getParamIndex('join_br1_conv1_SPR_f')).value =init_weight(optss, 1, 1, channel, channel, 'single');  %--->
net.params(net.getParamIndex('join_br1_conv1_SPR_b')).value=zeros(channel, 1, 'single');

net.addLayer('br2_BN_SPR', dagnn.BatchNorm('numChannels', channel), {'br2_conv1_SPR'}, {'br2_Norm_SPR'},{'join_br1_bn_SPR_f', 'join_br1_bn_SPR_b', 'join_br1_bn_SPR_m'});
net.params(net.getParamIndex('join_br1_bn_SPR_f')).value= ones(channel, 1, 'single');
net.params(net.getParamIndex('join_br1_bn_SPR_b')).value=zeros(channel, 1, 'single');
net.params(net.getParamIndex('join_br1_bn_SPR_m')).value=zeros(channel, 2, 'single');
net.addLayer('br2_SPR_relu', dagnn.ReLU(), 'br2_Norm_SPR', 'br2_SPR_relu_out');

net.addLayer('br2_reshape1',dagnn.Reshape('shape',{H*W channel 1}),'br2_SPR_relu_out','br2_conv1_SPR_reshape');

%conv2
net.addLayer('br2_conv2_SPR', dagnn.Conv('size', [1,1,channel,channel],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), inputlayer, 'br2_conv2_SPR', {'join_br1_conv2_SPR_f', 'join_br1_conv2_SPR_b'});

net.params(net.getParamIndex('join_br1_conv2_SPR_f')).value =init_weight(optss, 1, 1, channel, channel, 'single');  %--->
net.params(net.getParamIndex('join_br1_conv2_SPR_b')).value=zeros(channel, 1, 'single');

net.addLayer('br2_BN2_SPR', dagnn.BatchNorm('numChannels', channel), {'br2_conv2_SPR'}, {'br2_Norm2_SPR'},{'join_br1_bn2_SPR_f', 'join_br1_bn2_SPR_b', 'join_br1_bn2_SPR_m'});
net.params(net.getParamIndex('join_br1_bn2_SPR_f')).value= ones(channel, 1, 'single');
net.params(net.getParamIndex('join_br1_bn2_SPR_b')).value=zeros(channel, 1, 'single');
net.params(net.getParamIndex('join_br1_bn2_SPR_m')).value=zeros(channel, 2, 'single');
net.addLayer('br2_SPR_relu2', dagnn.ReLU(), 'br2_Norm2_SPR', 'br2_SPR_relu2_out');

net.addLayer('br2_reshape2',dagnn.Reshape('shape',{H*W channel 1}),'br2_SPR_relu2_out','br2_conv2_SPR_reshape');
net.addLayer('br2_transpose',dagnn.BatchMatrixTranspose(),'br2_conv2_SPR_reshape', 'br2_conv2_SPR_reshape_transpose');

%matrix multiply 1
net.addLayer('br2_mm1',dagnn.BatchMatrixMultiplication(),{'br2_conv1_SPR_reshape','br2_conv2_SPR_reshape_transpose'},'br2_MM1_out');

net.addLayer('br2_softmax1',dagnn.SoftMax(),'br2_MM1_out','br2_MM1_out_softmax'); %size is (H*W)x(H*W)
%conv3
net.addLayer('br2_conv3_SPR', dagnn.Conv('size', [1,1,channel,channel],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), inputlayer, 'br2_conv3_SPR', {'join_br1_conv3_SPR_f', 'join_br1_conv3_SPR_b'});

net.params(net.getParamIndex('join_br1_conv3_SPR_f')).value =init_weight(optss, 1, 1, channel, channel, 'single');  %--->
net.params(net.getParamIndex('join_br1_conv3_SPR_b')).value=zeros(channel, 1, 'single');

net.addLayer('br2_BN3_SPR', dagnn.BatchNorm('numChannels', channel), {'br2_conv3_SPR'}, {'br2_Norm3_SPR'},{'join_br1_bn3_SPR_f', 'join_br1_bn3_SPR_b', 'join_br1_bn3_SPR_m'});
net.params(net.getParamIndex('join_br1_bn3_SPR_f')).value= ones(channel, 1, 'single');
net.params(net.getParamIndex('join_br1_bn3_SPR_b')).value=zeros(channel, 1, 'single');
net.params(net.getParamIndex('join_br1_bn3_SPR_m')).value=zeros(channel, 2, 'single');
net.addLayer('br2_SPR_relu3', dagnn.ReLU(), 'br2_Norm3_SPR', 'br2_SPR_relu3_out');

net.addLayer('br2_reshape3',dagnn.Reshape('shape',{H*W channel 1}),'br2_SPR_relu3_out','br2_conv3_SPR_reshape');

%matrix multiply 2
net.addLayer('br2_mm2',dagnn.BatchMatrixMultiplication(),{'br2_MM1_out_softmax','br2_conv3_SPR_reshape'},'br2_MM2_out'); %size  is (HW*channel )

net.addLayer('br2_reshape4',dagnn.Reshape('shape',{H W channel}),'br2_MM2_out','br2_MM2_out_reshape');% size is ( H x W x channel)

%1x1conv
net.addLayer('br2_conv4_SPR', dagnn.Conv('size', [1,1,channel,channel],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'br2_MM2_out_reshape', 'br2_MM2_out_reshape_rescale', {'join_br1_conv4_SPR_f', 'join_br1_conv4_SPR_b'});
net.params(net.getParamIndex('join_br1_conv4_SPR_f')).value =init_weight(optss, 1, 1, channel, channel, 'single');  %--->
net.params(net.getParamIndex('join_br1_conv4_SPR_b')).value=zeros(channel, 1, 'single');

%sum
net.addLayer('br2_sum_SPR',dagnn.Sum(),{inputlayer,'br2_MM2_out_reshape_rescale'},outputlayer_spr);


%sa net************************\
net.addLayer('br2_conv11', dagnn.Conv('size', [7,7,channel,channel/r],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [2,2]), inputlayer, 'br2_conv_11', {'join_br1_conv11_f', 'join_br1_conv11_b'});

f = net.getParamIndex('join_br1_conv11_f') ;
net.params(f).value=single(randn(7,7,channel,channel/r) /sqrt(1*1*channel))/1e8;
net.params(f).learningRate=1;
net.params(f).weightDecay=10;

f = net.getParamIndex('join_br1_conv11_b') ;
net.params(f).value=single(zeros(channel/r,1));
net.params(f).learningRate=2;
net.params(f).weightDecay=10;

net.addLayer('br2_SA_relu1', dagnn.ReLU(),'br2_conv_11','br2_SA_relu_1');

net.addLayer('br2_conv12', dagnn.Conv('size', [5,5,channel/r,1],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'br2_SA_relu_1', 'br2_conv_12', {'join_br1_conv12_f', 'join_br1_conv12_b'});

f = net.getParamIndex('join_br1_conv12_f') ;
net.params(f).value=single(randn(5,5,channel/r,1) /sqrt(1*1*channel/r))/1e8;
net.params(f).learningRate=1;
net.params(f).weightDecay=10;

f = net.getParamIndex('join_br1_conv12_b') ;
net.params(f).value=single(zeros(1,1));
net.params(f).learningRate=2;
net.params(f).weightDecay=10;

net.addLayer('br2_SA_relu2', dagnn.ReLU(),'br2_conv_12','br2_SA_relu_2');

%add deconv layer for a_conv3
deconvblock=dagnn.ConvTranspose('size', [5,5,1,1], 'upsample', 1);
net.addLayer('br2_a_deconv3',deconvblock, {'br2_SA_relu_2'},{'br2_deconv3'},{'join_deconv3_f','join_deconv3_b'});
    
f = net.getParamIndex('join_deconv3_f') ;
net.params(f).value=single(randn(5,5,1,1) /sqrt(1*1*1))/1e8;
net.params(f).learningRate=1;
net.params(f).weightDecay=10;

f = net.getParamIndex('join_deconv3_b') ;
net.params(f).value=single(zeros(1,1));
net.params(f).learningRate=2;
net.params(f).weightDecay=10;

net.addLayer('br2_SA_relu3', dagnn.ReLU(),'br2_deconv3','br2_SA_relu_3');

%add deconv2
deconvblock=dagnn.ConvTranspose('size', [7,7,1,1], 'upsample', 2);
net.addLayer('br2_b_deconv3',deconvblock, {'br2_SA_relu_3'},{'br2_b_deconv3'},{'join_b_deconv3_f','join_b_deconv3_b'});
    
f = net.getParamIndex('join_b_deconv3_f') ;
net.params(f).value=single(randn(7,7,1,1) /sqrt(1*1*1))/1e8;
net.params(f).learningRate=1;
net.params(f).weightDecay=10;

f = net.getParamIndex('join_b_deconv3_b') ;
net.params(f).value=single(zeros(1,1));
net.params(f).learningRate=2;
net.params(f).weightDecay=10;

net.addLayer('br2_sigmoid1', dagnn.Sigmoid(),'br2_b_deconv3','br2_sigmoid_1');
net.addLayer('br2_scale1',dagnn.Scale('hasBias',0),{inputlayer,'br2_sigmoid_1'},outputlayer_sa);


%% fusion
net.addLayer('br2_concat',dagnn.Concat(),{outputlayer_spr, outputlayer_sa},'br2_fusion');
net.addLayer('br2_conv_fusion', dagnn.Conv('size', [1,1,2*channel,channel],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'br2_fusion', outputlayer_sum, {'join_br1_conv_fusion_f', 'join_br1_conv_fusion_b'});
net.params(net.getParamIndex('join_br1_conv_fusion_f')).value =init_weight(optss, 1, 1, 2*channel, channel, 'single');  %--->
net.params(net.getParamIndex('join_br1_conv_fusion_b')).value=zeros(channel, 1, 'single');

end