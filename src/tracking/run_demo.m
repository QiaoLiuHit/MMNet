%run the tracker on all videos in the "sequences" folder

%set pre-tranined model
model='./networks/VOT-TIR17/MMNet-VID-VOT-TIR17.mat';

% set tracking sequences
seqTir={
     'airplane'  
};
 
%run tracking
for s=1:numel(seqTir)
    run_MMNet(seqTir{s}, model);
end

