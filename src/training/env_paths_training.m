function opts = env_paths_training(opts)
%% for VID dataset
opts.rootDataDir = '/media/joe/000A68F0000410AD/ILSVRC2015_curated/Data/VID/train/'; % curated training data
opts.imdbVideoPath = '/media/joe/000FA49A000EC49F/Desktop/Joe/tracker_benchmark_v1.0/trackers/siamese_fc_master/training/imdb_video_100.mat';
opts.imageStatsPath = '/media/joe/000FA49A000EC49F/Desktop/Joe/tracker_benchmark_v1.0/trackers/siamese_fc_master/training/ILSVRC2015.stats.mat';

%%  for TIR dataset
%     opts.rootDataDir = '/media/joe/000A68F0000410AD/TIRDataset-DSNet-curated/TrainingData/';
%     opts.imdbVideoPath = '/media/joe/000A68F0000410AD/scripts/TIRDataset-curation/imdb_TIRvideo-MMNet.mat';
%     opts.imageStatsPath = '/media/joe/000A68F0000410AD/scripts/TIRDataset-curation/TIRDataset-MMNet-stats.mat';

end

