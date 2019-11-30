function paths = env_paths_tracking(varargin)
    paths.net_base = './networks/'; % e.g. '/home/joe/mmnet/src/tracking/networks/';
    paths.stats = './ILSVRC2015.stats.mat'; % e.g.'/home/joe/mmnet/src/tracking/ILSVRC2015.stats.mat';
    paths.video_base = './sequences/'; % e.g.'/home/joe/mmnet/src/tracking/sequences/';
    paths = vl_argparse(paths, varargin);
end
