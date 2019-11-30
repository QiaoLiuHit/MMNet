%input: video sequence name and path of the model
%output: tracking results
function bboxes = run_MMNet(video_name,model) 
   % add path and setting enviroment
    warning off;
    startup;
    paths = env_paths_tracking();
    % initial the tracking target's parameters  
    [img_files, pos, target_sz]=load_video_info(paths.video_base,video_name);
    tracker_params.imgFiles = vl_imreadjpeg(img_files,'numThreads', 12);
    tracker_params.targetPosition = pos;%[cy cx];
    tracker_params.targetSize = target_sz;%round([h w]);
    
    tracker_params.net = model;
    % Call the main tracking function
    [bboxes, fps] = tracker(tracker_params);
    fprintf('The average speed %f fps\n', fps);
end
