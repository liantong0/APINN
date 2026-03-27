function setup_paths()
    addpath(genpath(fullfile(fileparts(mfilename('fullpath')), 'src')));
    addpath(genpath(fullfile(fileparts(mfilename('fullpath')), 'data')));
end
