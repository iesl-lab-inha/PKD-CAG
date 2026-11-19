%% =========================================================
%  CONTINUOUS DATA COLLECTION (XDF via LabRecorder) + WEBCAM MP4
%  + SEQUENTIAL STIMULUS PRESENTATION (AVI/MP4) + EVENT LOG CSV
%
%  Acquisition sequence:
%   1) Sends "start" to LabRecorder (RCS) -> biosignals saved as .xdf
%   2) Starts continuous webcam recording -> one MP4 for whole session
%   3) Plays ALL stimulus videos (AVI/MP4) one-by-one from stimDir
%   4) Inserts baseline and rest black screens between clips
%   5) Writes an event log CSV with timestamps to segment later
%   6) Sends "stop" to LabRecorder
%
%  Requirements:
%   - LabRecorder running with Remote Control Server enabled (port 22345)
%   - EmotiBit streams already visible in LabRecorder
%   - MATLAB Webcam Support Package installed
%
%  NOTE:
%   - This logs events using MATLAB's tic/toc clock.
%   - Event timestamps are local; XDF alignment requires validation.
%% =========================================================

%% =========================
% SUBJECT SETTINGS
%% =========================
sub = "participant_001_session01";   % change per participant/session
baseDir = fullfile(pwd, "recordings");

% Stimulus video directory
stimDir = fullfile(pwd, "stimuli");

% Baseline/rest durations (seconds)
baseline_sec = 10;   % black screen before each clip
rest_sec     = 10;   % black screen after each clip

% Webcam settings
camFrameRate = 30;

% Display / playback pacing
% Enable approximate stimulus frame pacing.
% Disable pacing to process frames without an explicit playback pause.
pacePlayback = true;

%% =========================
% RCS SETTINGS (LabRecorder remote control)
%% =========================
rcsHost = "127.0.0.1";
rcsPort = 22345;

%% =========================
% CREATE OUTPUT PATHS
%% =========================
ts = string(datetime('now','Format','yyyyMMdd_HHmmss'));
sessionDir = fullfile(baseDir, sub, "Session_" + ts);

webcamOut = fullfile(sessionDir, "Webcam", sub + "_" + ts + "_webcam.mp4");
eventOut  = fullfile(sessionDir, "Logs",   sub + "_" + ts + "_events.csv");

if ~exist(fileparts(webcamOut), 'dir'); mkdir(fileparts(webcamOut)); end
if ~exist(fileparts(eventOut),  'dir'); mkdir(fileparts(eventOut));  end

%% =========================
% VERIFY STIMULUS DIRECTORY
%% =========================
stimDir = string(strtrim(char(stimDir)));  % remove accidental spaces
if ~isfolder(stimDir)
    error("Stimulus folder does not exist: " + stimDir);
end

%% =========================
% LOAD STIMULUS FILES (AVI + MP4; recursive)
%% =========================
aviFiles = dir(fullfile(stimDir, "**", "*.avi"));
mp4Files = dir(fullfile(stimDir, "**", "*.mp4"));
MP4Files = dir(fullfile(stimDir, "**", "*.MP4"));

files = [aviFiles; mp4Files; MP4Files];

if isempty(files)
    error("No .avi/.mp4 files found in: " + stimDir);
end

% Sort by name (IMPORTANT: prefix stimulus filenames with 01_, 02_, ... for order)
[~, idx] = sort({files.name});
files = files(idx);

disp("Found " + num2str(numel(files)) + " stimulus video(s).");
disp("First few files:");
for k = 1:min(5, numel(files))
    disp("  " + string(files(k).name));
end

%% =========================
% OPEN EVENT LOG CSV
%% =========================
fid = fopen(eventOut, 'w');
if fid == -1
    error("Could not open event log for writing: " + eventOut);
end
fprintf(fid, "t_sec,event,filename,fullpath\n");

logEvent = @(t, ev, fn, fp) fprintf(fid, "%.3f,%s,%s,%s\n", t, ev, fn, fp);

%% =========================
% CONNECT TO LABRECORDER (RCS) + START RECORDING
%% =========================
rcs = tcpclient(char(rcsHost), rcsPort, "Timeout", 5);

disp("Sending 'start' to LabRecorder...");
write(rcs, uint8([char("start") 10]));

%% =========================
% START WEBCAM RECORDING (CONTINUOUS)
%% =========================
cam = webcam;

vw = VideoWriter(char(webcamOut), "MPEG-4");
vw.FrameRate = camFrameRate;
open(vw);

%% =========================
% SETUP STIMULUS DISPLAY WINDOW
%% =========================
hFig = figure( ...
    'Color','k', ...
    'MenuBar','none', ...
    'ToolBar','none', ...
    'NumberTitle','off', ...
    'Name','Stimulus Presentation');

hAx = axes('Parent', hFig);
axis(hAx, 'off');

% maximize window (fullscreen-like)
try
    set(hFig, 'WindowState', 'maximized');
catch
    % older MATLAB versions may not support WindowState
end

% Determine a black frame size (fallback)
blackH = 720; blackW = 1280;
blackFrame = zeros(blackH, blackW, 3, 'uint8');

showBlack = @() imshow(blackFrame, 'Parent', hAx);

%% =========================
% RUN SESSION
%% =========================
t0 = tic;
logEvent(0, "session_start", "", "");

disp("Session started. Playing clips sequentially...");

for i = 1:numel(files)
    fname = string(files(i).name);
    fpath = string(fullfile(files(i).folder, files(i).name));

    disp("--------------------------------------------------");
    disp("Clip " + i + "/" + numel(files) + ": " + fname);

    % ---------- Baseline ----------
    t = toc(t0);
    logEvent(t, "baseline_start", fname, fpath);

    showBlack(); drawnow;

    tb = tic;
    while toc(tb) < baseline_sec
        camFrame = snapshot(cam);
        writeVideo(vw, camFrame);
        pause(1/camFrameRate);
    end

    t = toc(t0);
    logEvent(t, "baseline_end", fname, fpath);

    % ---------- Play Stimulus Clip ----------
    t = toc(t0);
    logEvent(t, "clip_start", fname, fpath);

    try
        v = VideoReader(char(fpath));
    catch ME
        warning("Could not open video: %s\nReason: %s", fpath, ME.message);
        t = toc(t0);
        logEvent(t, "clip_open_failed", fname, fpath);
        continue;
    end

    % Update black frame size to match stimulus if available
    try
        blackFrame = zeros(v.Height, v.Width, 3, 'uint8');
    catch
        % ignore if not available
    end

    while hasFrame(v) && isvalid(hFig)
        stimFrame = readFrame(v);
        imshow(stimFrame, 'Parent', hAx);
        drawnow;

        camFrame = snapshot(cam);
        writeVideo(vw, camFrame);

        if pacePlayback
            fr = v.FrameRate;
            if isempty(fr) || fr <= 0
                fr = 25; % fallback
            end
            pause(1/fr);
        end
    end

    t = toc(t0);
    logEvent(t, "clip_end", fname, fpath);

    % ---------- Rest ----------
    t = toc(t0);
    logEvent(t, "rest_start", fname, fpath);

    showBlack(); drawnow;

    tr = tic;
    while toc(tr) < rest_sec
        camFrame = snapshot(cam);
        writeVideo(vw, camFrame);
        pause(1/camFrameRate);
    end

    t = toc(t0);
    logEvent(t, "rest_end", fname, fpath);
end

t = toc(t0);
logEvent(t, "session_end", "", "");

%% =========================
% STOP RECORDING + CLEANUP
%% =========================
disp("Stopping webcam writer...");
close(vw);
clear cam;

disp("Sending 'stop' to LabRecorder...");
write(rcs, uint8([char("stop") 10]));
clear rcs;

fclose(fid);

if isvalid(hFig); close(hFig); end

disp("DONE.");
disp("Webcam video saved to: " + webcamOut);
disp("Event log saved to:   " + eventOut);
disp("Biosignals (.xdf) saved by LabRecorder in the folder you set in LabRecorder.")
