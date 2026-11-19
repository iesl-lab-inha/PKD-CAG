%% =========================================================
% SUBJECT AND RECORDING SETTINGS
%% =========================================================
sub = "participant_001_Anger";
duration_sec = 10;

%% =========================================================
% RECORDING OUTPUT DIRECTORY
%% =========================================================
baseDir = fullfile(pwd, "recordings");

%% =========================================================
% LABRECORDER REMOTE CONTROL SETTINGS
%% =========================================================
rcsHost = "127.0.0.1";
rcsPort = 22345;

%% =========================================================
% VIDEO OUTPUT PATH (AUTO-GENERATED)
%% =========================================================
ts = string(datetime('now','Format','yyyyMMdd_HHmmss'));
videoOut = fullfile(baseDir, sub, "Video", sub + "_" + ts + ".mp4");

if ~exist(fileparts(videoOut), 'dir')
    mkdir(fileparts(videoOut));
end

%% =========================================================
% CONNECT TO LABRECORDER (RCS)
%% =========================================================
rcs = tcpclient(char(rcsHost), rcsPort, "Timeout", 5);

%% =========================================================
% START RECORDING (BIOSIGNALS) - PATH SET IN LABRECORDER UI
%% =========================================================
disp("Sending start...");
write(rcs, uint8([char("start") 10]));

%% =========================================================
% START WEBCAM RECORDING (VIDEO)
%% =========================================================
cam = webcam;
vw = VideoWriter(char(videoOut), "MPEG-4");
vw.FrameRate = 30;
open(vw);

t0 = tic;
while toc(t0) < duration_sec
    frame = snapshot(cam);
    writeVideo(vw, frame);
end

close(vw);
clear cam;

%% =========================================================
% STOP RECORDING (BIOSIGNALS)
%% =========================================================
disp("Sending stop...");
write(rcs, uint8([char("stop") 10]));

%% =========================================================
% CLEANUP
%% =========================================================
clear rcs;

disp("DONE: LabRecorder stopped and video saved successfully.");
disp("Video file:");
disp(videoOut);
disp("Biosignals (.xdf) are saved by LabRecorder in the folder you set in LabRecorder.");
