% Directory containing your WAV files
sourceDirectory = "custom_audio";
% Directory to save the cropped files
outputDirectory = "one_second_custom_audio";

% Get a list of WAV files in the source directory
files = dir(fullfile(sourceDirectory, '*.wav'));

for i = 1:length(files)
    % Full path to the current file
    filePath = fullfile(files(i).folder, files(i).name);
    
    % Read the audio file
    [audioData, sampleRate] = audioread(filePath);
    
    % Number of samples to keep (1 second worth)
    samplesToKeep = sampleRate * 1; % 1 second
    
    % Crop the audio to the first second
    croppedAudio = audioData(1:min(end, samplesToKeep), :);
    
    % Full path for the output file
    outputFilePath = fullfile(outputDirectory, files(i).name);
    
    % Write the cropped audio to a new file
    audiowrite(outputFilePath, croppedAudio, sampleRate);
end

disp('Cropping complete!');
