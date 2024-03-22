clear all;

%% Define paths to your dataset
dataFolder = 'speechImageData';
trainDataFolder = fullfile(dataFolder, 'TrainData');
valDataFolder = fullfile(dataFolder, 'ValData');

%% Create imageDatastores for training and validation datasets
adsTrain = imageDatastore(trainDataFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
adsValidation = imageDatastore(valDataFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% Define input size
inputSize = [98 50 1]; % Adjust as needed, including channel dimension for grayscale

%% Apply combined noise and SpecAugment augmentations
adsTrain.ReadFcn = @(x) combinedAugmentation(x, inputSize);
adsValidation.ReadFcn = @(x) combinedAugmentation(x, inputSize);

%% Define Network Architecture
numClasses = numel(categories(adsTrain.Labels)); % Use adsTrain to determine the number of classes dynamically
timePoolSize = ceil(inputSize(1)/8);
numF = 16; % Number of filters for the convolutional layers
dropoutProb = 0.2; % Dropout probability

layers = [
    imageInputLayer(inputSize)
    
    convolution2dLayer(3, numF, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3, 'Stride', 2, 'Padding', 'same')
    
    convolution2dLayer(3, 2*numF, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3, 'Stride', 2, 'Padding', 'same')
    
    convolution2dLayer(3, 4*numF, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3, 'Stride', 2, 'Padding', 'same')
    
    convolution2dLayer(3, 4*numF, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3, 4*numF, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([timePoolSize, 1])
    dropoutLayer(dropoutProb)
    
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

%% Specify Training Options with L2 Regularization
minibatchsize = 64;
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', minibatchsize, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'L2Regularization', 1e-4, ... % Example L2 regularization factor
    'ValidationData', adsValidation, ...
    'ValidationFrequency', floor(numel(adsTrain.Files)/minibatchsize));

%% Train Network
trainedNet = trainNetwork(adsTrain, layers, options);

%% Plot Confusion Matrix for Validation Set
% We'll adjust this part to calculate YValidation after the training
YValidation = classify(trainedNet, adsValidation);
TValidation = adsValidation.Labels;
figure('Units', 'normalized', 'Position', [0.2 0.2 0.5 0.5]);
cm = confusionchart(TValidation, YValidation, ...
    'Title', 'Confusion Matrix for Validation Data', ...
    'ColumnSummary', 'column-normalized', 'RowSummary', 'row-normalized');
sortClasses(cm, categories(adsTrain.Labels));

%% Combined Augmentation Function
function outputImage = combinedAugmentation(imagePath, inputSize)
    img = imread(imagePath);
    img = imresize(img, inputSize(1:2));
    img = double(img) / 255; % Normalize to [0, 1]

    % Add Gaussian noise
    noiseVariance = 0.01; % Adjust as needed
    img = addGaussianNoise(img, noiseVariance);

    % Apply SpecAugment
    img = applySpecAugment(img);

    outputImage = img;
end

%% Gaussian Noise Function
function imgWithNoise = addGaussianNoise(img, noiseVariance)
    noise = sqrt(noiseVariance) * randn(size(img));
    imgWithNoise = img + noise;
    imgWithNoise = max(min(imgWithNoise, 1), 0); % Clip values to [0, 1]
end

%% SpecAugment Function
function imgOut = applySpecAugment(imgIn)
    [height, width, ~] = size(imgIn);

    % Define the size of the masks
    freqMaskWidth = randi([1, floor(width * 0.15)], 1); % Up to 15% of the width
    timeMaskHeight = randi([1, floor(height * 0.15)], 1); % Up to 15% of the height

    % Define the starting points of the masks
    freqMaskStart = randi([1, width - freqMaskWidth + 1], 1);
    timeMaskStart = randi([1, height - timeMaskHeight + 1], 1);

    % Apply frequency and time masking
    imgOut = imgIn;
    imgOut(:, freqMaskStart:(freqMaskStart+freqMaskWidth-1)) = 0;
    imgOut(timeMaskStart:(timeMaskStart+timeMaskHeight-1), :) = 0;
end
