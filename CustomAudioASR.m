clear all;

%% Define paths to your dataset
dataFolder = 'customDataImage';
trainDataFolder = fullfile(dataFolder, 'TrainData');
valDataFolder = fullfile(dataFolder, 'ValData');

%% Create imageDatastores for training and validation datasets
% The 'LabelSource' parameter labels the images based on the folder names
adsTrain = imageDatastore(trainDataFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
adsValidation = imageDatastore(valDataFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% Define input size and resize the images in the datastores
inputSize = [98 50 1]; % Adjust as needed, including channel dimension for grayscale
adsTrain.ReadFcn = @(x)imresize(imread(x), inputSize(1:2));
adsValidation.ReadFcn = @(x)imresize(imread(x), inputSize(1:2));

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

%% Specify Training Options
minibatchsize = 64;
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', minibatchsize, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
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
