# Image_Identification

### Images

cam = webcam;

folder = 'Charlotte_Picture/';

subfolder = 'Charlotte/';

numImage = 100;

for i = 1:numImage

    img = snapshot(cam);
    
    image(img);
    
    cropImg = imcrop(img,[280,0,720,720]);
    
    image(cropImg);
    
    resizeImg = imresize(cropImg,1/10);
    
    image(resizeImg);
    
    pause(0.1);
    
    j = num2str(i);
    
    filename = strcat(folder,subfolder,"image_",j,".jpg");
    
    imwrite(resizeImg,filename);
    
end

### Distinguishing Images

imds = imageDatastore("Images/","IncludeSubfolders",true, ...

    "FileExtensions",".jpg",'LabelSource','foldernames');

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);

inputSize = [72 73 3];

numClasses = 2;

layers = [

    imageInputLayer(inputSize)
    
    convolution2dLayer(5,20)
    
    batchNormalizationLayer
    
    reluLayer
    
    fullyConnectedLayer(numClasses)
    
    softmaxLayer
    
    classificationLayer];

options = trainingOptions('sgdm', ...

    'MaxEpochs',5, ...
    
    'ValidationData',imdsValidation, ...
    
    'ValidationFrequency',1, ...
    
    'Verbose',false, ...
    
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers,options);

### Evaluation Data Set

YPred = classify(net,imdsValidation);

YValidation = imdsValidation.Labels;

accuracy = mean(YPred == YValidation)


save('Trained_Model.mat','net');

net = coder.loadDeepLearningNetwork('Trained_Model.mat');

imds = imageDatastore("Alex_Scarf/","IncludeSubfolders",true, ...

    "FileExtensions",".jpg",'LabelSource','foldernames');

YPred = classify(net,imds);

YValidation = imds.Labels;

accuracy = mean(YPred == YValidation)

Yconf = predict(net,imds);

### Visualising the Data

net = coder.loadDeepLearningNetwork('Trained_Model.mat');

numImage = 100;

folder = "Alex_Scarf/";

subfolder = "Alex/";

Folders = ["images/Alex/","images/Charlotte/", ...

           "Charlotte_Picture/Charlotte/","Alex_Scarf/Alex/"];
           
numFolders = numel(Folders);


hold on

for index = 1:numFolders

    images = zeros(72,73,3,100);
    
    for i = 1:numImage
    
        j = num2str(i);
        
        filename = strcat(Folders(index),"image_",j,".jpg");
        
        Image = imread(filename);
        
        images(:,:,:,i) = Image;
        
    end
    
    size(images)
    
    act = activations(net,images,"relu","OutputAs","rows");
    
    size(act)
    
    actReduced = tsne(act);
    
    size(actReduced)
    
    scatter(actReduced(:,1),actReduced(:,2));
end

legend(Folders);

### Detecting Anomalies

net = coder.loadDeepLearningNetwork('Trained_Model.mat');

TrueFolders = ["images/Alex/","images/Charlotte/"];

numFolders = numel(TrueFolders);

centroids = zeros(2,93840);

for i = 1:numFolders

    imds = imageDatastore(TrueFolders(i),"IncludeSubfolders",false, ...
    
    "FileExtensions",".jpg");
    
    act = activations(net,imds,"relu","OutputAs","rows");
    
    size(act);
    
    M = mean(act,1);
    
    size(M)
    
    size(centroids(i,:))
    
    centroids(i,:) = M;
    
end

numImage = 100;

Folders = ["images/Alex/","images/Charlotte/", ...

           "Charlotte_Picture/Charlotte/","Alex_Scarf/Alex/"];
hold on

threshold = 0.3;

for index = 1:numel(Folders)

    images = zeros(72,73,3,100);
    
    for i = 1:numImage
    
        j = num2str(i);
        
        filename = strcat(Folders(index),"image_",j,".jpg");
        
        Image = imread(filename);
        
        images(:,:,:,i) = Image;
        
    end
    act = activations(net,images,"relu","OutputAs","rows");
    
    activationSize = size(act);
    
    distances = zeros(1,activationSize(1));
   
    counter = 0;
    
    for k = 1:activationSize(1)
    
        actk = act(k);
        
        distanceA = norm(centroids(1)-actk);
        
        distanceC = norm(centroids(2)-actk);
        
        minimum = min([distanceA,distanceC]);
        
        distances(k) = minimum(1);
        
        if minimum > threshold
        
           counter = counter+1;
           
        end
        
    end
    
    histogram(distances,20);
    
    Folders(index)
    
    anomalyProp = counter/activationSize(1)
    
end

legend(Folders);

