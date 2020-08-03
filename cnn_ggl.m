clear;
data=h5read('spectrs.hdf5','/Spec');
T=h5read('spectrs.hdf5','/Time');
%%
data=permute(data,[3,2,1,4]);
temp=zeros(227,227,3,14178);
%%
for i=1:14178
    temp(:,:,:,i)=padarray(data(:,:,:,i),[3 3],0,'pre');
end
train_set=randperm(14178);
data_train=temp(:,:,:,train_set(1:12000));
data_test=temp(:,:,:,train_set(12001:end));

%%

T_train=T(train_set(1:12000));
T_test=T(train_set(12001:end));

%% Google Net pretrained
net=alexnet;%googlenet;
lgraph = layerGraph(net.Layers);

lgraph=removeLayers(lgraph,{'drop7','fc8','prob','output'});
% numClasses=3;
newLayers = [
    dropoutLayer(0.5,'Name','newDropout')
    fullyConnectedLayer(200,'Name','fcn')
    dropoutLayer(0.5,'Name','newDropout2')
    fullyConnectedLayer(30,'Name','fcn1','WeightLearnRateFactor',5,'BiasLearnRateFactor',5)
    fullyConnectedLayer(1,'Name','fcn2','WeightLearnRateFactor',5,'BiasLearnRateFactor',5)
%     softmaxLayer('Name','softmax')
%     classificationLayer('Name','classoutput')
    regressionLayer('Name','reg')
    ];
lgraph = addLayers(lgraph,newLayers);

lgraph = connectLayers(lgraph,'relu7','newDropout');

%%
%New CNN
% netWidth = 224/2;
% layers = [
%     imageInputLayer([224 224 3],'Name','input')
%     convolution2dLayer(10,netWidth,'Padding','same','Name','convInp')
%     batchNormalizationLayer('Name','BNInp')
%     reluLayer('Name','reluInp')
%     
%     convolutionalUnit(netWidth,5,3,'S1U1')
%     additionLayer(2,'Name','add11')
%     reluLayer('Name','relu11')
%     convolutionalUnit(netWidth,8,1,'S1U2')
%     additionLayer(2,'Name','add12')
%     reluLayer('Name','relu12')
%     
%     convolutionalUnit(2*netWidth,5,2,'S2U1')
%     additionLayer(2,'Name','add21')
%     reluLayer('Name','relu21')
%     
%    
%     
%     convolutionalUnit(2*netWidth,3,1,'S2U2')
%     additionLayer(2,'Name','add22')
%     reluLayer('Name','relu22')
%     
%     convolutionalUnit(4*netWidth,10,2,'S3U1')
%     additionLayer(2,'Name','add31')
%     reluLayer('Name','relu31')
%     convolutionalUnit(4*netWidth,10,1,'S3U2')
%     additionLayer(2,'Name','add32')
%     reluLayer('Name','relu32')
%     
%     dropoutLayer(0.5,'Name','Dropout')
%     
%     averagePooling2dLayer(4,'Name','globalPool')
%     fullyConnectedLayer(1,'Name','fcFinal1')
%     reluLayer('Name','reluFinal')
% %     softmaxLayer('Name','softmax')
% %     classificationLayer('Name','classoutput')];
%     regressionLayer('Name','reg')];
% lgraph=layerGraph(layers);
% %%
% lgraph = connectLayers(lgraph,'reluInp','add11/in2');
% lgraph = connectLayers(lgraph,'relu11','add12/in2');
% %%
% skip1 = [
%     convolution2dLayer(1,2*netWidth,'Stride',2,'Name','skipConv1')
%     batchNormalizationLayer('Name','skipBN1')];
% 
% lgraph = addLayers(lgraph,skip1);
% lgraph = connectLayers(lgraph,'relu12','skipConv1');
% lgraph = connectLayers(lgraph,'skipBN1','add21/in2');
% lgraph = connectLayers(lgraph,'relu21','add22/in2');
% 
% skip2 = [
%     convolution2dLayer(1,4*netWidth,'Stride',2,'Name','skipConv2')
%     batchNormalizationLayer('Name','skipBN2')];
% lgraph = addLayers(lgraph,skip2);
% lgraph = connectLayers(lgraph,'relu22','skipConv2');
% lgraph = connectLayers(lgraph,'skipBN2','add31/in2');
% 
% lgraph = connectLayers(lgraph,'relu31','add32/in2');
%%
figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);
plot(lgraph);

%%
options = trainingOptions('adam',...
    'MiniBatchSize',15,...
    'MaxEpochs',100,...
    'InitialLearnRate',1e-3,...
    'ValidationData',{data_test,T_test},...
    'ValidationFrequency',400,...
    'Verbose',1,...
    'shuffle','every-epoch',...
    'ExecutionEnvironment','gpu',...
    'Plots','training-progress');
%%

rng default

trainedGN=trainNetwork(data_train,T_train,lgraph,options);
% trainedGN=trainNetwork(data_train,T_train,layers,options);

%%
x=0;
k=size(data_test);
k=k(4);

for i=1:k
x=x+abs(trainedGN.predict(data_test(:,:,:,i))-T_test(i));
end
x=x/k;
disp(x)
%%

function layers = convolutionalUnit(numF,size,stride,tag)
layers = [
    convolution2dLayer(size,numF,'Padding','same','Stride',stride,'Name',[tag,'conv1'])
    batchNormalizationLayer('Name',[tag,'BN1'])
    reluLayer('Name',[tag,'relu1'])
    convolution2dLayer(size,numF,'Padding','same','Name',[tag,'conv2'])
    batchNormalizationLayer('Name',[tag,'BN2'])];
end