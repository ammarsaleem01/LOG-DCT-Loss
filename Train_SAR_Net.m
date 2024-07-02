clear all;clc;close all;
%% Define intial vairbles for starting loop
addpath(genpath(fullfile(cd,'Utility')))
% for NetCoice_i = [0 1 4 5]+20% [20;30;50;60]
% NetChoice: DnCNN with 20: L2-loss, 21: L1-loss, 24:DCT-L2-loss, 25:LOG-DCT-L2-loss
NetCoice_i = 20;% [0 1 4 5]+20% [20;30;50;60]
for aaX = 1:3
if floor(NetCoice_i/10)==2
    MiniBatchSize = 40;
    LearnRateDropPeriod = 1;
    LearnRateDropFactor = 0.9;
    InitialLearnRate = 1e-2;
    VerboseFrequency = 500 ;
    ValidationFrequency = 1000;
end

MaxEpochs = 10;
% Trained on GPU-RTX-A4000-16GB, CPU-Core-i9-12900-32GB-RAM
ExecutionEnvironment = 'multi-gpu';
reset(gpuDevice())
%% Read SAR Image
% 0:Nochange, 1:0-1, 2:0-255, 3:Input_Clipped_wrt_Ref_with_noChange, 4:Input_Rescaled_wrt_Ref
RescaleFlag = 1;
DataSetName = "WonsanNorthKoreaSE";% KapikuleTurkeySE,CordobaSpainSE,RichmondUSASE,ShahrudIranSE
ImgRef = ReadDataSet(DataSetName);
ImgIn_Rescaled = RescaleDataSet(ImgRef,ImgRef,RescaleFlag);
RSv = char(DataSetName);
%% Create Train Dataset
DataAvailability = 50; %[100 80 70 50];
NoiseLevel = 0; % 0.03

% Options for Phase domain
MaskingOpt = "box"; % ["box","random","linearTheta","linearRandom"]
PhaseType = "UniformRandom"; % Options for phase Errors                     "UniformRandom","GausRandom","Sine","Poly"
type = "valGaus";% "stdGaus"; % Option for adding noise.                                "stdGauss": based on standard-deviation of clean PH.

% Options for converting to Patches
FullImgFlag = 0;% 0:Convert to Patch, 1:Do not convert to patch
N2XFlag = 0; % 0: N2N, 1:N2C
AugmentationAngles = [0,30,45,60];
AngleConsider = [AugmentationAngles,AugmentationAngles+90,...
        AugmentationAngles+180,AugmentationAngles+270];
PatchSize = 256;
PatchStep = 64;
VerboseFlag = 1;
DisplayFlag = 1;
% ResidualFlag = 0;
% RescaleFlag = DataTrain.opts.RescaleFlag;% 1;
% SubtractMeanFlag = DataTrain.opts.SubtractMeanFlag;% 1;
% Rotationflag =1;

N1 = size(ImgRef,1);
N2 = size(ImgRef,2);
N3 = 2*floor(sqrt((N1).^2* DataAvailability/100)/2);
N4 = 2*floor(sqrt((N2).^2* DataAvailability/100)/2);

opts.N1 = N1;
opts.N2 = N2;

opts.Nlim1 = N3;
opts.Nlim2 = N4;

opts.DAL = DataAvailability/100;
opts.Masking = MaskingOpt;
M = MaskOpt(opts.DAL,N1,N2,opts.Masking);
opts.Mask = M;

opts.RescaleFlag = RescaleFlag;

% opts.SubtractMeanFlag  = SubtractMeanFlag;
% opts.ResidualFlag = ResidualFlag;
opts.DataAvailability = DataAvailability;
% opts.NoiseLevel = NoiseLevel;
opts.NoiseSigma = NoiseLevel;
opts.PhaseType = PhaseType;
opts.type = type;
% opts.PEMincFlag = PEMincFlag;
% opts.PEM_factor = PEM_factor;
% opts.PEM_min = PEM_min;
% opts.MaxIter = maxIt;
% opts.MiniBatchSize = MiniBatchSize;

nH = @(z)H(z,opts); % The forward Fourier sampling operator
nHH=@(z)HH(z,opts); % The backward Fourier sampling operator

% Options for defining of model
NetChoice = NetCoice_i;%[0 1 4 5]+30% [20;30;50;60]
ResidualFlagModel = 1;
ResidualFlag = 0;
%% Convert to Patches
[Patch_Img_In_Train,Patch_ImgOut_Train] = ConvertDataPatch(...
    ImgIn_Rescaled,AngleConsider,N2XFlag,FullImgFlag,...
    PatchSize,PatchStep,nH,nHH,VerboseFlag,opts);
if DisplayFlag == 1
ii = randi(size(Patch_Img_In_Train,4)-12,1);
% for ii = 1:12:size(Patch_Img_In_Total,4)
    figure(1)
    imagesc(imtile(Patch_Img_In_Train(:,:,:,ii:ii+11)));
    colormap gray;colorbar;
    axis image;axis on;
    drawnow;
    figure(2)
    imagesc(imtile(Patch_ImgOut_Train(:,:,:,ii:ii+11)));
    colormap gray;colorbar;
    axis image;axis on;
    drawnow;
    figure(3)
    imagesc(imtile(Patch_ImgOut_Train(:,:,:,ii:ii+11)-Patch_Img_In_Train(:,:,:,ii:ii+11)));
    colormap gray;colorbar;
    axis image;axis on;
    drawnow;
    pause(0.3);
end

switch RescaleFlag % 0:Nochange,1:0-1,2:0-255,3:Output_Clipped_wrt_Input_with_noChange
    case 0
        Rsf = 'NoRescale_';
    case 1
        Rsf = 'Rescale_0_1_';
    case 2
        Rsf = 'Rescale_0_255_';
    case 3
        Rsf = 'Clipped2Input_';
    otherwise
        Rsf = '';
end
RzF = '';
if ResidualFlag==1
    RzF = 'Res_';
end
AugF = '';
if length(AugmentationAngles) > 1
    AugF = 'AngleAug_';
end
SmF = '';
NoiseLevelText = num2str(NoiseLevel);
switch N2XFlag
    case 0
        N2Xtext = 'N2N';
    case 1
        N2Xtext = 'N2C';
    otherwise
        error('N2XFlag = 0,1');
end


%% Define Network
    % 0: Standard DnCNN with 20-Layers
    % 1: % DnCNN with maeRegressionLayer - 20 layers
    % ,Patch_ImgOut_Total
    PatchSizeX = size(Patch_Img_In_Train,1);
    PatchSizeY = size(Patch_Img_In_Train,2);
    NumCh = size(Patch_Img_In_Train,3);
    if aaX == 1
        [layers,NetName,lastLayerName] = DefineModels(PatchSizeX,PatchSizeY,NetChoice,NumCh,ResidualFlagModel);
        analyzeNetwork(layers)
    else
        [~,NetName,~] = DefineModels(PatchSizeX,PatchSizeY,NetChoice,NumCh,ResidualFlagModel);
        FileName = sprintf([NetName,'_',num2str(PatchSizeX),'x',num2str(PatchSizeY),...
        '_',RSv,'_SAR_',N2Xtext,'_NL_',NoiseLevelText,'_DAL_',num2str(DataAvailability),'_',...
        RzF,AugF,Rsf,'Net_',num2str(NetChoice),'_mBz_',num2str(MiniBatchSize),...
        '_mEp_',num2str(MaxEpochs),'.mat']);
        as = load(fullfile(cd,'TrainedNetworks',FileName));
        layers = layerGraph(as.trainedNet);
        try
            NetName = as.NetName;
            lastLayerName = as.lastLayerName;
        catch
            NetName = ['Net_',num2str(NetChoice)];
        end
    end
%% Define validation data
% 0:Nochange, 1:0-1, 2:0-255, 3:Input_Clipped_wrt_Ref_with_noChange, 4:Input_Rescaled_wrt_Ref
DataSetNameValid = "KapikuleTurkeySE";
ImgRefValid = ReadDataSet(DataSetNameValid);
ImgRefValid_Rescaled = RescaleDataSet(ImgRefValid,ImgRefValid,RescaleFlag);
% Validation Options for Forward model
N1_Valid = size(ImgRefValid,1);
N2_Valid = size(ImgRefValid,2);
N3_Valid = 2*floor(sqrt((N1_Valid).^2* DataAvailability/100)/2);
N4_Valid = 2*floor(sqrt((N2_Valid).^2* DataAvailability/100)/2);

opts_Valid = opts;
opts_Valid.N1 = N1_Valid;
opts_Valid.N2 = N2_Valid;

opts_Valid.Nlim1 = N3_Valid;
opts_Valid.Nlim2 = N4_Valid;

opts_Valid.DAL = DataAvailability/100;
opts_Valid.Masking = MaskingOpt;
M_Valid = MaskOpt(opts_Valid.DAL,N1_Valid,N2_Valid,opts_Valid.Masking);
opts_Valid.Mask = M_Valid;

nH_Valid = @(z)H(z,opts_Valid); % The forward Fourier sampling operator
nHH_Valid=@(z)HH(z,opts_Valid); % The backward Fourier sampling operator

[Patch_Img_In_Valid,Patch_ImgOut_Valid] = ConvertDataPatch(...
    ImgRefValid_Rescaled,0,N2XFlag,FullImgFlag,...
    PatchSize,ceil(PatchSize/2),nH_Valid,nHH_Valid,VerboseFlag,opts_Valid);
%% Train network
optsTrain = trainingOptions('adam', ...
		'LearnRateSchedule','piecewise', ...
		'InitialLearnRate',InitialLearnRate,...
		'LearnRateDropFactor',LearnRateDropFactor, ...
		'LearnRateDropPeriod',LearnRateDropPeriod, ...
		'MaxEpochs',MaxEpochs, ...
		'MiniBatchSize',MiniBatchSize, ...
		'Shuffle','every-epoch', ...
		'Plots','training-progress', ...
		'Verbose',true, ...
		'VerboseFrequency',VerboseFrequency,...
		'ExecutionEnvironment', ExecutionEnvironment,...
		'ValidationData',{Patch_Img_In_Valid,Patch_ImgOut_Valid}, ...
		'ValidationFrequency',ValidationFrequency);
[trainedNet,traininfo] = ...
	trainNetwork(Patch_Img_In_Train,Patch_ImgOut_Train,layers,optsTrain);

%% Calculate Performance on Valdation Data
layers = trainedNet;
batchsizeValid = size(Patch_Img_In_Valid,4);
YPred_XValidation = Patch_Img_In_Valid*0;
while 1
    try
        newV = 1:batchsizeValid:size(Patch_Img_In_Valid,4);
        for ll = newV
            if ll ~= newV(end)
                YPred_XValidationBatch = activations(trainedNet,...
                    Patch_Img_In_Valid(:,:,:,ll+(1:batchsizeValid)-1),...
                    lastLayerName,'OutputAs','channels',...
                    'ExecutionEnvironment',ExecutionEnvironment);
                YPred_XValidation(:,:,:,ll+(1:batchsizeValid)-1) = ...
                    YPred_XValidationBatch;
            else
                YPred_XValidationBatch = activations(trainedNet,...
                    Patch_Img_In_Valid(:,:,:,ll:end),...
                    lastLayerName,'OutputAs','channels',...
                    'ExecutionEnvironment',ExecutionEnvironment);
                YPred_XValidation(:,:,:,ll:end) = YPred_XValidationBatch;
            end
        end
        break;
    catch
        batchsizeValid = ceil(batchsizeValid/2);
    end
end
[snrYPred_XV,psnrYPred_XV,ssimYPred_XV] = ...
    SNRcalculate(YPred_XValidation,Patch_ImgOut_Valid);

[snrXValidation,psnrXValidation,ssimXValidation] = ...
    SNRcalculate(Patch_Img_In_Valid,Patch_ImgOut_Valid);

if DisplayFlag == 1
	ii = randi(size(YPred_XValidation,4)-12,1);
	figure(4)
	imagesc(imtile(Patch_ImgOut_Valid(:,:,:,ii:ii+11)));
	colormap gray;colorbar;
	axis image;axis on;
	drawnow;
	figure(5)
	imagesc(imtile(Patch_Img_In_Valid(:,:,:,ii:ii+11)));
	colormap gray;colorbar;
	axis image;axis on;
	drawnow;
	figure(6)
	imagesc(imtile(Patch_Img_In_Valid(:,:,:,ii:ii+11)-Patch_ImgOut_Valid(:,:,:,ii:ii+11)));
	colormap gray;colorbar;
	axis image;axis on;
	drawnow;
end

DataPerf = [snrYPred_XV,psnrYPred_XV,ssimYPred_XV,snrXValidation,psnrXValidation,ssimXValidation];
disp(mean(DataPerf,1))
%% Save Network
FileName = sprintf([NetName,'_',num2str(PatchSizeX),'x',num2str(PatchSizeY),...
        '_',RSv,'_SAR_',N2Xtext,'_NL_',NoiseLevelText,'_DAL_',num2str(DataAvailability),'_',...
        RzF,AugF,Rsf,'Net_',num2str(NetChoice),'_mBz_',num2str(MiniBatchSize),...
        '_mEp_',num2str(MaxEpochs),'.mat']);
nCount = aaX;
save(fullfile(cd,'TrainedNetworks',FileName),...
    'trainedNet','traininfo','MaxEpochs','opts','optsTrain','NetName',...
    'MiniBatchSize','NetChoice','ResidualFlag','DataPerf','DataSetNameValid',...
    'RescaleFlag','VerboseFlag','nCount','ResidualFlagModel',...
    'DataSetName','N2XFlag','lastLayerName','-v7.3')
disp(datetime('now'))
disp([num2str(aaX),': ',FileName,' has been written'])
clear trainedNet traininfo MaxEpochs opts  NetName ...
    MiniBatchSize NetChoice ResidualFlag  DataPerf ...
    RescaleFlag  layers opts AugmentationAngles lastLayerName ...
    Patch_Img_In_Valid Patch_ImgOut_Valid Patch_Img_In_Train Patch_ImgOut_Train
end
end

% ------------------------------------------------------------------------
%% 			HELPING FUNCTIONS

% ------------------------------------------------------------------------
%% Read Dataset as one full image
function ImgRef = ReadDataSet(DataSetName)
switch DataSetName
    case "WonsanNorthKoreaSE"
        ImgRef = imread(fullfile(cd,'SAR_Datasets','SAR_Images_Straight','WonsanNorthKoreaSE.jpg'));
    case "KapikuleTurkeySE"
        ImgRef = imread(fullfile(cd,'SAR_Datasets','SAR_Images_Straight','KapikuleTurkeySE.jpg'));
    case "CordobaSpainSE"
        ImgRef = imread(fullfile(cd,'SAR_Datasets','SAR_Images_Straight','CordobaSpainSE.jpg'));
    case "RichmondUSASE"
        ImgRef = imread(fullfile(cd,'SAR_Datasets','SAR_Images_Straight','RichmondUSASE.jpg'));
    case "ShahrudIranSE"
        ImgRef = imread(fullfile(cd,'SAR_Datasets','SAR_Images_Straight','ShahrudIranSE.jpg'));
    otherwise
        error('The dataset not avaialble')
end
% Convert from RGB to gray
if size(ImgRef,3)==3
    ImgRef = rgb2gray(ImgRef);
elseif size(ImgRef,3)==1
    disp('the image is already in gray scale')
else
    error(['The image has ',num2str(size(ImgRef,3)),' channels'])
end
ImgRef = single(ImgRef);
[N1,N2]=size(ImgRef);
N1 = 2*floor(sqrt((N1).^2)/2);
N2 = 2*floor(sqrt((N2).^2)/2);
ImgRef = imresize(ImgRef,[N1,N2]);
end

% ------------------------------------------------------------------------
%% Rescale Dataset based on Input
function ImgIn_Rescaledc = RescaleDataSet(ImgIn,ImgRef,RescaleFlag)
% 0:Nochange, 1:0-1, 2:0-255, 3:Input_Clipped_wrt_Ref_with_noChange, 4:Input_Rescaled_wrt_Ref
ImgIn_Rescaledc = ImgIn;
for ii = 1:size(ImgIn,4)
    switch RescaleFlag
        case 0
            ImgIn_Rescaledc(:,:,:,ii) = ImgIn(:,:,:,ii);
        case 1
            ImgIn_Rescaledc(:,:,:,ii) = rescale(ImgIn(:,:,:,ii));
        case 2
            ImgIn_Rescaledc(:,:,:,ii) = rescale(ImgIn(:,:,:,ii))*255;
        case 3
            maxImgRef = max(ImgRef(:,:,:,ii));
            minImgRef = min(ImgRef(:,:,:,ii));
            A = ImgRef(:,:,:,ii);
            A(A<minImgRef) = minImgRef;
            A(A>maxImgRef) = maxImgRef;
            ImgIn_Rescaledc(:,:,:,ii) = A;
        case 4
            maxImgRef = max(ImgRef(:,:,:,ii));
            minImgRef = min(ImgRef(:,:,:,ii));
            maxImgIn = max(ImgIn(:,:,:,ii));
            minImgIn = min(ImgIn(:,:,:,ii));
            ImgIn_Rescaledc(:,:,:,ii) = (ImgIn(:,:,:,ii)-minImgIn)/...
                (maxImgIn-minImgIn)*(maxImgRef-minImgRef)+minImgRef;
        otherwise
            error('Rescale 0,1,2,3,4');
    end
end
end

% ------------------------------------------------------------------------
%% Forward Model
function [ImgConv,ImgFRef,g] = ForwardModelPE(ImgRef,nH,nHH,opts)
ImgFRef_Rescale = RescaleDataSet(ImgRef,ImgRef,opts.RescaleFlag);
ImgFRef = addphase(ImgFRef_Rescale);                                       % Adding complex valued uniform Random phase to magnitude of RF
g = nH(ImgFRef);                                                           % Clean and full DAL Phase histories (PH)
optsNoise.type = opts.type;                                                % Noise distribution type
optsNoise.NoiseSigma = opts.NoiseSigma;                                    % Setting NL to option
optsNoise.PhaseType = opts.PhaseType;                                      % "UniformRandom","GausRandom","Sine","Poly"
gn = AddNoise(g,optsNoise);                                                % Add Noise to PH
f_conv = reshape(nHH(gn),opts.N1,opts.N2);                                 % Conventional Reconstruction (CR)
f_conv_abs = abs(f_conv);
f_conv_ang = atan2(imag(f_conv),real(f_conv));
f_conv_abs_resc = RescaleDataSet(f_conv_abs,ImgFRef_Rescale,opts.RescaleFlag);
ImgConv = f_conv_abs_resc.*exp(1i*f_conv_ang);
end

% ------------------------------------------------------------------------
%% Read data from SAR image and convert to patches to make dataset with
% Forwared model applied to patches
function [Patch_Img_In_Total,Patch_ImgOut_Total] = ConvertDataPatch(...
    ImgRef,AngleConsider,N2XFlag,FullImgFlag,...
    PatchSize,PatchStep,nH,nHH,VerboseFlag,opts)

% Use forward model to creaate images with speckle
Patch_Img_In_Total=zeros(PatchSize,PatchSize,1,1e5,'single');
Patch_ImgOut_Total=zeros(PatchSize,PatchSize,1,1e5,'single');
kk = 0;
for Angle_Img = AngleConsider
    % Angle_Img = 30;% AngleConsider
    % optDataSet.RescaleFlag = opts.RescaleFlag;
    % optDataSet.L = L;% Number of Looks
    
    [ImgConv_1,ImgFRef_1,g_1] = ForwardModelPE(ImgRef,nH,nHH,opts);
    [ImgConv_2,ImgFRef_2,g_2] = ForwardModelPE(ImgRef,nH,nHH,opts);

    ImgFRefAbs_1 = abs(ImgFRef_1);
    ImgFRefAbs_2 = abs(ImgFRef_2);
    
    ImgConvAbs_1 = abs(ImgConv_1);
    ImgConvAbs_2 = abs(ImgConv_2);

    % ImgConvAbs_1 = RescaleDataSet(abs(ImgConv_1),ImgFRefAbs_1,opts.RescaleFlag);
    % ImgConvAbs_2 = RescaleDataSet(abs(ImgConv_2),ImgFRefAbs_2,opts.RescaleFlag);

    switch N2XFlag
        case 0
            Img_In = ImgConvAbs_1;
            ImgOut = ImgConvAbs_2;
        case 1
            Img_In = ImgConvAbs_1;
            ImgOut = ImgFRefAbs_1;
        otherwise
            error('N2X Falg: Either 0-1')
    end

    switch FullImgFlag
        case 0
            BlkSize = [PatchSize,PatchSize];
            optRotPatch.BlkStepRow = PatchStep;
            optRotPatch.BlkStepCol = PatchStep;
            
            Patch_Img_In = ...
                rotate2patch(Img_In,BlkSize,Angle_Img,optRotPatch);
            Patch_Img_In_resh = ...
                reshape(Patch_Img_In,[BlkSize,1,size(Patch_Img_In,2)]);

            Patch_ImgOut = ...
                rotate2patch(ImgOut,BlkSize,Angle_Img,optRotPatch);
            Patch_ImgOut_resh = ...
                reshape(Patch_ImgOut,[BlkSize,1,size(Patch_ImgOut,2)]);
            

            Patch_Img_In_Total(1:size(Patch_Img_In_resh,1),...
                1:size(Patch_Img_In_resh,2),1:size(Patch_Img_In_resh,3),...
                kk+(1:size(Patch_Img_In_resh,4))) = Patch_Img_In_resh;

            Patch_ImgOut_Total(1:size(Patch_ImgOut_resh,1),...
                1:size(Patch_ImgOut_resh,2),1:size(Patch_ImgOut_resh,3),...
                kk+(1:size(Patch_ImgOut_resh,4))) = Patch_ImgOut_resh;

            kk = kk + size(Patch_ImgOut_resh,4);
            if VerboseFlag == 1
                disp([size(Patch_ImgOut_resh,[1,2,3]),kk]);
            end
        case 1
            Patch_Img_In_Total = rotate2patch(Img_In,[],Angle_Img,[]);
            Patch_ImgOut_Total = rotate2patch(ImgOut,[],Angle_Img,[]);
        otherwise
            error('FullImgFlag - 0,1')
    end

end
if kk>0
    Patch_Img_In_Total = Patch_Img_In_Total(:,:,:,1:kk);
    Patch_ImgOut_Total = Patch_ImgOut_Total(:,:,:,1:kk);
end
end

% ------------------------------------------------------------------------
% Add uniform random phase to reflectivity field
function Xc = addphase(X)
% adds random uncorrelated uniform [-pi,pi] phase to matrix x
% [M,N]=size(x);
% if class(x)== 'uint8'
% diag(phase) would be a matrix such that xc=diag(phase)*x;
Xc = double(X);
for ii = 1:size(X,4)
    for kk = 1:size(X,3)
        x = X(:,:,kk,ii);
        phi=rand(size(x),'double');
        phi=pi*(2*phi-1);
        phase=exp(sqrt(-1)*phi);
        xc = double(x).*phase;
        Xc(:,:,kk,ii) = xc;
    end
end
end
% ------------------------------------------------------------------------
%% Forward Model
function y=H(x,parameters)
%% y = H(x,parameters)
% It band limit the fucntion according to the paramters, the band limit is
% defined though Nlim, the actual interpretation is its the low pass filter
% which is applied by cutting the high frequency compenent when FFT is
% taken. By cutting means that its removed from original fft fucntion but
% not set to zero.
% The output y is smaller in size as compared to original fucntion and is
% in frequency domain.

N1 = parameters.N1;
N2 = parameters.N2;
N3 = parameters.Nlim1;
N4 = parameters.Nlim2;

M = parameters.Mask;
x=reshape(x,N1,N2);
y=M.*fft2(x,N1,N2);
end
% ------------------------------------------------------------------------
%% Conventional Reconstruction Model
function x=HH(y,parameters)
%% x=HH(y,parameters)
% The fucntion reconverts the fft image done by H fucntion and keeping 
% the band limiting operation in consideration, the sie is again back to
% the same of original x variable but the aim was to band limit it. Now the
% frequency comonents are replace d 

%no normalization i.e. produces x=F'*y, where F is DFT.
N1 = parameters.N1;
N2 = parameters.N2;
N3 = parameters.Nlim1;
N4 = parameters.Nlim2;
y=reshape(y,N1,N2);
M = parameters.Mask;
x = ifft2(y.*M);
end

% ------------------------------------------------------------------------
%% Masking option for non avaialbility of reflection
function M = MaskOpt(DAL,N1,N2,Mopt)
% Finds the mask for SAR model H(x) = M.*FFT2(X)
M = zeros(N1,N2);
switch Mopt
    case "random"
        M = rand([N1,N2]);
        DALorig = DAL;
        Morig = M;
        while 1
        Mval = 1-DAL;
        M(M>=Mval) = 1;
        M(M<Mval) = 0;
        a = length(find(M(:)==1));
        b = length(find(M(:)==0));
        eta = a/(a+b);
        if eta-DALorig<-1e-2
            DAL = DAL + 1e-3;
            M = Morig;
        elseif eta-DALorig>1e-2
            DAL = DAL - 1e-3;
            M = Morig;
        else
            break;
        end
        end
    case "linearTheta"
        M = repmat(rand([N1,1]),1,N2);
        DALorig = DAL;
        Morig = M;
        while 1
        Mval = 1-DAL;
        M(M>=Mval) = 1;
        M(M<Mval) = 0;
        a = length(find(M(:)==1));
        b = length(find(M(:)==0));
        eta = a/(a+b);
        if eta-DALorig<-1e-3
            DAL = DAL + 1e-3;
            M = Morig;
        elseif eta-DALorig>1e-3
            DAL = DAL - 1e-3;
            M = Morig;
        else
            break;
        end
        end
    case "linearRandom"
        M = repmat(rand([N1,1]),1,N2);
        p = ceil(sqrt(sqrt(DAL))*N2);
        q = randi(N2-p+1,[N1,1]);
        for i = 1:N1 
            M(i,1:q(i)) = 0;
            M(i,q(i)+p+1:N1) = 0;
        end
        DALorig = DAL;
        Morig = M;
        while 1
        Mval = 1-DAL;
        M(M>=Mval) = 1;
        M(M<Mval) = 0;
        a = length(find(M(:)==1));
        b = length(find(M(:)==0));
        eta = a/(a+b);
        if eta-DALorig<-1e-2
            DAL = DAL + 1e-3;
            M = Morig;
        elseif eta-DALorig>1e-2
            DAL = DAL - 1e-3;
            M = Morig;
        else
            break;
        end
        end
    case "linearFreq"
        M = repmat(rand([1,N2]),N1,1);
        DALorig = DAL;
        Morig = M;
        while 1
        Mval = 1-DAL;
        M(M>=Mval) = 1;
        M(M<Mval) = 0;
        a = length(find(M(:)==1));
        b = length(find(M(:)==0));
        eta = a/(a+b);
        if eta-DALorig<-1e-2
            DAL = DAL + 1e-3;
            M = Morig;
        elseif eta-DALorig>1e-2
            DAL = DAL - 1e-3;
            M = Morig;
        else
            break;
        end
        end
    case "box"
        DALorig = DAL;
        if DAL == 1
            M = ones(N1,N2);
        else
            while 1
                M = zeros(N1,N2);
                N3 = ceil(sqrt(DAL)*N1/2)*2;
                N4 = ceil(sqrt(DAL)*N2/2)*2;
                M(1:N3/2,1:N4/2) = 1;
                M(1:N3/2,N2-N4/2+1:N2) = 1;
                M(N1-N3/2+1:N1,1:N4/2) = 1;
                M(N1-N3/2+1:N1,N2-N4/2+1:N2) = 1;

                a = length(find(M(:)==1));
                b = length(find(M(:)==0));
                eta = a/(a+b);
                if eta-DALorig<-1e-2
                    DAL = DAL + 1e-3;
                elseif eta-DALorig>1e-2
                    DAL = DAL - 1e-3;
                else
                    break;
                end
            end
        end
    case "LinearThetaBoxFreq"
        if DAL == 1
            M = ones(N1,N2);
        else
            N3 = ceil(sqrt(DAL)*N1/2)*2;
            N4 = ceil(sqrt(DAL)*N2/2)*2;
            M(:,1:N4/2) = 1;
            M(:,N2-N4/2+1:N2) = 1;
            M(:,1:N4/2) = 1;
            M(:,N2-N4/2+1:N2) = 1;
            Mp = repmat(rand([N1,1]),1,N2);
            M = M.*Mp;
        end
    otherwise
        error('Mask option not listed');
end
end

% ------------------------------------------------------------------------
%% Add noise to measured phase histories
function gn = AddNoise(g,optsNoise)
switch optsNoise.type
    case "stdGaus"
        g_std = std(abs(g(:)));
        Noise = randn(size(g))+1i*randn(size(g));
        gn = g + Noise * g_std * optsNoise.NoiseSigma;
    case "valGaus"
        Noise = randn(size(g))+1i*randn(size(g));
        gn = g + Noise * optsNoise.NoiseSigma;
    otherwise
        error('wrong noise option')
end
end

% ------------------------------------------------------------------------
%% Calculate SNR
function [snrXlog,psnrXlog,ssimX] = SNRcalculate(Xin,Xref)
% Calculate SNR and PSNR
snrXlog = zeros(size(Xin,4),1);
psnrXlog = zeros(size(Xin,4),1);
ssimX = zeros(size(Xin,4),1);
for ii = 1:size(Xin,4)
    Xref_ii = Xref(:,:,:,ii);
    Xref_ii_M = (Xref_ii-min(Xref_ii,[],[1,2]))./(max(Xref_ii,[],[1,2])-min(Xref_ii,[],[1,2]));

    Xin_ii = Xin(:,:,:,ii);
    A = [Xin_ii(:),ones(size(Xin_ii(:)),'like',Xin_ii)]\Xref_ii_M(:);
    Xin_ii_M = Xin_ii*A(1)+A(2);
    IdxMin = find(Xin_ii_M<min(Xref_ii_M(:)));
    if ~isempty(IdxMin)
        Xin_ii_M(IdxMin) = min(Xref_ii_M(:));
    end
    IdxMax = find(Xin_ii_M>max(Xref_ii_M(:)));
    if ~isempty(IdxMax)
        Xin_ii_M(IdxMax) = max(Xref_ii_M(:));
    end
    sseX = ((Xin_ii_M(:)-Xref_ii_M(:))).^2;
    sseX(sseX<=1e-10) = 1e-10;
    snrX = mean(((Xref_ii_M(:))).^2)./mean(sseX);
    psnrX = mean((max(Xref_ii_M(:))).^2)./mean(sseX);
    snrXlog(ii) = 10*log10((snrX));
    psnrXlog(ii) = 10*log10((psnrX));
    [ssimX_global, ssimX_local] = ssim(Xin_ii_M(:),Xref_ii_M(:));
    ssimX(ii) = mean(ssimX_local);
end
end

% ------------------------------------------------------------------------
%% Define all models
function [layers ,NetName,lastLayerName] = DefineModels(PatchSizeX,PatchSizeY,NetChoice,NumCh,ResFlag)
switch NetChoice
    case 20
        layers = DnCNN_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag);
        NetName = 'DnCNN_Net';
        lastLayerName = layers.Layers(end-1).Name;
    case 21
        layers = DnCNN_MAE_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag);
        NetName = 'DnCNN_MAE_Net';
        lastLayerName = layers.Layers(end-1).Name;
    case 24
        layers = DnCNN_DCT_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag);
        NetName = 'DnCNN_DCT_Net';
        lastLayerName = layers.Layers(end-1).Name;
    case 25
        layers = DnCNN_DCT_LOG_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag);
        NetName = 'DnCNN_DCT_LOG_Net';
        lastLayerName = layers.Layers(end-1).Name;
    case 30
        layers = AlternateFiltersWithSkip_Net(PatchSizeX,PatchSizeY,NumCh,16,ResFlag);
        NetName = 'AlternateFiltersWithSkip_Net';
        lastLayerName = layers.Layers(end-2).Name;
    case 31
        layers = AlternateFiltersWithSkip_MAE_Net(PatchSizeX,PatchSizeY,NumCh,16,ResFlag);
        NetName = 'AlternateFiltersWithSkip_MAE_Net';
        lastLayerName = layers.Layers(end-2).Name;
    case 34
        layers = AlternateFiltersWithSkip_DCT_Net(PatchSizeX,PatchSizeY,NumCh,16,ResFlag);
        NetName = 'AlternateFiltersWithSkip_DCT_Net';
        lastLayerName = layers.Layers(end-2).Name;
    case 35
        layers = AlternateFiltersWithSkip_DCT_LOG_Net(PatchSizeX,PatchSizeY,NumCh,16,ResFlag);
        NetName = 'AlternateFiltersWithSkip_DCT_LOG_Net';
        lastLayerName = layers.Layers(end-2).Name;
    case 50
        layers = MDRU_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag);
        NetName = 'MRDU_Net';
        lastLayerName = layers.Layers(end-1).Name;
    case 51
        layers = MDRU_MAE_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag);
        NetName = 'MDRU_MAE_Net';
        lastLayerName = layers.Layers(end-1).Name;
    case 54
        layers = MDRU_DCT_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag);
        NetName = 'MDRUnet_DCT_Net';
        lastLayerName = layers.Layers(end-1).Name;
    case 55
        layers = MDRU_DCT_LOG_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag);
        NetName = 'MDRUnet_DCT_LOG_Net';
        lastLayerName = layers.Layers(end-1).Name;
    case 60
        layers = MRDDA_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag);
        NetName = 'MRDDA_Net';
        lastLayerName = layers.Layers(end-1).Name;
    case 61
        layers = MRDDA_MAE_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag);
        NetName = 'MRDDA_MAE_Net';
        lastLayerName = layers.Layers(end-1).Name;
    case 64
        layers = MRDDA_DCT_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag);
        NetName = 'MRDDA_DCT_Net';
        lastLayerName = layers.Layers(end-1).Name;
    case 65
        layers = MRDDA_DCT_LOG_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag);
        NetName = 'MRDDA_DCT_LOG_Net';
        lastLayerName = layers.Layers(end-1).Name;
    otherwise
        error('NetChoice does not exist')
end
disp(['Net Name = ',NetName])
end

% ------------------------------------------------------------------------
%% Define DnCNN_Net
function lGraph_layer = DnCNN_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag)
disp('-----------------DnCNN_Net with L2 Regression-----------------')
% DnCNN with LOG DCT L2 Regression - 20 layers
% PatchSizeX=512;PatchSizeY=512;NumCh=1;ResFlag=1;
disp('DnCNN with L2 Regression - 20 layers')
net = denoisingNetwork('DnCNN');
layers = net.Layers;
layers(1,1) = imageInputLayer([PatchSizeX,PatchSizeY,NumCh],...
    'Name','InputLayer',...
    'Normalization','none');
layers(2,1) = convolution2dLayer(layers(2,1).FilterSize, ...
    layers(2,1).NumFilters,...
    'Padding', layers(2,1).PaddingSize,...
    'Name', layers(2,1).Name,...
    'WeightsInitializer','narrow-normal');
layers(end-1,1) = convolution2dLayer(...
    layers(end-1,1).FilterSize, NumCh,...
    'Padding', layers(end-1,1).PaddingSize,...
    'Name', layers(end-1,1).Name,...
    'WeightsInitializer','narrow-normal');
if ResFlag
    addition_residual_layer = additionLayer(2,"Name",'AdditionResdiual');
    layer_end_name = layers(end,1).Name;
    layers = layers(1:end-1,1);
    layers = [layers;addition_residual_layer;regressionLayer('Name',layer_end_name)];
else
    layers(end,1) = regressionLayer('Name',layers(end,1).Name);
end
lGraph_layer = layerGraph(layers);
if ResFlag
    lGraph_layer = connectLayers(lGraph_layer,layers(1,1).Name,...
    [addition_residual_layer.Name,'/',addition_residual_layer.InputNames{2}]);
end
end

% ------------------------------------------------------------------------
%% Define DnCNN_MAE_Net
function lGraph_layer = DnCNN_MAE_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag)
disp('-----------------DnCNN_MAE_Net with L1 Regression-----------------')
% DnCNN with LOG DCT L2 Regression - 20 layers
% PatchSizeX=512;PatchSizeY=512;NumCh=1;ResFlag=1;
disp('DnCNN with L1 Regression - 20 layers')
net = denoisingNetwork('DnCNN');
layers = net.Layers;
layers(1,1) = imageInputLayer([PatchSizeX,PatchSizeY,NumCh],...
    'Name','InputLayer',...
    'Normalization','none');
layers(2,1) = convolution2dLayer(layers(2,1).FilterSize, ...
    layers(2,1).NumFilters,...
    'Padding', layers(2,1).PaddingSize,...
    'Name', layers(2,1).Name,...
    'WeightsInitializer','narrow-normal');
layers(end-1,1) = convolution2dLayer(...
    layers(end-1,1).FilterSize, NumCh,...
    'Padding', layers(end-1,1).PaddingSize,...
    'Name', layers(end-1,1).Name,...
    'WeightsInitializer','narrow-normal');
if ResFlag
    addition_residual_layer = additionLayer(2,"Name",'AdditionResdiual');
    layer_end_name = layers(end,1).Name;
    layers = layers(1:end-1,1);
    layers = [layers;addition_residual_layer;maeRegressionLayer(layer_end_name)];
else
    layers(end,1) = maeRegressionLayer(layers(end,1).Name);
end
lGraph_layer = layerGraph(layers);
if ResFlag
    lGraph_layer = connectLayers(lGraph_layer,layers(1,1).Name,...
    [addition_residual_layer.Name,'/',addition_residual_layer.InputNames{2}]);
end
end

% ------------------------------------------------------------------------
%% Define DnCNN_DCT_Net
function lGraph_layer = DnCNN_DCT_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag)
disp('-----------------DnCNN_DCT_Net with DCT-L2 Regression-----------------')
% DnCNN with DCT L2 Regression - 20 layers
% PatchSizeX=512;PatchSizeY=512;NumCh=1;ResFlag=1;
disp('DnCNN with DCT L2 Regression - 20 layers')
net = denoisingNetwork('DnCNN');
layers = net.Layers;
layers(1,1) = imageInputLayer([PatchSizeX,PatchSizeY,NumCh],...
    'Name','InputLayer',...
    'Normalization','none');
layers(2,1) = convolution2dLayer(layers(2,1).FilterSize, ...
    layers(2,1).NumFilters,...
    'Padding', layers(2,1).PaddingSize,...
    'Name', layers(2,1).Name,...
    'WeightsInitializer','narrow-normal');
layers(end-1,1) = convolution2dLayer(...
    layers(end-1,1).FilterSize, NumCh,...
    'Padding', layers(end-1,1).PaddingSize,...
    'Name', layers(end-1,1).Name,...
    'WeightsInitializer','narrow-normal');
if ResFlag
    addition_residual_layer = additionLayer(2,"Name",'AdditionResdiual');
    layer_end_name = layers(end,1).Name;
    layers = layers(1:end-1,1);
    layers = [layers;addition_residual_layer;DCTRegressionLayer(layer_end_name)];
else
    layers(end,1) = DCTRegressionLayer(layers(end,1).Name);
end
lGraph_layer = layerGraph(layers);
if ResFlag
    lGraph_layer = connectLayers(lGraph_layer,layers(1,1).Name,...
    [addition_residual_layer.Name,'/',addition_residual_layer.InputNames{2}]);
end
end

% ------------------------------------------------------------------------
%% Define DnCNN_DCT_LOG_Net
function lGraph_layer = DnCNN_DCT_LOG_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag)
disp('-----------------DnCNN_DCT_LOG_Net with LOG-DCT-L2 Regression-----------------')
% DnCNN with LOG DCT L2 Regression - 20 layers
% PatchSizeX=512;PatchSizeY=512;NumCh=1;ResFlag=1;
disp('DnCNN with DCT L2 Regression - 20 layers')
net = denoisingNetwork('DnCNN');
layers = net.Layers;
layers(1,1) = imageInputLayer([PatchSizeX,PatchSizeY,NumCh],...
    'Name','InputLayer',...
    'Normalization','none');
layers(2,1) = convolution2dLayer(layers(2,1).FilterSize, ...
    layers(2,1).NumFilters,...
    'Padding', layers(2,1).PaddingSize,...
    'Name', layers(2,1).Name,...
    'WeightsInitializer','narrow-normal');
layers(end-1,1) = convolution2dLayer(...
    layers(end-1,1).FilterSize, NumCh,...
    'Padding', layers(end-1,1).PaddingSize,...
    'Name', layers(end-1,1).Name,...
    'WeightsInitializer','narrow-normal');
if ResFlag
    addition_residual_layer = additionLayer(2,"Name",'AdditionResdiual');
    layer_end_name = layers(end,1).Name;
    layers = layers(1:end-1,1);
    layers = [layers;addition_residual_layer;DCT_LOG_RegressionLayer(layer_end_name)];
else
    layers(end,1) = DCT_LOG_RegressionLayer(layers(end,1).Name);
end
lGraph_layer = layerGraph(layers);
if ResFlag
    lGraph_layer = connectLayers(lGraph_layer,layers(1,1).Name,...
    [addition_residual_layer.Name,'/',addition_residual_layer.InputNames{2}]);
end
% analyzeNetwork(lGraph_layer)
end

% ------------------------------------------------------------------------
function lgraph = MDRU_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag)
% PatchSizeX = 170;PatchSizeY = PatchSizeX;NumCh = 1;ResFlag=1;
disp('----------------- MDRU_Net with L2 Regression-----------------')
%% Input Layer
InputLayer = imageInputLayer([PatchSizeX,PatchSizeY,NumCh],...
    'Name','InputLayer','Normalization','none');

outputSize = ceil([PatchSizeX,PatchSizeY]/32)*32;
layer_upsampling_input = resize2dLayer("OutputSize",outputSize,'Name','UpscaleInput','Method','bilinear');
layer_downsampling_input = resize2dLayer("OutputSize",[PatchSizeX,PatchSizeY],'Name','DownscaleInput','Method','bilinear');
if ResFlag
    addition_residual_layer = additionLayer(2,"Name",'AdditionResdiual');
end
%=================== MaxPool-Layer ===================%
p = maxPooling2dLayer(2,'Stride',2);
%=================== Upsampling-Layer ===================%
U = resize2dLayer('Scale',[2 2],'method','bilinear');
%=================== Block-Conv ===================%
LayerNumber = 1;
FilterSize = 3; NumFilt = 64;
bnFlag = 0; reluFlag = 1; SuffixLayer = '_1';
L_1 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,SuffixLayer,bnFlag, reluFlag );
SuffixLayer = '_2';
L_2 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,SuffixLayer,bnFlag, reluFlag );
p1 = p;
p1.Name = 'p1';
lgraph = layerGraph([InputLayer;layer_upsampling_input;L_1;L_2;p1]);
%=================== Block-1 ===================%
LayerNameIn = p1.Name;
Type = 5;Siffix = 'M1';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
p2 = p;
p2.Name = 'p2';
lgraph = addLayers(lgraph,p2);
lgraph = connectLayers(lgraph,LayerNameOut,p2.Name);
%=================== Block-2 ===================%
LayerNameIn = p2.Name;
Type = 4;Siffix = 'M2';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
p3 = p;
p3.Name = 'p3';
lgraph = addLayers(lgraph,p3);
lgraph = connectLayers(lgraph,LayerNameOut,p3.Name);
%=================== Block-3 ===================%
LayerNameIn = p3.Name;
Type = 3;Siffix = 'M3';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
p4 = p;
p4.Name = 'p4';
lgraph = addLayers(lgraph,p4);
lgraph = connectLayers(lgraph,LayerNameOut,p4.Name);
%=================== Block-4 ===================%
LayerNameIn = p4.Name;
Type = 2;Siffix = 'M4';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
p5 = p;
p5.Name = 'p5';
lgraph = addLayers(lgraph,p5);
lgraph = connectLayers(lgraph,LayerNameOut,p5.Name);
%=================== Block-5 ===================%
LayerNameIn = p5.Name;
Type = 1;Siffix = 'M5';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
%=================== Block-6 ===================%
U1 = U;
U1.Name = 'U1';
Siffix = 'C1';
DepthConcatLayer = depthConcatenationLayer(2,'Name',['depthCat_',Siffix]);
lgraph = addLayers(lgraph,[U1;DepthConcatLayer]);
lgraph = connectLayers(lgraph,LayerNameOut,U1.Name);
[lgraph,LayerNameOut] = SRD_S(lgraph,p4.Name,1,'S5');
lgraph = connectLayers(lgraph,LayerNameOut,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{2}]);
LayerNameIn = [DepthConcatLayer.Name,'/',DepthConcatLayer.OutputNames{1}];
Type = 2;Siffix = 'M6';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
%=================== Block-7 ===================%
U2 = U;
U2.Name = 'U2';
Siffix = 'C2';
DepthConcatLayer = depthConcatenationLayer(2,'Name',['depthCat_',Siffix]);
lgraph = addLayers(lgraph,[U2;DepthConcatLayer]);
lgraph = connectLayers(lgraph,LayerNameOut,U2.Name);
[lgraph,LayerNameOut] = SRD_S(lgraph,p3.Name,2,'S4');
lgraph = connectLayers(lgraph,LayerNameOut,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{2}]);
LayerNameIn = [DepthConcatLayer.Name,'/',DepthConcatLayer.OutputNames{1}];
Type = 3;Siffix = 'M7';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
%=================== Block-8 ===================%
U3 = U;
U3.Name = 'U3';
Siffix = 'C3';
DepthConcatLayer = depthConcatenationLayer(2,'Name',['depthCat_',Siffix]);
lgraph = addLayers(lgraph,[U3;DepthConcatLayer]);
lgraph = connectLayers(lgraph,LayerNameOut,U3.Name);
[lgraph,LayerNameOut] = SRD_S(lgraph,p2.Name,3,'S3');
lgraph = connectLayers(lgraph,LayerNameOut,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{2}]);
LayerNameIn = [DepthConcatLayer.Name,'/',DepthConcatLayer.OutputNames{1}];
Type = 4;Siffix = 'M8';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
%=================== Block-9 ===================%
U4 = U;
U4.Name = 'U4';
Siffix = 'C4';
DepthConcatLayer = depthConcatenationLayer(2,'Name',['depthCat_',Siffix]);
lgraph = addLayers(lgraph,[U4;DepthConcatLayer]);
lgraph = connectLayers(lgraph,LayerNameOut,U4.Name);
[lgraph,LayerNameOut] = SRD_S(lgraph,p1.Name,4,'S2');
lgraph = connectLayers(lgraph,LayerNameOut,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{2}]);
LayerNameIn = [DepthConcatLayer.Name,'/',DepthConcatLayer.OutputNames{1}];
Type = 5;Siffix = 'M9';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
%=================== Block-Concat ===================%
U5 = U;
U5.Name = 'U5';
Siffix = 'C5';
DepthConcatLayer = depthConcatenationLayer(2,'Name',['depthCat_',Siffix]);
lgraph = addLayers(lgraph,[U5;DepthConcatLayer]);
lgraph = connectLayers(lgraph,LayerNameOut,U5.Name);
[lgraph,LayerNameOut] = SRD_S(lgraph,layer_upsampling_input.Name,5,'S1');
lgraph = connectLayers(lgraph,LayerNameOut,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{2}]);
%=================== Block-Conv ===================%
LayerNumber = 2;
FilterSize = 3; NumFilt = 64;
bnFlag = 0; reluFlag = 1; SuffixLayer = '_1';
C_1 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,...
    SuffixLayer,bnFlag, reluFlag );
SuffixLayer = '_2';NumFilt = 32;
C_2 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,...
    SuffixLayer,bnFlag, reluFlag );
reluFlag = 0; SuffixLayer = '_3';NumFilt = NumCh;
C_3 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,...
    SuffixLayer,bnFlag, reluFlag );
lgraph = addLayers(lgraph,[C_1;C_2;C_3]);
lgraph = connectLayers(lgraph,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.OutputNames{1}],C_1(1).Name);
% ====================== Reg-Layer ==================== %
RegLayer = regressionLayer('Name','routput');
% RegLayer = maeRegressionLayer('routput');
% RegLayer = DCTRegressionLayer('routput');
% RegLayer = DCT_LOG_RegressionLayer('routput');
% =================== Connect Layers ===================%
LayerName = [C_3(end,1).Name,'/',C_3(end,1).OutputNames{1}];
if ResFlag
    lgraph = addLayers(lgraph,[layer_downsampling_input;addition_residual_layer;RegLayer]);
    lgraph = connectLayers(lgraph,InputLayer.Name,...
    [addition_residual_layer.Name,'/',addition_residual_layer.InputNames{2}]);
else
    lgraph = addLayers(lgraph,[layer_downsampling_input;RegLayer]);
end
lgraph = connectLayers(lgraph,LayerName,[layer_downsampling_input.Name]);
% analyzeNetwork(lgraph)
end

% ------------------------------------------------------------------------
function lgraph = MDRU_MAE_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag)
% PatchSizeX = 170;PatchSizeY = PatchSizeX;NumCh = 1;ResFlag=1;
disp('----------------- MDRU_MAE_Net with L1 Regression-----------------')
%% Input Layer
InputLayer = imageInputLayer([PatchSizeX,PatchSizeY,NumCh],...
    'Name','InputLayer','Normalization','none');

outputSize = ceil([PatchSizeX,PatchSizeY]/32)*32;
layer_upsampling_input = resize2dLayer("OutputSize",outputSize,'Name','UpscaleInput','Method','bilinear');
layer_downsampling_input = resize2dLayer("OutputSize",[PatchSizeX,PatchSizeY],'Name','DownscaleInput','Method','bilinear');
if ResFlag
    addition_residual_layer = additionLayer(2,"Name",'AdditionResdiual');
end
%=================== MaxPool-Layer ===================%
p = maxPooling2dLayer(2,'Stride',2);
%=================== Upsampling-Layer ===================%
U = resize2dLayer('Scale',[2 2],'method','bilinear');
%=================== Block-Conv ===================%
LayerNumber = 1;
FilterSize = 3; NumFilt = 64;
bnFlag = 0; reluFlag = 1; SuffixLayer = '_1';
L_1 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,SuffixLayer,bnFlag, reluFlag );
SuffixLayer = '_2';
L_2 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,SuffixLayer,bnFlag, reluFlag );
p1 = p;
p1.Name = 'p1';
lgraph = layerGraph([InputLayer;layer_upsampling_input;L_1;L_2;p1]);
%=================== Block-1 ===================%
LayerNameIn = p1.Name;
Type = 5;Siffix = 'M1';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
p2 = p;
p2.Name = 'p2';
lgraph = addLayers(lgraph,p2);
lgraph = connectLayers(lgraph,LayerNameOut,p2.Name);
%=================== Block-2 ===================%
LayerNameIn = p2.Name;
Type = 4;Siffix = 'M2';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
p3 = p;
p3.Name = 'p3';
lgraph = addLayers(lgraph,p3);
lgraph = connectLayers(lgraph,LayerNameOut,p3.Name);
%=================== Block-3 ===================%
LayerNameIn = p3.Name;
Type = 3;Siffix = 'M3';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
p4 = p;
p4.Name = 'p4';
lgraph = addLayers(lgraph,p4);
lgraph = connectLayers(lgraph,LayerNameOut,p4.Name);
%=================== Block-4 ===================%
LayerNameIn = p4.Name;
Type = 2;Siffix = 'M4';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
p5 = p;
p5.Name = 'p5';
lgraph = addLayers(lgraph,p5);
lgraph = connectLayers(lgraph,LayerNameOut,p5.Name);
%=================== Block-5 ===================%
LayerNameIn = p5.Name;
Type = 1;Siffix = 'M5';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
%=================== Block-6 ===================%
U1 = U;
U1.Name = 'U1';
Siffix = 'C1';
DepthConcatLayer = depthConcatenationLayer(2,'Name',['depthCat_',Siffix]);
lgraph = addLayers(lgraph,[U1;DepthConcatLayer]);
lgraph = connectLayers(lgraph,LayerNameOut,U1.Name);
[lgraph,LayerNameOut] = SRD_S(lgraph,p4.Name,1,'S5');
lgraph = connectLayers(lgraph,LayerNameOut,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{2}]);
LayerNameIn = [DepthConcatLayer.Name,'/',DepthConcatLayer.OutputNames{1}];
Type = 2;Siffix = 'M6';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
%=================== Block-7 ===================%
U2 = U;
U2.Name = 'U2';
Siffix = 'C2';
DepthConcatLayer = depthConcatenationLayer(2,'Name',['depthCat_',Siffix]);
lgraph = addLayers(lgraph,[U2;DepthConcatLayer]);
lgraph = connectLayers(lgraph,LayerNameOut,U2.Name);
[lgraph,LayerNameOut] = SRD_S(lgraph,p3.Name,2,'S4');
lgraph = connectLayers(lgraph,LayerNameOut,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{2}]);
LayerNameIn = [DepthConcatLayer.Name,'/',DepthConcatLayer.OutputNames{1}];
Type = 3;Siffix = 'M7';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
%=================== Block-8 ===================%
U3 = U;
U3.Name = 'U3';
Siffix = 'C3';
DepthConcatLayer = depthConcatenationLayer(2,'Name',['depthCat_',Siffix]);
lgraph = addLayers(lgraph,[U3;DepthConcatLayer]);
lgraph = connectLayers(lgraph,LayerNameOut,U3.Name);
[lgraph,LayerNameOut] = SRD_S(lgraph,p2.Name,3,'S3');
lgraph = connectLayers(lgraph,LayerNameOut,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{2}]);
LayerNameIn = [DepthConcatLayer.Name,'/',DepthConcatLayer.OutputNames{1}];
Type = 4;Siffix = 'M8';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
%=================== Block-9 ===================%
U4 = U;
U4.Name = 'U4';
Siffix = 'C4';
DepthConcatLayer = depthConcatenationLayer(2,'Name',['depthCat_',Siffix]);
lgraph = addLayers(lgraph,[U4;DepthConcatLayer]);
lgraph = connectLayers(lgraph,LayerNameOut,U4.Name);
[lgraph,LayerNameOut] = SRD_S(lgraph,p1.Name,4,'S2');
lgraph = connectLayers(lgraph,LayerNameOut,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{2}]);
LayerNameIn = [DepthConcatLayer.Name,'/',DepthConcatLayer.OutputNames{1}];
Type = 5;Siffix = 'M9';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
%=================== Block-Concat ===================%
U5 = U;
U5.Name = 'U5';
Siffix = 'C5';
DepthConcatLayer = depthConcatenationLayer(2,'Name',['depthCat_',Siffix]);
lgraph = addLayers(lgraph,[U5;DepthConcatLayer]);
lgraph = connectLayers(lgraph,LayerNameOut,U5.Name);
[lgraph,LayerNameOut] = SRD_S(lgraph,layer_upsampling_input.Name,5,'S1');
lgraph = connectLayers(lgraph,LayerNameOut,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{2}]);
%=================== Block-Conv ===================%
LayerNumber = 2;
FilterSize = 3; NumFilt = 64;
bnFlag = 0; reluFlag = 1; SuffixLayer = '_1';
C_1 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,...
    SuffixLayer,bnFlag, reluFlag );
SuffixLayer = '_2';NumFilt = 32;
C_2 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,...
    SuffixLayer,bnFlag, reluFlag );
reluFlag = 0; SuffixLayer = '_3';NumFilt = NumCh;
C_3 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,...
    SuffixLayer,bnFlag, reluFlag );
lgraph = addLayers(lgraph,[C_1;C_2;C_3]);
lgraph = connectLayers(lgraph,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.OutputNames{1}],C_1(1).Name);
% ====================== Reg-Layer ==================== %
% RegLayer = regressionLayer('Name','routput');
RegLayer = maeRegressionLayer('routput');
% RegLayer = DCTRegressionLayer('routput');
% RegLayer = DCT_LOG_RegressionLayer('routput');
% =================== Connect Layers ===================%
LayerName = [C_3(end,1).Name,'/',C_3(end,1).OutputNames{1}];
if ResFlag
    lgraph = addLayers(lgraph,[layer_downsampling_input;addition_residual_layer;RegLayer]);
    lgraph = connectLayers(lgraph,InputLayer.Name,...
    [addition_residual_layer.Name,'/',addition_residual_layer.InputNames{2}]);
else
    lgraph = addLayers(lgraph,[layer_downsampling_input;RegLayer]);
end
lgraph = connectLayers(lgraph,LayerName,[layer_downsampling_input.Name]);
% analyzeNetwork(lgraph)
end

% ------------------------------------------------------------------------
function lgraph = MDRU_DCT_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag)
% PatchSizeX = 170;PatchSizeY = PatchSizeX;NumCh = 1;ResFlag=1;
disp('----------------- MDRU_DCT_Net with DCT-L2 Regression-----------------')
%% Input Layer
InputLayer = imageInputLayer([PatchSizeX,PatchSizeY,NumCh],...
    'Name','InputLayer','Normalization','none');

outputSize = ceil([PatchSizeX,PatchSizeY]/32)*32;
layer_upsampling_input = resize2dLayer("OutputSize",outputSize,'Name','UpscaleInput','Method','bilinear');
layer_downsampling_input = resize2dLayer("OutputSize",[PatchSizeX,PatchSizeY],'Name','DownscaleInput','Method','bilinear');
if ResFlag
    addition_residual_layer = additionLayer(2,"Name",'AdditionResdiual');
end
%=================== MaxPool-Layer ===================%
p = maxPooling2dLayer(2,'Stride',2);
%=================== Upsampling-Layer ===================%
U = resize2dLayer('Scale',[2 2],'method','bilinear');
%=================== Block-Conv ===================%
LayerNumber = 1;
FilterSize = 3; NumFilt = 64;
bnFlag = 0; reluFlag = 1; SuffixLayer = '_1';
L_1 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,SuffixLayer,bnFlag, reluFlag );
SuffixLayer = '_2';
L_2 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,SuffixLayer,bnFlag, reluFlag );
p1 = p;
p1.Name = 'p1';
lgraph = layerGraph([InputLayer;layer_upsampling_input;L_1;L_2;p1]);
%=================== Block-1 ===================%
LayerNameIn = p1.Name;
Type = 5;Siffix = 'M1';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
p2 = p;
p2.Name = 'p2';
lgraph = addLayers(lgraph,p2);
lgraph = connectLayers(lgraph,LayerNameOut,p2.Name);
%=================== Block-2 ===================%
LayerNameIn = p2.Name;
Type = 4;Siffix = 'M2';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
p3 = p;
p3.Name = 'p3';
lgraph = addLayers(lgraph,p3);
lgraph = connectLayers(lgraph,LayerNameOut,p3.Name);
%=================== Block-3 ===================%
LayerNameIn = p3.Name;
Type = 3;Siffix = 'M3';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
p4 = p;
p4.Name = 'p4';
lgraph = addLayers(lgraph,p4);
lgraph = connectLayers(lgraph,LayerNameOut,p4.Name);
%=================== Block-4 ===================%
LayerNameIn = p4.Name;
Type = 2;Siffix = 'M4';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
p5 = p;
p5.Name = 'p5';
lgraph = addLayers(lgraph,p5);
lgraph = connectLayers(lgraph,LayerNameOut,p5.Name);
%=================== Block-5 ===================%
LayerNameIn = p5.Name;
Type = 1;Siffix = 'M5';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
%=================== Block-6 ===================%
U1 = U;
U1.Name = 'U1';
Siffix = 'C1';
DepthConcatLayer = depthConcatenationLayer(2,'Name',['depthCat_',Siffix]);
lgraph = addLayers(lgraph,[U1;DepthConcatLayer]);
lgraph = connectLayers(lgraph,LayerNameOut,U1.Name);
[lgraph,LayerNameOut] = SRD_S(lgraph,p4.Name,1,'S5');
lgraph = connectLayers(lgraph,LayerNameOut,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{2}]);
LayerNameIn = [DepthConcatLayer.Name,'/',DepthConcatLayer.OutputNames{1}];
Type = 2;Siffix = 'M6';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
%=================== Block-7 ===================%
U2 = U;
U2.Name = 'U2';
Siffix = 'C2';
DepthConcatLayer = depthConcatenationLayer(2,'Name',['depthCat_',Siffix]);
lgraph = addLayers(lgraph,[U2;DepthConcatLayer]);
lgraph = connectLayers(lgraph,LayerNameOut,U2.Name);
[lgraph,LayerNameOut] = SRD_S(lgraph,p3.Name,2,'S4');
lgraph = connectLayers(lgraph,LayerNameOut,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{2}]);
LayerNameIn = [DepthConcatLayer.Name,'/',DepthConcatLayer.OutputNames{1}];
Type = 3;Siffix = 'M7';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
%=================== Block-8 ===================%
U3 = U;
U3.Name = 'U3';
Siffix = 'C3';
DepthConcatLayer = depthConcatenationLayer(2,'Name',['depthCat_',Siffix]);
lgraph = addLayers(lgraph,[U3;DepthConcatLayer]);
lgraph = connectLayers(lgraph,LayerNameOut,U3.Name);
[lgraph,LayerNameOut] = SRD_S(lgraph,p2.Name,3,'S3');
lgraph = connectLayers(lgraph,LayerNameOut,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{2}]);
LayerNameIn = [DepthConcatLayer.Name,'/',DepthConcatLayer.OutputNames{1}];
Type = 4;Siffix = 'M8';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
%=================== Block-9 ===================%
U4 = U;
U4.Name = 'U4';
Siffix = 'C4';
DepthConcatLayer = depthConcatenationLayer(2,'Name',['depthCat_',Siffix]);
lgraph = addLayers(lgraph,[U4;DepthConcatLayer]);
lgraph = connectLayers(lgraph,LayerNameOut,U4.Name);
[lgraph,LayerNameOut] = SRD_S(lgraph,p1.Name,4,'S2');
lgraph = connectLayers(lgraph,LayerNameOut,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{2}]);
LayerNameIn = [DepthConcatLayer.Name,'/',DepthConcatLayer.OutputNames{1}];
Type = 5;Siffix = 'M9';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
%=================== Block-Concat ===================%
U5 = U;
U5.Name = 'U5';
Siffix = 'C5';
DepthConcatLayer = depthConcatenationLayer(2,'Name',['depthCat_',Siffix]);
lgraph = addLayers(lgraph,[U5;DepthConcatLayer]);
lgraph = connectLayers(lgraph,LayerNameOut,U5.Name);
[lgraph,LayerNameOut] = SRD_S(lgraph,layer_upsampling_input.Name,5,'S1');
lgraph = connectLayers(lgraph,LayerNameOut,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{2}]);
%=================== Block-Conv ===================%
LayerNumber = 2;
FilterSize = 3; NumFilt = 64;
bnFlag = 0; reluFlag = 1; SuffixLayer = '_1';
C_1 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,...
    SuffixLayer,bnFlag, reluFlag );
SuffixLayer = '_2';NumFilt = 32;
C_2 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,...
    SuffixLayer,bnFlag, reluFlag );
reluFlag = 0; SuffixLayer = '_3';NumFilt = NumCh;
C_3 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,...
    SuffixLayer,bnFlag, reluFlag );
lgraph = addLayers(lgraph,[C_1;C_2;C_3]);
lgraph = connectLayers(lgraph,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.OutputNames{1}],C_1(1).Name);
% ====================== Reg-Layer ==================== %
% RegLayer = regressionLayer('Name','routput');
% RegLayer = maeRegressionLayer('routput');
RegLayer = DCTRegressionLayer('routput');
% RegLayer = DCT_LOG_RegressionLayer('routput');
% =================== Connect Layers ===================%
LayerName = [C_3(end,1).Name,'/',C_3(end,1).OutputNames{1}];
if ResFlag
    lgraph = addLayers(lgraph,[layer_downsampling_input;addition_residual_layer;RegLayer]);
    lgraph = connectLayers(lgraph,InputLayer.Name,...
    [addition_residual_layer.Name,'/',addition_residual_layer.InputNames{2}]);
else
    lgraph = addLayers(lgraph,[layer_downsampling_input;RegLayer]);
end
lgraph = connectLayers(lgraph,LayerName,[layer_downsampling_input.Name]);
% analyzeNetwork(lgraph)
end

% ------------------------------------------------------------------------
function lgraph = MDRU_DCT_LOG_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag)
% PatchSizeX = 170;PatchSizeY = PatchSizeX;NumCh = 1;ResFlag=1;
disp('----------------- MDRU_DCT_LOG_Net with DCT-LOG-L2 Regression-----------------')
%% Input Layer
InputLayer = imageInputLayer([PatchSizeX,PatchSizeY,NumCh],...
    'Name','InputLayer','Normalization','none');

outputSize = ceil([PatchSizeX,PatchSizeY]/32)*32;
layer_upsampling_input = resize2dLayer("OutputSize",outputSize,'Name','UpscaleInput','Method','bilinear');
layer_downsampling_input = resize2dLayer("OutputSize",[PatchSizeX,PatchSizeY],'Name','DownscaleInput','Method','bilinear');
if ResFlag
    addition_residual_layer = additionLayer(2,"Name",'AdditionResdiual');
end
%=================== MaxPool-Layer ===================%
p = maxPooling2dLayer(2,'Stride',2);
%=================== Upsampling-Layer ===================%
U = resize2dLayer('Scale',[2 2],'method','bilinear');
%=================== Block-Conv ===================%
LayerNumber = 1;
FilterSize = 3; NumFilt = 64;
bnFlag = 0; reluFlag = 1; SuffixLayer = '_1';
L_1 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,SuffixLayer,bnFlag, reluFlag );
SuffixLayer = '_2';
L_2 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,SuffixLayer,bnFlag, reluFlag );
p1 = p;
p1.Name = 'p1';
lgraph = layerGraph([InputLayer;layer_upsampling_input;L_1;L_2;p1]);
%=================== Block-1 ===================%
LayerNameIn = p1.Name;
Type = 5;Siffix = 'M1';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
p2 = p;
p2.Name = 'p2';
lgraph = addLayers(lgraph,p2);
lgraph = connectLayers(lgraph,LayerNameOut,p2.Name);
%=================== Block-2 ===================%
LayerNameIn = p2.Name;
Type = 4;Siffix = 'M2';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
p3 = p;
p3.Name = 'p3';
lgraph = addLayers(lgraph,p3);
lgraph = connectLayers(lgraph,LayerNameOut,p3.Name);
%=================== Block-3 ===================%
LayerNameIn = p3.Name;
Type = 3;Siffix = 'M3';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
p4 = p;
p4.Name = 'p4';
lgraph = addLayers(lgraph,p4);
lgraph = connectLayers(lgraph,LayerNameOut,p4.Name);
%=================== Block-4 ===================%
LayerNameIn = p4.Name;
Type = 2;Siffix = 'M4';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
p5 = p;
p5.Name = 'p5';
lgraph = addLayers(lgraph,p5);
lgraph = connectLayers(lgraph,LayerNameOut,p5.Name);
%=================== Block-5 ===================%
LayerNameIn = p5.Name;
Type = 1;Siffix = 'M5';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
%=================== Block-6 ===================%
U1 = U;
U1.Name = 'U1';
Siffix = 'C1';
DepthConcatLayer = depthConcatenationLayer(2,'Name',['depthCat_',Siffix]);
lgraph = addLayers(lgraph,[U1;DepthConcatLayer]);
lgraph = connectLayers(lgraph,LayerNameOut,U1.Name);
[lgraph,LayerNameOut] = SRD_S(lgraph,p4.Name,1,'S5');
lgraph = connectLayers(lgraph,LayerNameOut,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{2}]);
LayerNameIn = [DepthConcatLayer.Name,'/',DepthConcatLayer.OutputNames{1}];
Type = 2;Siffix = 'M6';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
%=================== Block-7 ===================%
U2 = U;
U2.Name = 'U2';
Siffix = 'C2';
DepthConcatLayer = depthConcatenationLayer(2,'Name',['depthCat_',Siffix]);
lgraph = addLayers(lgraph,[U2;DepthConcatLayer]);
lgraph = connectLayers(lgraph,LayerNameOut,U2.Name);
[lgraph,LayerNameOut] = SRD_S(lgraph,p3.Name,2,'S4');
lgraph = connectLayers(lgraph,LayerNameOut,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{2}]);
LayerNameIn = [DepthConcatLayer.Name,'/',DepthConcatLayer.OutputNames{1}];
Type = 3;Siffix = 'M7';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
%=================== Block-8 ===================%
U3 = U;
U3.Name = 'U3';
Siffix = 'C3';
DepthConcatLayer = depthConcatenationLayer(2,'Name',['depthCat_',Siffix]);
lgraph = addLayers(lgraph,[U3;DepthConcatLayer]);
lgraph = connectLayers(lgraph,LayerNameOut,U3.Name);
[lgraph,LayerNameOut] = SRD_S(lgraph,p2.Name,3,'S3');
lgraph = connectLayers(lgraph,LayerNameOut,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{2}]);
LayerNameIn = [DepthConcatLayer.Name,'/',DepthConcatLayer.OutputNames{1}];
Type = 4;Siffix = 'M8';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
%=================== Block-9 ===================%
U4 = U;
U4.Name = 'U4';
Siffix = 'C4';
DepthConcatLayer = depthConcatenationLayer(2,'Name',['depthCat_',Siffix]);
lgraph = addLayers(lgraph,[U4;DepthConcatLayer]);
lgraph = connectLayers(lgraph,LayerNameOut,U4.Name);
[lgraph,LayerNameOut] = SRD_S(lgraph,p1.Name,4,'S2');
lgraph = connectLayers(lgraph,LayerNameOut,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{2}]);
LayerNameIn = [DepthConcatLayer.Name,'/',DepthConcatLayer.OutputNames{1}];
Type = 5;Siffix = 'M9';
[lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix);
%=================== Block-Concat ===================%
U5 = U;
U5.Name = 'U5';
Siffix = 'C5';
DepthConcatLayer = depthConcatenationLayer(2,'Name',['depthCat_',Siffix]);
lgraph = addLayers(lgraph,[U5;DepthConcatLayer]);
lgraph = connectLayers(lgraph,LayerNameOut,U5.Name);
[lgraph,LayerNameOut] = SRD_S(lgraph,layer_upsampling_input.Name,5,'S1');
lgraph = connectLayers(lgraph,LayerNameOut,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{2}]);
%=================== Block-Conv ===================%
LayerNumber = 2;
FilterSize = 3; NumFilt = 64;
bnFlag = 0; reluFlag = 1; SuffixLayer = '_1';
C_1 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,...
    SuffixLayer,bnFlag, reluFlag );
SuffixLayer = '_2';NumFilt = 32;
C_2 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,...
    SuffixLayer,bnFlag, reluFlag );
reluFlag = 0; SuffixLayer = '_3';NumFilt = NumCh;
C_3 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,...
    SuffixLayer,bnFlag, reluFlag );
lgraph = addLayers(lgraph,[C_1;C_2;C_3]);
lgraph = connectLayers(lgraph,...
    [DepthConcatLayer.Name,'/',DepthConcatLayer.OutputNames{1}],C_1(1).Name);
% ====================== Reg-Layer ==================== %
% RegLayer = regressionLayer('Name','routput');
% RegLayer = maeRegressionLayer('routput');
% RegLayer = DCTRegressionLayer('routput');
RegLayer = DCT_LOG_RegressionLayer('routput');
% =================== Connect Layers ===================%
LayerName = [C_3(end,1).Name,'/',C_3(end,1).OutputNames{1}];
if ResFlag
    lgraph = addLayers(lgraph,[layer_downsampling_input;addition_residual_layer;RegLayer]);
    lgraph = connectLayers(lgraph,InputLayer.Name,...
    [addition_residual_layer.Name,'/',addition_residual_layer.InputNames{2}]);
else
    lgraph = addLayers(lgraph,[layer_downsampling_input;RegLayer]);
end
lgraph = connectLayers(lgraph,LayerName,[layer_downsampling_input.Name]);
% analyzeNetwork(lgraph)
end
%% Multi Dilated Convolution Module 
function [lgraph,LayerNameOut] = MDC_M(lgraph,LayerNameIn,Type,Siffix)
% Type = 3;Siffix = 'T1_a';
FilterSize = 3;
switch Type
    case 1, NumFilt = 256;
    case 2, NumFilt = 128;
    case 3, NumFilt = 64;
    case 4, NumFilt = 64;
    case 5, NumFilt = 32;
    otherwise, error('wrong type 1 2 3 4 5')
end
if Type > 1
    DepthConcatLayer = depthConcatenationLayer(Type,'Name',['depthCat_',Siffix]);
    LayerNameOut = [DepthConcatLayer.Name,'/',DepthConcatLayer.OutputNames{1}];
    lgraph = addLayers(lgraph,DepthConcatLayer);
    for ii = 1:Type
        LayerNumber = ii; DilationFactor = ii;
        bnFlag = 0; reluFlag = 1; SuffixLayer = ['_',Siffix];
        layer_conv = GenerateDilatedConvolutionModule(...
                LayerNumber,FilterSize,NumFilt,SuffixLayer,bnFlag,...
                    reluFlag,DilationFactor);
        lgraph = addLayers(lgraph,layer_conv);
        % Conect conv layer to addition layer
        lgraph = connectLayers(lgraph,...
            [layer_conv(end).Name,'/',layer_conv(end).OutputNames{1}],...
            [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{ii}]);
        % Connect input layer to conv layer
        lgraph = connectLayers(lgraph,LayerNameIn,...
            [layer_conv(1).Name,'/',layer_conv(1).InputNames{1}]);
    end
elseif Type == 1
    LayerNumber = Type;
    bnFlag = 0; reluFlag = 1; SuffixLayer = ['_',Siffix];
    layer_conv = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,...
        SuffixLayer,bnFlag, reluFlag );
%     layer_conv = convolution2dLayer([FilterSize FilterSize], NumFilt, ...
%             'Name', ['conv_',num2str(1),'_',Siffix], ...
%             'Padding', 'same');
    lgraph = addLayers(lgraph,layer_conv);
    lgraph = connectLayers(lgraph,LayerNameIn,...
            [layer_conv(1).Name,'/',layer_conv(1).InputNames{1}]);
    LayerNameOut = layer_conv(end).Name;
else
    error('wrng type 1 2 3 4 5')
end
end
%% SRD Module
function [lgraph,LayerNameOut] = SRD_S(lgraph,LayerNameIn,Blocks,Siffix)

LayerNameOut = LayerNameIn;
for ii = 1:Blocks
    Suffix = ['DR',num2str(ii),'_',Siffix];
    [lgraph,LayerNameOutDR] = DR_block(lgraph,LayerNameOut,ii,Suffix);
    DepthConcatLayer = ...
        depthConcatenationLayer(2,'Name',['depthCat_',Suffix]);
    lgraph = addLayers(lgraph,DepthConcatLayer);
    lgraph = connectLayers(lgraph,LayerNameOutDR,...
        [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{1}]);
    lgraph = connectLayers(lgraph,LayerNameOut,...
        [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{2}]);    
    LayerNameOut = [DepthConcatLayer.Name,'/',DepthConcatLayer.OutputNames{1}];
end
end
%% Dense Residual Block
function [lgraph,LayerNameOut] = DR_block(lgraph,LayerNameIn,alpha,Siffix)

FilterSize=3; NumFilt=32; DilationFactor = 1; bnFlag = 0; reluFlag = 1; 
LayerNumber = 1;SuffixLayer = ['_N_',Siffix];
layer_conv_1 = GenerateDilatedConvolutionModule(...
    LayerNumber,FilterSize,NumFilt,SuffixLayer,bnFlag,...
    reluFlag,DilationFactor);
LayerNumber = 2;SuffixLayer = ['_D_',Siffix];DilationFactor = alpha; 
layer_conv_2 = GenerateDilatedConvolutionModule(...
    LayerNumber,FilterSize,NumFilt,SuffixLayer,bnFlag,...
    reluFlag,DilationFactor);
lgraph = addLayers(lgraph,[layer_conv_1;layer_conv_2]);
% Connect input layer to conv layer
lgraph = connectLayers(lgraph,LayerNameIn,...
    [layer_conv_1(1).Name,'/',layer_conv_1(1).InputNames{1}]);
LayerNameOut = [layer_conv_2(end).Name,'/',layer_conv_2(end).OutputNames{1}];
end

% ------------------------------------------------------------------------
% function for Multiscale Residual Dual Dense Attention Net
function lgraph = MRDDA_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag)
% PatchSizeX = 196;PatchSizeY = PatchSizeX;NumCh = 1;ResFlag = 0;
disp('----------------- MRDDA_Net with L2 Regression-----------------')
NumFiltIn = 64; 
NumFiltOut = 64;
%% Input Layer
InputLayer = imageInputLayer([PatchSizeX,PatchSizeY,NumCh],...
    'Name','InputLayer','Normalization','none');
lgraph = layerGraph(InputLayer);
[lgraph,LayerNameOut_MSB] = H_MSB(lgraph,InputLayer.Name,[],'MSB',NumFiltIn,NumFiltIn);
% NumFiltOut = 512;
Siffix = 'A';
[lgraph,LayerNameOut_RDDAB_A] = H_RDDAB_Mod(lgraph,LayerNameOut_MSB,[],...
                                            Siffix,NumFiltIn,NumFiltOut);
% NumFiltOut = 512;
Siffix = 'B';
[lgraph,LayerNameOut_RDDAB_B] = H_RDDAB_Mod(lgraph,LayerNameOut_RDDAB_A,[],...
                                            Siffix,NumFiltIn,NumFiltOut);
% NumFiltOut = 512;
Siffix = 'C';
[lgraph,LayerNameOut_RDDAB_C] = H_RDDAB_Mod(lgraph,LayerNameOut_RDDAB_B,[],...
                                            Siffix,NumFiltIn,NumFiltOut);
DepthConcat_layer_End = depthConcatenationLayer(3,...
    'Name','depthCat_last_mrddanet');

FilterSize = 3; NumFilt = 1;
layer_conv_mrdda = convolution2dLayer([FilterSize FilterSize], NumFilt, ...
    'Name', 'conv_MRDDA', ...
    'Padding', 'same');
add_layer_mrdda = additionLayer(2,'Name','add_MRDDA');
% LOSS Layer
layer_reg = regressionLayer('Name','mse_routput');
% layer_reg = maeRegressionLayer('mae_routput');
% layer_reg = DCTRegressionLayer('dct_l2_routput');
% layer_reg = DCT_LOG_RegressionLayer('log_dct_l2_routput');

% Add DepthConcat_layer_End, layer_conv_mrdda and add_layer_mrdda to lgraph

if ResFlag == 1
lgraph = addLayers(lgraph,...
    [DepthConcat_layer_End;layer_conv_mrdda;add_layer_mrdda;layer_reg]);
else
lgraph = addLayers(lgraph,...
    [DepthConcat_layer_End;layer_conv_mrdda;layer_reg]);
end    

% connect LayerNameOut_RDDAB_C to Depth_concat
lgraph = connectLayers(lgraph,LayerNameOut_RDDAB_C,...
     [DepthConcat_layer_End.Name,'/',DepthConcat_layer_End.InputNames{1}]);
% connect LayerNameOut_RDDAB_B to Depth_concat
lgraph = connectLayers(lgraph,LayerNameOut_RDDAB_B,...
     [DepthConcat_layer_End.Name,'/',DepthConcat_layer_End.InputNames{2}]);
% connect LayerNameOut_RDDAB_A to Depth_concat
lgraph = connectLayers(lgraph,LayerNameOut_RDDAB_A,...
     [DepthConcat_layer_End.Name,'/',DepthConcat_layer_End.InputNames{3}]);
% connect LayerNameOut_RDDAB_A to Depth_concat
if ResFlag == 1
lgraph = connectLayers(lgraph,InputLayer.Name,...
     [add_layer_mrdda.Name,'/',add_layer_mrdda.InputNames{2}]);
end
% analyzeNetwork(lgraph)
end

% ------------------------------------------------------------------------
% function for Multiscale Residual Dual Dense Attention Net
function lgraph = MRDDA_MAE_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag)
% PatchSizeX = 196;PatchSizeY = PatchSizeX;NumCh = 1;ResFlag = 0;
disp('----------------- MRDDA_MAE_Net with L1 Regression-----------------')
NumFiltIn = 64; 
NumFiltOut = 64;
%% Input Layer
InputLayer = imageInputLayer([PatchSizeX,PatchSizeY,NumCh],...
    'Name','InputLayer','Normalization','none');
lgraph = layerGraph(InputLayer);
[lgraph,LayerNameOut_MSB] = H_MSB(lgraph,InputLayer.Name,[],'MSB',NumFiltIn,NumFiltIn);
% NumFiltOut = 512;
Siffix = 'A';
[lgraph,LayerNameOut_RDDAB_A] = H_RDDAB_Mod(lgraph,LayerNameOut_MSB,[],...
                                            Siffix,NumFiltIn,NumFiltOut);
% NumFiltOut = 512;
Siffix = 'B';
[lgraph,LayerNameOut_RDDAB_B] = H_RDDAB_Mod(lgraph,LayerNameOut_RDDAB_A,[],...
                                            Siffix,NumFiltIn,NumFiltOut);
% NumFiltOut = 512;
Siffix = 'C';
[lgraph,LayerNameOut_RDDAB_C] = H_RDDAB_Mod(lgraph,LayerNameOut_RDDAB_B,[],...
                                            Siffix,NumFiltIn,NumFiltOut);
DepthConcat_layer_End = depthConcatenationLayer(3,...
    'Name','depthCat_last_mrddanet');

FilterSize = 3; NumFilt = 1;
layer_conv_mrdda = convolution2dLayer([FilterSize FilterSize], NumFilt, ...
    'Name', 'conv_MRDDA', ...
    'Padding', 'same');
add_layer_mrdda = additionLayer(2,'Name','add_MRDDA');
% LOSS Layer
% layer_reg = regressionLayer('Name','mse_routput');
layer_reg = maeRegressionLayer('mae_routput');
% layer_reg = DCTRegressionLayer('dct_l2_routput');
% layer_reg = DCT_LOG_RegressionLayer('log_dct_l2_routput');

% Add DepthConcat_layer_End, layer_conv_mrdda and add_layer_mrdda to lgraph

if ResFlag == 1
lgraph = addLayers(lgraph,...
    [DepthConcat_layer_End;layer_conv_mrdda;add_layer_mrdda;layer_reg]);
else
lgraph = addLayers(lgraph,...
    [DepthConcat_layer_End;layer_conv_mrdda;layer_reg]);
end    

% connect LayerNameOut_RDDAB_C to Depth_concat
lgraph = connectLayers(lgraph,LayerNameOut_RDDAB_C,...
     [DepthConcat_layer_End.Name,'/',DepthConcat_layer_End.InputNames{1}]);
% connect LayerNameOut_RDDAB_B to Depth_concat
lgraph = connectLayers(lgraph,LayerNameOut_RDDAB_B,...
     [DepthConcat_layer_End.Name,'/',DepthConcat_layer_End.InputNames{2}]);
% connect LayerNameOut_RDDAB_A to Depth_concat
lgraph = connectLayers(lgraph,LayerNameOut_RDDAB_A,...
     [DepthConcat_layer_End.Name,'/',DepthConcat_layer_End.InputNames{3}]);
% connect LayerNameOut_RDDAB_A to Depth_concat
if ResFlag == 1
lgraph = connectLayers(lgraph,InputLayer.Name,...
     [add_layer_mrdda.Name,'/',add_layer_mrdda.InputNames{2}]);
end
% analyzeNetwork(lgraph)
end

% ------------------------------------------------------------------------
% function for Multiscale Residual Dual Dense Attention Net
function lgraph = MRDDA_DCT_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag)
% PatchSizeX = 196;PatchSizeY = PatchSizeX;NumCh = 1;ResFlag = 0;
disp('----------------- MRDDA_DCT_Net with DCT-L2 Regression-----------------')
NumFiltIn = 64; 
NumFiltOut = 64;
%% Input Layer
InputLayer = imageInputLayer([PatchSizeX,PatchSizeY,NumCh],...
    'Name','InputLayer','Normalization','none');
lgraph = layerGraph(InputLayer);
[lgraph,LayerNameOut_MSB] = H_MSB(lgraph,InputLayer.Name,[],'MSB',NumFiltIn,NumFiltIn);
% NumFiltOut = 512;
Siffix = 'A';
[lgraph,LayerNameOut_RDDAB_A] = H_RDDAB_Mod(lgraph,LayerNameOut_MSB,[],...
                                            Siffix,NumFiltIn,NumFiltOut);
% NumFiltOut = 512;
Siffix = 'B';
[lgraph,LayerNameOut_RDDAB_B] = H_RDDAB_Mod(lgraph,LayerNameOut_RDDAB_A,[],...
                                            Siffix,NumFiltIn,NumFiltOut);
% NumFiltOut = 512;
Siffix = 'C';
[lgraph,LayerNameOut_RDDAB_C] = H_RDDAB_Mod(lgraph,LayerNameOut_RDDAB_B,[],...
                                            Siffix,NumFiltIn,NumFiltOut);
DepthConcat_layer_End = depthConcatenationLayer(3,...
    'Name','depthCat_last_mrddanet');

FilterSize = 3; NumFilt = 1;
layer_conv_mrdda = convolution2dLayer([FilterSize FilterSize], NumFilt, ...
    'Name', 'conv_MRDDA', ...
    'Padding', 'same');
add_layer_mrdda = additionLayer(2,'Name','add_MRDDA');
% LOSS Layer
% layer_reg = regressionLayer('Name','mse_routput');
% layer_reg = maeRegressionLayer('mae_routput');
layer_reg = DCTRegressionLayer('dct_l2_routput');
% layer_reg = DCT_LOG_RegressionLayer('log_dct_l2_routput');

% Add DepthConcat_layer_End, layer_conv_mrdda and add_layer_mrdda to lgraph

if ResFlag == 1
lgraph = addLayers(lgraph,...
    [DepthConcat_layer_End;layer_conv_mrdda;add_layer_mrdda;layer_reg]);
else
lgraph = addLayers(lgraph,...
    [DepthConcat_layer_End;layer_conv_mrdda;layer_reg]);
end    

% connect LayerNameOut_RDDAB_C to Depth_concat
lgraph = connectLayers(lgraph,LayerNameOut_RDDAB_C,...
     [DepthConcat_layer_End.Name,'/',DepthConcat_layer_End.InputNames{1}]);
% connect LayerNameOut_RDDAB_B to Depth_concat
lgraph = connectLayers(lgraph,LayerNameOut_RDDAB_B,...
     [DepthConcat_layer_End.Name,'/',DepthConcat_layer_End.InputNames{2}]);
% connect LayerNameOut_RDDAB_A to Depth_concat
lgraph = connectLayers(lgraph,LayerNameOut_RDDAB_A,...
     [DepthConcat_layer_End.Name,'/',DepthConcat_layer_End.InputNames{3}]);
% connect LayerNameOut_RDDAB_A to Depth_concat
if ResFlag == 1
lgraph = connectLayers(lgraph,InputLayer.Name,...
     [add_layer_mrdda.Name,'/',add_layer_mrdda.InputNames{2}]);
end
% analyzeNetwork(lgraph)
end

% ------------------------------------------------------------------------
% function for Multiscale Residual Dual Dense Attention Net
function lgraph = MRDDA_DCT_LOG_Net(PatchSizeX,PatchSizeY,NumCh,ResFlag)
% PatchSizeX = 196;PatchSizeY = PatchSizeX;NumCh = 1;ResFlag = 0;
disp('----------------- MRDDA_DCT_LOG_Net with LOG-DCT-L2 Regression-----------------')
NumFiltIn = 64; 
NumFiltOut = 64;
%% Input Layer
InputLayer = imageInputLayer([PatchSizeX,PatchSizeY,NumCh],...
    'Name','InputLayer','Normalization','none');
lgraph = layerGraph(InputLayer);
[lgraph,LayerNameOut_MSB] = H_MSB(lgraph,InputLayer.Name,[],'MSB',NumFiltIn,NumFiltIn);
% NumFiltOut = 512;
Siffix = 'A';
[lgraph,LayerNameOut_RDDAB_A] = H_RDDAB_Mod(lgraph,LayerNameOut_MSB,[],...
                                            Siffix,NumFiltIn,NumFiltOut);
% NumFiltOut = 512;
Siffix = 'B';
[lgraph,LayerNameOut_RDDAB_B] = H_RDDAB_Mod(lgraph,LayerNameOut_RDDAB_A,[],...
                                            Siffix,NumFiltIn,NumFiltOut);
% NumFiltOut = 512;
Siffix = 'C';
[lgraph,LayerNameOut_RDDAB_C] = H_RDDAB_Mod(lgraph,LayerNameOut_RDDAB_B,[],...
                                            Siffix,NumFiltIn,NumFiltOut);
DepthConcat_layer_End = depthConcatenationLayer(3,...
    'Name','depthCat_last_mrddanet');

FilterSize = 3; NumFilt = 1;
layer_conv_mrdda = convolution2dLayer([FilterSize FilterSize], NumFilt, ...
    'Name', 'conv_MRDDA', ...
    'Padding', 'same');
add_layer_mrdda = additionLayer(2,'Name','add_MRDDA');
% LOSS Layer
% layer_reg = regressionLayer('Name','mse_routput');
% layer_reg = maeRegressionLayer('mae_routput');
% layer_reg = DCTRegressionLayer('dct_l2_routput');
layer_reg = DCT_LOG_RegressionLayer('log_dct_l2_routput');

% Add DepthConcat_layer_End, layer_conv_mrdda and add_layer_mrdda to lgraph

if ResFlag == 1
lgraph = addLayers(lgraph,...
    [DepthConcat_layer_End;layer_conv_mrdda;add_layer_mrdda;layer_reg]);
else
lgraph = addLayers(lgraph,...
    [DepthConcat_layer_End;layer_conv_mrdda;layer_reg]);
end    

% connect LayerNameOut_RDDAB_C to Depth_concat
lgraph = connectLayers(lgraph,LayerNameOut_RDDAB_C,...
     [DepthConcat_layer_End.Name,'/',DepthConcat_layer_End.InputNames{1}]);
% connect LayerNameOut_RDDAB_B to Depth_concat
lgraph = connectLayers(lgraph,LayerNameOut_RDDAB_B,...
     [DepthConcat_layer_End.Name,'/',DepthConcat_layer_End.InputNames{2}]);
% connect LayerNameOut_RDDAB_A to Depth_concat
lgraph = connectLayers(lgraph,LayerNameOut_RDDAB_A,...
     [DepthConcat_layer_End.Name,'/',DepthConcat_layer_End.InputNames{3}]);
% connect LayerNameOut_RDDAB_A to Depth_concat
if ResFlag == 1
lgraph = connectLayers(lgraph,InputLayer.Name,...
     [add_layer_mrdda.Name,'/',add_layer_mrdda.InputNames{2}]);
end
% analyzeNetwork(lgraph)
end
%% Mutli-scale Block
function [lgraph,LayerNameOut] = H_MSB(lgraph,LayerNameIn,Type,Siffix,NumFiltIn,NumFiltOut)
% Mutli-scale Block
% Define intial conv-relu-layer
LayerNumber = 1; FilterSize = 3; % NumFiltIn = 64;
bnFlag = 0; reluFlag = 1;% LayerNameIn = 'MSB'; 
layer_1 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFiltIn,...
    Siffix,bnFlag,reluFlag);
% deffine conv-layer with 3 5 7 receptive fields
LayerNumber = 21; FilterSize = 3; %  NumFiltIn = 64;
bnFlag = 0; reluFlag = 0;% LayerNameIn = 'MSB'; 
layer_21 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFiltIn,...
    Siffix,bnFlag,reluFlag);
LayerNumber = 22; FilterSize = 5; %  NumFiltIn = 64;
bnFlag = 0; reluFlag = 0;% LayerNameIn = 'MSB'; 
layer_22 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFiltIn,...
    Siffix,bnFlag,reluFlag);
LayerNumber = 23; FilterSize = 7; %  NumFiltIn = 64;
bnFlag = 0; reluFlag = 0;% LayerNameIn = 'MSB'; 
layer_23 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFiltIn,...
    Siffix,bnFlag,reluFlag);
% Define concat-layer
% LayerNameIn = 'MSB';
DepthConcatLayer = depthConcatenationLayer(3,'Name',['depthCat_',Siffix]);
% define final conv-relu-layer
LayerNumber = 3; FilterSize = 3;
bnFlag = 0; reluFlag = 1;% LayerNameIn = 'MSB'; 
layer_3 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFiltOut,...
    Siffix,bnFlag,reluFlag);
% Add all layers to lgraph and connect series layers
lgraph = addLayers(lgraph,[layer_1;layer_21;DepthConcatLayer;layer_3]);
lgraph = addLayers(lgraph,layer_22);
lgraph = addLayers(lgraph,layer_23);
% connect LayerNameIn to layer-1
lgraph = connectLayers(lgraph,...
            LayerNameIn,[layer_1(1).Name,'/',layer_1(1).InputNames{1}]);
% connect layer-1 to layer-22 and layer-23
lgraph = connectLayers(lgraph,...
            [layer_1(end).Name,'/',layer_1(end).OutputNames{1}],...
            [layer_22.Name,'/',layer_22.InputNames{1}]);
lgraph = connectLayers(lgraph,...
            [layer_1(end).Name,'/',layer_1(end).OutputNames{1}],...
            [layer_23.Name,'/',layer_23.InputNames{1}]);
% Connect layer-22 and layer-23 to concat-layer input
lgraph = connectLayers(lgraph,...
            [layer_22(end).Name,'/',layer_22(end).OutputNames{1}],...
            [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{2}]);
lgraph = connectLayers(lgraph,...
            [layer_23(end).Name,'/',layer_23(end).OutputNames{1}],...
            [DepthConcatLayer.Name,'/',DepthConcatLayer.InputNames{3}]);
% define output layer name
LayerNameOut = [layer_3(end).Name,'/',layer_3(end).OutputNames{1}];
end
%% H RDDAB Module
function [lgraph,LayerNameOut] = H_RDDAB_Mod(lgraph,LayerNameIn,Type,...
                                            Siffix,NumFiltIn,NumFiltOut)
Siffix_layer = ['MOD_',Siffix];
[lgraph,LayerNameOut_RDDAB_1] = H_RDDAB(lgraph,LayerNameIn,...
    [],['1_',Siffix_layer],NumFiltIn,NumFiltOut);
[lgraph,LayerNameOut_RDDAB_2] = H_RDDAB(lgraph,LayerNameOut_RDDAB_1,...
    [],['2_',Siffix_layer],NumFiltIn,NumFiltOut);
FilterSize = 3; NumFilt = NumFiltIn;
layer_conv_RDDAB_end = convolution2dLayer([FilterSize FilterSize], NumFilt, ...
    'Name', ['conv_RDDAB_',Siffix_layer], ...
    'Padding', 'same');
add_layer_rddab_mod = additionLayer(2,'Name',['add_RDDAB_',Siffix_layer]);
% Add layer_conv_RDDAB_end and add_layer_end to lgraph
lgraph = addLayers(lgraph,[layer_conv_RDDAB_end;add_layer_rddab_mod]);
% connect LayerNameOut_RDDAB_2 to layer_conv_RDDAB_end
lgraph = connectLayers(lgraph,LayerNameOut_RDDAB_2,...
    [layer_conv_RDDAB_end.Name,'/',layer_conv_RDDAB_end.InputNames{1}]);
% connect LayerNameIn to add_layer_end
lgraph = connectLayers(lgraph,LayerNameIn,...
    [add_layer_rddab_mod.Name,'/',add_layer_rddab_mod.InputNames{2}]);
% Define output-layer name
LayerNameOut = [add_layer_rddab_mod.Name,'/',...
    add_layer_rddab_mod.OutputNames{1}];


end
%% H Residual Dense Dual Attention Block
function [lgraph,LayerNameOut] = H_RDDAB(lgraph,LayerNameIn,Type,Siffix,NumFiltIn,NumFiltOut)
% RD-1-2-3
Siffix_layer = ['RDDAB_',Siffix];
[lgraph,LayerNameOut_RD] = H_RD(lgraph,LayerNameIn,Type,Siffix_layer,NumFiltIn,NumFiltOut);
[lgraph,LayerNameOut_CAB] = H_CAB(lgraph,LayerNameOut_RD,Type,Siffix_layer);
[lgraph,LayerNameOut_PAB] = H_PAB(lgraph,LayerNameOut_RD,Type,Siffix_layer);
DepthConcat_layer_End = depthConcatenationLayer(2,'Name',...
    ['depthCat_last_',Siffix_layer]);
FilterSize = 3; NumFilt = NumFiltIn;
layer_conv_end = convolution2dLayer([FilterSize FilterSize], NumFilt, ...
    'Name', ['conv_end_',Siffix_layer], ...
    'Padding', 'same');
add_layer_end = additionLayer(2,'Name',['add_end_',Siffix_layer]);

% Add depth and conv layer to lgraph
lgraph = addLayers(lgraph,[DepthConcat_layer_End;layer_conv_end;add_layer_end]);
% connect LayerNameOut_CAB to DepthConcat_layer_1
lgraph = connectLayers(lgraph,LayerNameOut_CAB,...
    [DepthConcat_layer_End.Name,'/',DepthConcat_layer_End.InputNames{1}]);
% Connect LayerNameOut_PAB to DepthConcat_layer_End input
lgraph = connectLayers(lgraph,LayerNameOut_PAB,...
    [DepthConcat_layer_End.Name,'/',DepthConcat_layer_End.InputNames{2}]);
% Connect LayerNameIn and add_layer_end
lgraph = connectLayers(lgraph,LayerNameIn,...
    [add_layer_end.Name,'/',add_layer_end.InputNames{2}]);

% output layer name
LayerNameIn2 = [add_layer_end.Name,'/',add_layer_end.OutputNames{1}];
LayerNameOut = LayerNameIn2;
end
%% H PreLu-Conv-Conv
function [layers] = H_PCC(LayerNumberM,FilterSize,NumFilt,...
    Siffix,bnFlag,reluFlag,NumCh)

layer_PreLu = preluLayer(NumCh,['PreLu_',num2str(LayerNumberM),'_',Siffix]);

LayerNumber = 1; % FilterSize = 3; NumFilt = 64;
% bnFlag = 0; reluFlag = 0; % LayerNameIn = 'MSB'; 
layer_1 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,...
    [num2str(LayerNumberM),'_',Siffix],bnFlag,reluFlag);

LayerNumber = 2; % FilterSize = 3; NumFilt = 64;
% bnFlag = 0; reluFlag = 0; % LayerNameIn = 'MSB'; 
layer_2 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,...
    [num2str(LayerNumberM),'_',Siffix],bnFlag,reluFlag);

layers = [layer_PreLu;layer_1;layer_2];

end
%% H Residual Dense Module
function [lgraph,LayerNameOut] = H_RD_MOD(lgraph,FilterSize,NumFilt,...
    bnFlag,reluFlag,LayerNameIn,LayerNumber,Siffix_layer,NumCh)

layers_1 = H_PCC(LayerNumber,FilterSize,NumFilt,...
    Siffix_layer,bnFlag,reluFlag,NumCh);
add_layer_1 = additionLayer(2,'Name',['add_',num2str(LayerNumber),'_',Siffix_layer]);

lgraph = addLayers(lgraph,[layers_1;add_layer_1]);
% connect LayerNameIn to layer-1
lgraph = connectLayers(lgraph,...
            LayerNameIn,[layers_1(1).Name,'/',layers_1(1).InputNames{1}]);
% Connect LayerNameIn and layer-add to concat-layer input
lgraph = connectLayers(lgraph,...
            LayerNameIn,[add_layer_1.Name,'/',add_layer_1.InputNames{2}]);
% define output layer name
LayerNameOut = [add_layer_1.Name,'/',add_layer_1.OutputNames{1}];
end
%% H Residual Dense
function [lgraph,LayerNameOut] = H_RD(lgraph,LayerNameIn,Type,Siffix,NumFiltIn,NumFiltOut)
bnFlag=0;reluFlag=0;
FilterSize=3;Siffix_layer = ['RD_MOD_',Siffix];
% NumFiltIn=64;
% RD Module-1
LayerNumber=1;NumCh=NumFiltIn;
[lgraph,LayerNameOut_1] = H_RD_MOD(lgraph,FilterSize,NumFiltIn,...
    bnFlag,reluFlag,LayerNameIn,LayerNumber,Siffix_layer,NumCh);
DepthConcat_layer_1 = depthConcatenationLayer(2,'Name',...
    ['depthCat_',num2str(LayerNumber),'_',Siffix_layer]);
lgraph = addLayers(lgraph,DepthConcat_layer_1);
% Connect LayerNameIn to DepthConcat_layer_1
lgraph = connectLayers(lgraph,LayerNameOut_1,...
    [DepthConcat_layer_1.Name,'/',DepthConcat_layer_1.InputNames{1}]);
lgraph = connectLayers(lgraph,LayerNameIn,...
    [DepthConcat_layer_1.Name,'/',DepthConcat_layer_1.InputNames{2}]);
% output layer name
LayerNameIn2 = [DepthConcat_layer_1.Name,'/',DepthConcat_layer_1.OutputNames{1}];

% RD Module-2
LayerNumber=2;NumCh=NumFiltIn*2;
[lgraph,LayerNameOut_2] = H_RD_MOD(lgraph,FilterSize,NumFiltIn*2,...
    bnFlag,reluFlag,LayerNameIn2,LayerNumber,Siffix_layer,NumCh);
DepthConcat_layer_2 = depthConcatenationLayer(3,'Name',...
    ['depthCat_',num2str(LayerNumber),'_',Siffix_layer]);
lgraph = addLayers(lgraph,DepthConcat_layer_2);
% Connect LayerNameIn to DepthConcat_layer_1
lgraph = connectLayers(lgraph,LayerNameOut_2,...
    [DepthConcat_layer_2.Name,'/',DepthConcat_layer_2.InputNames{1}]);
lgraph = connectLayers(lgraph,LayerNameIn2,...
    [DepthConcat_layer_2.Name,'/',DepthConcat_layer_2.InputNames{2}]);
lgraph = connectLayers(lgraph,LayerNameIn,...
    [DepthConcat_layer_2.Name,'/',DepthConcat_layer_2.InputNames{3}]);

% output layer name
LayerNameIn3 = [DepthConcat_layer_2.Name,'/',DepthConcat_layer_2.OutputNames{1}];

% RD Module-3
LayerNumber=3;NumCh=NumFiltIn*5;
[lgraph,LayerNameOut_3] = H_RD_MOD(lgraph,FilterSize,NumFiltIn*5,...
    bnFlag,reluFlag,LayerNameIn3,LayerNumber,Siffix_layer,NumCh);
DepthConcat_layer_3 = depthConcatenationLayer(4,'Name',...
    ['depthCat_',num2str(LayerNumber),'_',Siffix_layer]);
lgraph = addLayers(lgraph,DepthConcat_layer_3);
% Connect LayerNameIn to DepthConcat_layer_1
lgraph = connectLayers(lgraph,LayerNameOut_3,...
    [DepthConcat_layer_3.Name,'/',DepthConcat_layer_3.InputNames{1}]);
lgraph = connectLayers(lgraph,LayerNameIn3,...
    [DepthConcat_layer_3.Name,'/',DepthConcat_layer_3.InputNames{2}]);
lgraph = connectLayers(lgraph,LayerNameIn2,...
    [DepthConcat_layer_3.Name,'/',DepthConcat_layer_3.InputNames{3}]);
lgraph = connectLayers(lgraph,LayerNameIn,...
    [DepthConcat_layer_3.Name,'/',DepthConcat_layer_3.InputNames{4}]);

% output layer name
LayerNameIn4 = [DepthConcat_layer_3.Name,'/',DepthConcat_layer_3.OutputNames{1}];
% Define covolution Layer
LayerNumber = 4;
layer_conv_1 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFiltOut,...
    ['conv_1_',Siffix_layer],bnFlag,reluFlag);
lgraph = addLayers(lgraph,layer_conv_1);
% Connect RD_MOD to Conv-Layer
lgraph = connectLayers(lgraph,LayerNameIn4,...
    [layer_conv_1(1).Name,'/',layer_conv_1(1).InputNames{1}]);
% output layer name
LayerNameIn5 = [layer_conv_1(end).Name,'/',layer_conv_1(end).OutputNames{1}];
% Define Final output
LayerNameOut = LayerNameIn5;
end
%% H Channel Attention Block
function [lgraph,LayerNameOut] = H_CAB(lgraph,LayerNameIn,Type,Siffix)
% Define CAB Module
Siffix_layer = ['CAB_',Siffix];
layer_avgPool = globalAveragePooling2dLayer('Name',['avgPool2D_',Siffix_layer]);
LayerNumber = 1;  FilterSize = 3; % NumFilt = 64;
layer_2_conv = groupedConvolution2dLayer(FilterSize,1,'channel-wise',...
    'Padding','same','Name',['conv_',num2str(LayerNumber),'_ch_',Siffix_layer]);
layer_2_relu = reluLayer("Name",['relu_',num2str(LayerNumber),'_ch_',Siffix_layer]);

LayerNumber = 2;  FilterSize = 3; % NumFilt = 64;

layer_3_conv = groupedConvolution2dLayer(FilterSize,1,'channel-wise',...
    'Padding','same','Name',['conv_',num2str(LayerNumber),'_ch_',Siffix_layer]);

layer_sigmoid = sigmoidLayer("Name",['sigmoid_',Siffix_layer]);

layer_mult = multiplicationLayer(2,'Name',['mult_',Siffix_layer]);

lgraph = addLayers(lgraph,...
    [layer_avgPool;layer_2_conv;layer_2_relu;layer_3_conv;layer_sigmoid;layer_mult]);
% Connect input to mutliplicaiton-layer
lgraph = connectLayers(lgraph,LayerNameIn,...
    [layer_mult.Name,'/',layer_mult.InputNames{2}]);
% Connect input to averagepooling
lgraph = connectLayers(lgraph,LayerNameIn,...
    [layer_avgPool.Name,'/',layer_avgPool.InputNames{1}]);
% output layer name
LayerNameOut = [layer_mult.Name,'/',layer_mult.OutputNames{1}];
end
%% H Pixel Attention Block
function [lgraph,LayerNameOut] = H_PAB(lgraph,LayerNameIn,Type,Siffix)
% Pixel attenton block
% Define covolution Layer and relu
bnFlag=0;reluFlag=1;
FilterSize=3; Siffix_layer = ['PAB_',Siffix];
LayerNumber = 1;NumFilt=64;
layer_conv_1 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,...
    Siffix_layer,bnFlag,reluFlag);
LayerNumber = 2;NumFilt=1;bnFlag=0;reluFlag=0;
layer_conv_2 = GenerateConvolutionModule(LayerNumber,FilterSize,NumFilt,...
    Siffix_layer,bnFlag,reluFlag);
layer_sigmoid = sigmoidLayer("Name",['sigmoid_',Siffix_layer]);
layer_mult = multiplicationLayer(2,'Name',['mult_',Siffix_layer]);
% Add layers to lgraph
lgraph = addLayers(lgraph,...
    [layer_conv_1;layer_conv_2;layer_sigmoid;layer_mult]);
% Connect input to mutliplicaiton-layer
lgraph = connectLayers(lgraph,LayerNameIn,...
    [layer_mult.Name,'/',layer_mult.InputNames{2}]);
% Connect input to layer_conv_1
lgraph = connectLayers(lgraph,LayerNameIn,...
    [layer_conv_1(1).Name,'/',layer_conv_1(1).InputNames{1}]);
% output layer name
LayerNameOut = [layer_mult.Name,'/',layer_mult.OutputNames{1}];
end

% ------------------------------------------------------------------------
function layer = GenerateDilatedConvolutionModule(...
                LayerNumber,FilterSize,NumFilt,SuffixLayer,bnFlag,...
                    reluFlag,DilationFactor)
% LayerNumber = LayerNumberT(1)
% FilterSize = 2;%NumFilt=16;
% SuffixLayer = 'ar';
if isempty(DilationFactor)
    DilationFactor = 1;
end
layer_conv = convolution2dLayer([FilterSize FilterSize], NumFilt, ...
    'Name', ['conv_',num2str(LayerNumber),SuffixLayer], ...
    'Padding', 'same','DilationFactor',DilationFactor.*[1,1],...
    'WeightsInitializer','narrow-normal');
layer_bn = [];
if bnFlag == 1
    layer_bn = batchNormalizationLayer('Name',...
        ['bano_',num2str(LayerNumber),SuffixLayer]);
end
layer_relu = [];
if reluFlag == 1
    layer_relu = reluLayer('Name',...
        ['relu_',num2str(LayerNumber),SuffixLayer]);
end
layer = [layer_conv;layer_bn;layer_relu];
end

% ------------------------------------------------------------------------
%% rotate Image according to theta and convert to patches
function imgOut = rotate2patch(Img,BlkSize,Angle_Img,opt)
%% imgOut = rotate2patch(Img,BlkSize,Angle_Img,opt)
% Convert to patch and also rotate
% opt.BlkStepRow : Requirment of block size in row
% opt.BlkStepCol : Requirment of block size in column
% BlkSize : Requirment of block size
% Img :  Input image
% Angle_Img : 
if isempty(BlkSize)
    imgOut = Img;
    disp('No rotation applied')
else
    imgOut = zeros([BlkSize(1)*BlkSize(2),50e3],"double");
    kk = 1;
    [rows_Img, colm_Img] = size(Img);
    if isempty(opt.BlkStepRow)||isempty(opt.BlkStepCol)
        row_step = ceil(BlkSize(1)/10);
        col_step = ceil(BlkSize(2)/10);
    else
        row_step = opt.BlkStepRow;
        col_step = opt.BlkStepCol;
    end
    
    for ii = 1:row_step:rows_Img-ceil(BlkSize(1)*(1+sind(mod(Angle_Img,180))))
        for jj = 1:col_step:colm_Img-ceil(BlkSize(2)*(1+sind(mod(Angle_Img,180))))
            im_sub = Img(ii:ii+ceil(BlkSize(1)*(1+sind(mod(Angle_Img,180))))-1,...
                jj:jj+ceil(BlkSize(2)*(1+sind(mod(Angle_Img,180))))-1);
            im_sub_Th = imrotate(im_sub,Angle_Img,"bicubic","loose");
            [row_th,col_th] = size(im_sub_Th);
            row_st = floor(row_th/2)-floor(BlkSize(1)/2)+1;
            col_st = floor(col_th/2)-floor(BlkSize(2)/2)+1;
            im_sub_Theta = ...
                im_sub_Th(row_st:row_st+BlkSize(1)-1,...
                col_st:col_st+BlkSize(2)-1);
            imgOut(:,kk) = im_sub_Theta(:);
            kk = kk + 1;
        end
    end
    imgOut = imgOut(:,1:kk-1);
end
end

% ------------------------------------------------------------------------


% ------------------------------------------------------------------------



% ------------------------------------------------------------------------


% ------------------------------------------------------------------------






