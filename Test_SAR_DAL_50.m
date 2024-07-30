%% Define variable that can be changed
N2XFlag = 0; % 0: Noise-2-Noise, 1:Noise-2-Clean
SARcene_index = 1; % Select SAR scene index
% 1: "KapikuleTurkeySE",
% 2: "CordobaSpainSE",
% 3: "RichmondUSASE",
% 4: "ShahrudIranSE"
% ------------------------------------------------------------------------
SARsceneNameTestTotal = ["KapikuleTurkeySE","CordobaSpainSE","RichmondUSASE","ShahrudIranSE"];
DispFlag = 1; % Display Figures of specifie SAR Scene
VerboseFlag = 1; % Writes Calcualted parameters on command window
%% Load Trained File
addpath(genpath(fullfile(cd,'Utility')))
% 0:Nochange, 1:0-to-1
RescaleFlag = 1;
DataAvailability = 50; % Experiments for only 50% Data Availability
NoiseLevel = 0.03; % 0.03
switch N2XFlag
    case 0
        N2Xtext = 'N2N';
    case 1
        N2Xtext = 'N2C';
    otherwise
        error('N2XFlag = 0,1');
end

%% Define Test SAR Scene
SARsceneNameTest = SARsceneNameTestTotal(SARcene_index);
ImgRefTest = ReadDataSet(SARsceneNameTest);
ImgRefTest_Rescaled = RescaleDataSet(ImgRefTest,ImgRefTest,RescaleFlag);
% Validation Options for Forward model
N1_Test = size(ImgRefTest,1);
N2_Test = size(ImgRefTest,2);
N3_Test = 2*floor(sqrt((N1_Test).^2* DataAvailability/100)/2);
N4_Test = 2*floor(sqrt((N2_Test).^2* DataAvailability/100)/2);

opts_Test.N1 = N1_Test;
opts_Test.N2 = N2_Test;

opts_Test.Nlim1 = N3_Test;
opts_Test.Nlim2 = N4_Test;

opts_Test.RescaleFlag = RescaleFlag;
opts_Test.DataAvailability = DataAvailability;
% Options for Phase domain
opts_Test.PhaseType = "UniformRandom";
opts_Test.type = "valGaus";
opts_Test.DAL = DataAvailability/100;
opts_Test.NoiseSigma = NoiseLevel;
opts_Test.Masking = "box";
M_Valid = MaskOpt(opts_Test.DAL,N1_Test,N2_Test,opts_Test.Masking);
opts_Test.Mask = M_Valid;

nH_Valid = @(z)H(z,opts_Test); % The forward Fourier sampling operator
nHH_Valid=@(z)HH(z,opts_Test); % The backward Fourier sampling operator

[Patch_Img_In_Test,Patch_ImgOut_Test] = ConvertDataValid(...
    ImgRefTest_Rescaled,nH_Valid,nHH_Valid,opts_Test);

%% Read Network
NetRow = 0;
for NetCoice_i = [0 1 4 5]+20% [20;30;50;60]
    NetRow = NetRow + 1;
    NetChoice = NetCoice_i;%[0 1 4 5]+30% [20;30;50;60]
    reset(gpuDevice())
    if floor(NetCoice_i/10)==2
        MiniBatchSize = 40;
    end
    if floor(NetCoice_i/10)==5
        MiniBatchSize = 15;
    end
    NetName = DefineNetName(NetChoice);
    FileNameTrain = sprintf([NetName,'_256x256_WonsanNorthKoreaSE_SAR_',...
        N2Xtext,'_NL_0_DAL_50_AngleAug_Rescale_0_1_Net_',num2str(NetChoice),...
        '_mBz_',num2str(MiniBatchSize),'_mEp_10.mat']);
    load(fullfile(cd,'TrainedNetworks',FileNameTrain));
    %% Calculate Performance on Valdation Data
    YPred_XTest = activations(trainedNet,...
        Patch_Img_In_Test,'AdditionResdiual','OutputAs','channels',...
        'ExecutionEnvironment','multi-gpu');
    %% Calculate Parameters
    [snrYPred_XV,psnrYPred_XV,ssimYPred_XV] = ...
        P_SNR_SSIM_Calculate(YPred_XTest,Patch_ImgOut_Test);

    [snrXTest,psnrXTest,ssimXTest] = ...
        P_SNR_SSIM_Calculate(Patch_Img_In_Test,Patch_ImgOut_Test);
    if DispFlag == 1
        figure(1);imagesc(Patch_ImgOut_Test);colormap gray;colorbar;drawnow;
        figure(2);imagesc(Patch_Img_In_Test);colormap gray;colorbar;drawnow;
        figure(NetRow+2);imagesc(YPred_XTest);colormap gray;colorbar;drawnow;
    end
    DataPerf = [snrYPred_XV,psnrYPred_XV,ssimYPred_XV,snrXTest,psnrXTest,ssimXTest];
    if VerboseFlag == 1
        disp(['snrPred',' , ','psnrPred',' , ','ssimPred',' , ',...
            'snrConvIn',' , ','psnrConvIn',' , ','ssimConvIn']);
        disp(mean(DataPerf,1))
    end
end

%% =======================================================================
%% HELPING FUNCTIONS
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
N1 = 32*floor(N1/32);% 2*floor(sqrt((N1).^2)/2);
N2 = 32*floor(N2/32);%2*floor(sqrt((N2).^2)/2);
ImgRef = imcrop(ImgRef,[0,0,N2,N1]);
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
function [ImgConv,ImgFRef] = ForwardModelPE(ImgRef,nH,nHH,opts)
ImgFRef_Rescale = RescaleDataSet(ImgRef,ImgRef,opts.RescaleFlag);
ImgFRef = addphase(ImgFRef_Rescale);                                            % Adding complex valued uniform Random phase to magnitude of RF
g = nH(ImgFRef);                                                           % Clean and full DAL Phase histories (PH)
optsNoise.type = opts.type;                                                % Noise distribution type
optsNoise.NoiseSigma = opts.NoiseSigma;                                    % Setting NL to option
optsNoise.PhaseType = opts.PhaseType;                                      % "UniformRandom","GausRandom","Sine","Poly"
gn = AddNoise(g,optsNoise);                                                 % Add Noise to PH
f_conv = reshape(nHH(gn),opts.N1,opts.N2);                                  % Conventional Reconstruction (CR)
f_conv_abs = abs(f_conv);
f_conv_ang = atan2(imag(f_conv),real(f_conv));
f_conv_abs_resc = RescaleDataSet(f_conv_abs,ImgFRef_Rescale,opts.RescaleFlag);
ImgConv = f_conv_abs_resc.*exp(1i*f_conv_ang);
end
% ------------------------------------------------------------------------
%% Read data from SAR image and convert to patches to make dataset with Forwared model
function [Img_In,ImgOut] = ConvertDataValid(ImgRef,nH,nHH,opts)
[ImgConv_1,ImgFRef_1] = ForwardModelPE(ImgRef,nH,nHH,opts);
ImgOut = abs(ImgFRef_1);
Img_In = abs(ImgConv_1);
Img_In = cast(Img_In,'like',ImgRef);
ImgOut = cast(ImgOut,'like',ImgRef);
end
% ------------------------------------------------------------------------
%% Define all models
function [NetName] = DefineNetName(NetChoice)
switch NetChoice
    case 20
        NetName = 'DnCNN_Net';
    case 21
        NetName = 'DnCNN_MAE_Net';
    case 24
        NetName = 'DnCNN_DCT_Net';
    case 25
        NetName = 'DnCNN_DCT_LOG_Net';
    case 30
        NetName = 'AlternateFiltersWithSkip_Net';
    case 31
        NetName = 'AlternateFiltersWithSkip_MAE_Net';
    case 34
        NetName = 'AlternateFiltersWithSkip_DCT_Net';
    case 35
        NetName = 'AlternateFiltersWithSkip_DCT_LOG_Net';
    case 50
        NetName = 'MDRUnet_Net';
    case 51
        NetName = 'MDRUnet_MAE_Net';
    case 54
        NetName = 'MDRUnet_DCT_Net';
    case 55
        NetName = 'MDRUnet_DCT_LOG_Net';
    case 60
        NetName = 'MRDDA_Net';
    case 61
        NetName = 'MRDDA_MAE_Net';
    case 64
        NetName = 'MRDDA_DCT_Net';
    case 65
        NetName = 'MRDDA_DCT_LOG_Net';
    otherwise
        error('NetChoice does not exist')
end
disp(['Net Name = ',NetName])
end
% ------------------------------------------------------------------------
%% SNR, PSNR, SSIM Calculate
function [snrXlog,psnrXlog,ssimX] = P_SNR_SSIM_Calculate(Xin,Xref)
% Calculate SNR and PSNR
snrXlog = zeros(size(Xin,4),1);
psnrXlog = zeros(size(Xin,4),1);
ssimX = zeros(size(Xin,4),1);
for ii = 1:size(Xin,4)
    Xref_ii = Xref(:,:,:,ii);
    Xin_ii = Xin(:,:,:,ii);
    Xref_ii_M = (Xref_ii-min(Xref_ii,[],[1,2]))./(max(Xref_ii,[],[1,2])-min(Xref_ii,[],[1,2]));
    A = [Xin_ii(:),ones(size(Xin_ii(:)),'like',Xin_ii)]\Xref_ii_M(:);
    Xin_ii_M = Xin_ii*A(1)+A(2);
    mseX_Scaled = mean((Xin_ii_M(:)-Xref_ii_M(:)).^2);
    snrX_Scaled = mean(((Xin_ii_M(:))).^2)/mseX_Scaled;
    psnrX_Scaled = (max(Xref_ii_M(:)).^2)/mseX_Scaled;
    [ssimX_Scaled, ~] = ssim(Xin_ii_M(:),Xref_ii_M(:));
    snrXlog(ii) = 10*log10((snrX_Scaled));
    psnrXlog(ii) = 10*log10((psnrX_Scaled));
    ssimX(ii) = ssimX_Scaled;
end
end
% ------------------------------------------------------------------------
%% Masking function
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

try
    M = parameters.Mask;
    x=reshape(x,N1,N2);
    y=M.*fft2(x,N1,N2);
catch
    disp('Masking did not work')
    x=reshape(x,N1,N2);
    ytemp=fft2(x,N1,N2);
    y=[ytemp(1:N3/2,1:N4/2) ytemp(1:N3/2,N2-N4/2+1:N2);ytemp(N1-N3/2+1:N1,1:N4/2) ytemp(N1-N3/2+1:N1,N2-N4/2+1:N2)];
end 
end
% ------------------------------------------------------------------------
%% Inverse Model
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
try
    y=reshape(y,N1,N2);
    M = parameters.Mask;
    x = ifft2(y.*M);
    % x=x(:);
catch
    disp('Masking did not work')
    y=reshape(y,N3,N4);
    ytemp=zeros(N1,N2);
    ytemp(1:N3/2,1:N4/2)=y(1:N3/2,1:N4/2);
    ytemp(1:N3/2,N2-N4/2+1:N2)=y(1:N3/2,N4/2+1:N4);
    ytemp(N1-N3/2+1:N1,1:N4/2)=y(N3/2+1:N3,1:N4/2);
    ytemp(N1-N3/2+1:N1,N2-N4/2+1:N2)=y(N3/2+1:N3,N4/2+1:N4);
    % x=ifft2(ytemp)*N1*N2;
    x=ifft2(ytemp);
    % x=x(:) / (N1*N2);
    % x=x / (N1*N2);
end
end
% ------------------------------------------------------------------------
%% Add phase
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
%% Add Noise
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

































