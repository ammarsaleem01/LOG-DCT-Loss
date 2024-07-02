classdef DCT_LOG_RegressionLayer < nnet.layer.RegressionLayer
    % Example custom regression layer with mean-absolute-error loss.
    
    methods
        function layer = DCT_LOG_RegressionLayer(name)
            % layer = maeRegressionLayer(name) creates a
            % mean-absolute-error regression layer and specifies the layer
            % name.
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = 'DCT Log L2 Regression';
        end
        
        function loss = forwardLoss(~, Y, T)
            % the predictions Y and the training targets T.
            % Cacualte the size of input
            if isgpuarray(Y)
                sY = gather(Y);
            else
                sY = Y;
            end
            
            if isgpuarray(T)
                sT = gather(T);
            else
                sT = T;
            end
            R1 = size(Y,1);
            R2 = size(Y,2);
            R3 = size(Y,3);
            R4 = size(Y,4);
            sloss = 0;
            oneG = 1;
            for ii = oneG:R4
                for jj = oneG:R3
                    dctY = dct2(sY(oneG:R1,oneG:R2,jj,ii),[R1,R2]);
                    dctT = dct2(sT(oneG:R1,oneG:R2,jj,ii),[R1,R2]);
                    a = min([min(dctY(:)),min(dctT(:))])-100;
%                     sloss_ii_jj = sum((log(dctY-a)-log(dctT-a)).^2,[1,2])/(R1*R2);
                    sloss_ii_jj = sum((log(dctY-a)-log(dctT-a)).^2,[1,2]);
                    sloss = sloss + sloss_ii_jj;
                end
            end
            loss = cast(sloss,'like',Y);
        end
        
        function dLdY = backwardLoss(~, Y, T)
            % dLdY = backwardLoss(layer, Y, T) returns the derivatives of
            % the MAE loss with respect to the predictions Y.
            R1 = size(Y,1);
            R2 = size(Y,2);
            R3 = size(Y,3);
            R4 = size(Y,4);
            if isgpuarray(Y)
                sY = gather(Y);
            else
                sY = Y;
            end
            if isgpuarray(T)
                sT = gather(T);
            else
                sT = T;
            end
            sdLdY = zeros(size(Y));
            for ii = 1:R4
                for jj = 1:R3
                    dctY = dct2(sY(1:R1,1:R2,jj,ii),[R1,R2]);
                    dctT = dct2(sT(1:R1,1:R2,jj,ii),[R1,R2]);
                    a = min([min(dctY(:)),min(dctT(:))])-100;
                    sdLdY(1:R1,1:R2,jj,ii) = ...
                        (idct2(2*(log(dctY-a)-log(dctT-a))./(dctY-a),[R1,R2]));
%                         (idct2(2*(log(dctY-a)-log(dctT-a))./(dctY-a),[R1,R2]))/(R1*R2);
                end
            end
            dLdY = cast(sdLdY,'like',Y);
        end
    end
end