classdef maeRegressionLayer < nnet.layer.RegressionLayer
    % Example custom regression layer with mean-absolute-error loss.
    
    methods
        function layer = maeRegressionLayer(name)
            % layer = maeRegressionLayer(name) creates a
            % mean-absolute-error regression layer and specifies the layer
            % name.
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = 'Mean absolute error';
        end
        
        function loss = forwardLoss(~, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the MAE loss between
            % the predictions Y and the training targets T.
            
            % Calculate MAE.
            Rc = Y(:,:,:,1);
            R = cast(length(Rc(:)),'like',Y);
            meanAbsoluteError = cast(sum(abs(Y-T),[1 2 3])/R,'like',Y);
            
            % Take mean over mini-batch.
            N = cast(size(Y,4),'like',Y);
            loss = (sum(meanAbsoluteError(:))/N);
        end
        
        function dLdY = backwardLoss(~, Y, T)
            % dLdY = backwardLoss(layer, Y, T) returns the derivatives of
            % the MAE loss with respect to the predictions Y.
            Rc = Y(:,:,:,1);
            R = cast(length(Rc(:)),'like',Y);
            N = cast(size(Y,4),'like',Y);
            T = cast(T,'like',Y);
            dLdY = (sign(Y-T)/(N*R));
        end
    end
end