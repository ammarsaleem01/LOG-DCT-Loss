classdef MultiplicationLayer < nnet.layer.Layer

    properties
        % (Optional) Layer properties.

        % Layer properties go here.
        MultNum
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters.

        % Layer learnable parameters go here.
    end
    
    methods
        function layer = MultiplicationLayer(numMult,name)
            % (Optional) Create a myLayer.
            layer.Name = name;
            % Multiplication Number
            layer.MultNum = numMult;
            layer.Description = "Multiplication of input to a number";
        end
        
        function Z = predict(layer,X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X - Input data
            % Outputs:
            %         Z - Outputs of layer forward function
            
            % Layer forward function for prediction goes here.
            m = layer.MultNum;
            Z = m.*X;
        end

%         function [Z1, …, Zm, memory] = forward(layer, X1, …, Xn)
%             % (Optional) Forward input data through the layer at training
%             % time and output the result and a memory value.
%             %
%             % Inputs:
%             %         layer       - Layer to forward propagate through
%             %         X1, ..., Xn - Input data
%             % Outputs:
%             %         Z1, ..., Zm - Outputs of layer forward function
%             %         memory      - Memory value for custom backward propagation
% 
%             % Layer forward function for training goes here.
%         end

        function dLdX = backward(layer, ~, ~, dLdZ, ~)
            % (Optional) Backward propagate the derivative of the loss  
            % function through the layer.
            %
            % Inputs:
            %         layer             - Layer to backward propagate through
            %         X       - Input data
            %         Z       - Outputs of layer forward function            
            %         dLdZ    - Gradients propagated from the next layers
            %         memory  - Memory value from forward function
            % Outputs:
            %         dLdX    - Derivatives of the loss with respect to the
            %                             inputs
            
            % Layer backward function goes here.
            m = layer.MultNum;
            dLdX = m.*dLdZ;
        end
    end
end