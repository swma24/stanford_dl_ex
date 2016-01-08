function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.

%%% YOUR CODE HERE %%%
% pooledDim = size(pooledFeatures, 1);
poolFilter = (1/poolDim^2) * ones(poolDim);
for imageNum = 1:numImages
  for filterNum = 1:numFilters      
  cf = squeeze(convolvedFeatures(:,:,filterNum,imageNum));
%   for rowIdx = 1:pooledDim
%       rowStart = poolDim*(rowIdx-1) + 1;
%       rowEnd = rowStart + poolDim - 1;
%       for colIdx = 1:pooledDim
%           colStart = poolDim*(colIdx-1) + 1;
%           colEnd = colStart + poolDim - 1;
%           pf = squeeze(cf(rowStart:rowEnd,colStart:colEnd));
%           pv = sum(pf(:)) / poolDim^2;
%           pooledFeatures(rowIdx,colIdx,filterNum,imageNum) = pv;
%       end
%   end
    pooledFeaturesConvolved = conv2(cf,poolFilter,'valid');
    pooledFeatures(:,:,filterNum,imageNum) = pooledFeaturesConvolved(1:poolDim:end,1:poolDim:end);
  end
end

end

