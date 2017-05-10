-- general libraries
require 'torch'
require 'paths'
require 'xlua'
require 'math'
require 'nn'
require 'logroll'
require 'gnuplot'

-- program requires
require 'utils'
require 'io-utils'
require 'set-utils'
require 'nn-utils'
require 'TrainOptimPP'
require 'settings'

-- initialize settings
settings = Settings();   

-- initialize logs
flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. '/train.log');
plog = logroll.print_logger();
log = logroll.combine(flog, plog);

--local newModel = {settings.inputSize,30,20, settings.outputSize}
--local newModel = {4, settings.outputSize}
local newModel = {299, settings.outputSize}
local ModelName = newModel[1];
for i = 2,table.getn(newModel) do ModelName = ModelName .. '_' .. newModel[i] end
local model;
local ModelName = "Conv2299_11";
-- initialize the network
local modelType = "load";
if (modelType == "convolve") then
 --model = buildConvolveModel(newModel);
  -- input layer
  model = nn.Sequential();
  model:add(nn.TemporalConvolution(1, 1, 30,15));   -- layer size
  model:add(getAF());   -- layer type
  --model:add(nn.TemporalMaxPooling(2));
  -- hidden layers
  --model:add(nn.TemporalConvolution(2, 2, 4, 2));   -- layer size
  --model:add(getAF());   -- layer type
  --model:add(nn.TemporalMaxPooling(2));
  s = model:forward(torch.randn(20000,settings.inputSize,1)):size()
  ssize= s[2]*s[3];
  model:add(nn.Reshape(s[1],ssize))
  
  -- output layer
  ll = nn.Linear(ssize, settings.outputSize);   -- layer size  
  ll.weight:zero();   -- default weights
  ll.bias:zero();     -- default bias
  model:add(ll);
  --model:add(nn.LogSoftMax());   -- output layer type  
elseif (modelType == "load") then
 model = torch.load(settings.outputFolder .. settings.modFolder .. "/" .. ModelName .. ".mod");
else
  error('Model: not supported');
  return
end

print(model)
-- DNN
local nBatch=50;
local batchSize=20000;
local nEpoch=2;
local learningRate=0.08;
local extraTest=0;
local maxIter = 200;
local iter = 0;
local MinError = 0.1

repeat
  local err = TrainOptimPPfull(ModelName, model, settings.listsTest, nBatch, batchSize, nEpoch, learningRate, extraTest);
  iter=iter+1;
until err < MinError or iter > maxIter
local err = TrainOptimPPfull(ModelName, model, settings.listsTest, 1, batchSize, nEpoch, learningRate, 1); 
print(err)

