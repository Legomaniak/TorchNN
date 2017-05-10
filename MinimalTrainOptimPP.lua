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
settings = SettingsMulty();   

-- initialize logs
flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. '/train.log');
plog = logroll.print_logger();
log = logroll.combine(flog, plog);

startModel = {0, 0, 0}
Liter = 10;
Miter = 10;
Niter = 10;
Lkoef = 5;
Mkoef = 5;
Nkoef = 5;
ErrorsL = torch.Tensor(Liter)
ErrorsLM = torch.Tensor(Liter,Miter)
Errors = torch.Tensor(Liter,Miter,Niter)

nBatch=200;
batchSize=20000;
nEpoch=2;
learningRate=0.08;
extraTest=0;
maxIter = 5;
iter = 0;

for l = 1,Liter do
  local newModel = {settings.inputSize, startModel[1]+l*Lkoef, settings.outputSize}
      local ModelName = '';
      for i = 1,table.getn(newModel) do ModelName = ModelName .. '_' .. newModel[i] end
      local model;
      -- initialize the network
      local modelType = "classic";
      if (modelType == "classic") then
       model = buildFFModelNew(newModel);
      elseif (settings.model == "convolve") then
       model = buildConvolveModel(newModel);
      else
        error('Model: not supported');
      end
      
      --print(model)
      -- DNN

      --ErrorsL[l] = TrainOptimPP(ModelName,model)        
      ErrorsL[l] = TrainOptimPPfull(ModelName, model, settings.listsTest, nBatch, batchSize, nEpoch, learningRate, extraTest)
      if l%2==0 then
        torch.save(settings.outputFolder .. settings.statsFolder .. "/" .. 'MinimalErrorL' .. ".err", ErrorsL);
      else
        torch.save(settings.outputFolder .. settings.statsFolder .. "/" .. 'MinimalErrorL2' .. ".err", ErrorsL);
      end
      
  for m = 1,Miter do
    local newModel = {settings.inputSize, startModel[1]+l*Lkoef, startModel[2]+m*Mkoef, settings.outputSize}
      local ModelName = '';
      for i = 1,table.getn(newModel) do ModelName = ModelName .. '_' .. newModel[i] end
      local model;
      -- initialize the network
      local modelType = "classic";
      if (modelType == "classic") then
       model = buildFFModelNew(newModel);
      elseif (settings.model == "convolve") then
       model = buildConvolveModel(newModel);
      else
        error('Model: not supported');
      end
      
      --print(model)
      -- DNN

      --ErrorsLM[l][m] = TrainOptimPP(ModelName,model)        
      ErrorsLM[l][m] = TrainOptimPPfull(ModelName, model, settings.listsTest, nBatch, batchSize, nEpoch, learningRate, extraTest)
      if m%2==0 then
        torch.save(settings.outputFolder .. settings.statsFolder .. "/" .. 'MinimalErrorLM' .. ".err", ErrorsLM);
      else
        torch.save(settings.outputFolder .. settings.statsFolder .. "/" .. 'MinimalErrorLM2' .. ".err", ErrorsLM);
      end
      
    for n = 1,Niter do
      local newModel = {settings.inputSize, startModel[1]+l*Lkoef, startModel[2]+m*Mkoef, startModel[3]+n*Nkoef, settings.outputSize}
      local ModelName = '';
      for i = 1,table.getn(newModel) do ModelName = ModelName .. '_' .. newModel[i] end
      local model;
      -- initialize the network
      local modelType = "classic";
      if (modelType == "classic") then
       model = buildFFModelNew(newModel);
      elseif (settings.model == "convolve") then
       model = buildConvolveModel(newModel);
      else
        error('Model: not supported');
      end
      
      --print(model)
      -- DNN

      --Errors[l][m][n] = TrainOptimPP(ModelName,model)        
      Errors[l][m][n] = TrainOptimPPfull(ModelName, model, settings.listsTest, nBatch, batchSize, nEpoch, learningRate, extraTest)
      if n%2==0 then
        torch.save(settings.outputFolder .. settings.statsFolder .. "/" .. 'MinimalError' .. ".err", Errors);
      else
        torch.save(settings.outputFolder .. settings.statsFolder .. "/" .. 'MinimalError2' .. ".err", Errors);
      end
    end
  end
end

