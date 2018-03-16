require 'FillData'
require 'ReadDataset'
require 'TensorSaveLoad'
require 'TestModel'

    
function TrainingFromFile(model, listIn, listOut, selector, trainer) 
  local etime = sys.clock();    
  -- TRAINING
  model:training()
  
  local nFile = table.getn(listIn) 
  for noFile = 1,nFile do
    if Settings.DispFile then plog.info("Training from File: " .. noFile .. "/" .. nFile) end
    local dataIn = NactiData(Settings.ListFolder .. listIn[i]);
    local dataOut = NactiData(Settings.ListFolder .. listOut[i]);
    selector(model, dataIn, dataOut, trainer); 
  end
  log.info("Training " .. Settings.ModelName .. " completed in " .. sys.clock() - etime)
end

function Training(model, dataIn, dataOut, selector, trainer)  
  local etime = sys.clock();    
  -- TRAINING
  model:training()
  
--  local nFile = dataIn:size(1)
  local nFile = table.getn(dataIn)
  for noFile = 1,nFile do
    if Settings.DispFile then plog.info("Training from File: " .. noFile .. "/" .. nFile) end
--    selector(model, dataIn:select(1,noFile), dataOut:select(1,noFile), trainer); 
    selector(model, dataIn[noFile], dataOut[noFile], trainer); 
  end
  log.info("Training " .. Settings.ModelName .. " completed in " .. sys.clock() - etime)
end

function TrainingRaw(model, DataIn, DataOut)  
  local etime = sys.clock();    
  -- TRAINING
  model:training()
--  local nFile = DataIn:size(1)
  local nFile = table.getn(DataIn)
  for noFile = 1,nFile do
    if Settings.DispFile then plog.info("Training from File: " .. noFile .. "/" .. nFile) end
     -- prepare inputs & outputs tensors  
--      local dataIn=DataIn:select(1,noFile)
--      local dataOut=DataOut:select(1,noFile) 
      local dataIn=DataIn[noFile]
      local dataOut=DataOut[noFile]
      for batch = 1,Settings.BatchN do
        if Settings.DispBatch then plog.info("Training Batch: " .. batch .. "/" .. Settings.BatchN) end
        local tIn ={}
        table.insert(tIn,Settings.BatchSize)     
        local tOut ={}
        table.insert(tOut,Settings.BatchSize)     
        for i = 2,dataIn:dim() do
          table.insert(tIn,dataIn:size(i))
        end
        for i = 2,dataOut:dim() do
          table.insert(tOut,dataOut:size(i))
        end
        
        local inputs = torch.Tensor(torch.LongStorage(tIn))
        local outputs = torch.Tensor(torch.LongStorage(tOut))
        
        -- process batches
        for i = 1, Settings.BatchSize do      
          local dataM=torch.random(1,dataIn:size(1))
          inputs[i] = dataIn:select(1,dataM)
          outputs[i] = dataOut:select(1,dataM)
        end    
        TrainOptim(model, inputs, outputs); 
      end
    end
  log.info("Training " .. Settings.ModelName .. " completed in " .. sys.clock() - etime)
end

function TestModelFromFile(model, listIn, listOut, selector, tester) 
  local etime = sys.clock();    
  -- EVALUATION
  model:evaluate();  
  local err = 0
  local all = 0
  
  local nFile = table.getn(listIn) 
  for noFile = 1,nFile do
    if Settings.DispFile then plog.info("Testing from File: " .. noFile .. "/" .. nFile) end
    local dataIn = NactiData(Settings.ListFolder .. listIn[nFile]);
    local dataOut = NactiData(Settings.ListFolder .. listOut[nFile]);
    local f = function(model, dataIn, dataOut)
      local e,a = tester(model, dataIn, dataOut);
      err=err+e
      all=all+a
    end
    selector(model, dataIn, dataOut, f);
  end
  -- logs & export model
  err = 100*err/all
  log.info("Testing " .. Settings.ModelName .. " completed in " .. sys.clock() - etime)
  return err
end

function TestModel(model, dataIn, dataOut, selector, tester)
  local etime = sys.clock();    
  -- EVALUATION
  model:evaluate();  
  local err = 0
  local all = 0
  local nFile = dataIn:size(1)
  for noFile = 1,nFile do
    if Settings.DispFile then plog.info("Testing from File: " .. noFile .. "/" .. nFile) end    
    local f = function(model, dataIn, dataOut)
      local a,e = tester(model, dataIn, dataOut);
      err=err+e
      all=all+a
    end
    selector(model, dataIn:select(1,noFile), dataOut:select(1,noFile), f);
  end
  -- logs & export model
  err = 100*err/all
  log.info("Testing " .. Settings.ModelName .. " completed in " .. sys.clock() - etime)
  return err
end

function TestModelX(model, dataIn, dataOut, selector, tester)
  local etime = sys.clock();    
  -- EVALUATION
  model:evaluate();  
  local err = torch.Tensor(Settings.OutputSize,Settings.OutputSize):fill(0)
  local all = torch.Tensor(Settings.OutputSize):fill(0)
  local errn = torch.Tensor(Settings.OutputSize):fill(0)
--  local nFile = dataIn:size(1)
  local nFile = table.getn(dataIn) 
  for noFile = 1,nFile do
    if Settings.DispFile then plog.info("Testing from File: " .. noFile .. "/" .. nFile) end    
    local f = function(model, dataIn, dataOut)
      local a,e = tester(model, dataIn, dataOut);
      err=err+e
      all=all+a
    end
    selector(model, dataIn[noFile], dataOut[noFile], f);
  end
  -- logs & export model
  for i=1,err:size(1) do
    if all[i]==0 then
      err[i]=0
    else
      --err[i]=100*err[i]:sum()/all[i]
      err[i]=100*err[i]/all[i]
    end
  end
  --err = 100*err/all
  log.info("Testing " .. Settings.ModelName .. " completed in " .. sys.clock() - etime)
  return err
end