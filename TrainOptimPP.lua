require 'OptimPP'
require 'ReadDataset'
require 'TestNN'
require 'MyPCA'

function TrainOptimPP(modelN,modelC)
  -- criterion 
  local criterion = nn.CrossEntropyCriterion()

  -- TRAINING
  local etime = sys.clock();  

  -- mode training
  modelC:training()
  -- log
  
  -- training per batches
  local noBatch=1
  local input, output = ReadDataset(settings.lists[noBatch]);
  local nBatch = 20
  for n = 1,nBatch do
    plog.info("Training Batch: " .. n .. "/" .. nBatch);
    -- prepare inputs & outputs tensors    
    local batchSize = 30000;
    local inputs = torch.Tensor(batchSize,input:size(3));
    local targets = torch.Tensor(batchSize);
    
    -- process batches
    for i = 1, batchSize,1 do      
      --for j = 1, inputs:size(2), 1 do
      -- pick frame 
      local dataX=torch.random(input:size(1))
      local dataY=torch.random(input:size(2))
      --local input = datas[{{dataX},{dataY},{}}]:float():resize(datas:size(3));
      inputs[i] = input:select(1,dataX):select(1,dataY)
      targets[i] = output[dataX][dataY];
    end
    
    OptimPP(modelC, criterion, 5, 0.08, inputs, targets);
  end
  
  -- logs & export model
    plog.info("Saving: " .. modelN);
    torch.save(settings.outputFolder .. settings.modFolder .. "/" .. modelN .. ".mod", modelC);
    exportModel(settings.outputFolder .. settings.modFolder .. "/" .. modelN .. ".mod", settings.outputFolder .. settings.modFolder .. "/" .. modelN .. ".nnet");

  
  -- EVALUATION
  modelC:evaluate();

  local errorBatch=-1;
  for i = 1,table.getn(settings.listsTest), 1 do            
    local input, output = ReadDataset(settings.listsTest[i]);
    local err = TestNN(modelC, input, output, 10);
    log.info("Model " .. modelN .. " - err = " .. err); 
    errorBatch = err;
  end
  --[[
  for i = 1,table.getn(settings.listsTestX), 1 do            
    local input, output = ReadDataset(settings.listsTestX[i]);
    local err = TestNN(modelC, input, output, 10);
    log.info("Set " .. settings.listsTestX[i] .. " - err = " .. err);  
  end
    ]]
  log.info("Train " .. modelN .. " completed in " .. sys.clock() - etime);  
  return errorBatch
end

function TrainOptimPPfull(modelN,modelC, dataList, nBatch, batchSize, nEpoch, learningRate, extraTest)
    -- criterion 
  local criterion = nn.CrossEntropyCriterion()

  -- TRAINING
  local etime = sys.clock();  
  
  -- mode training
  modelC:training()
  -- log
  -- training per batches
  local nFile = table.getn(dataList)
  for noFile = 1,nFile do
    plog.info("Training from File: " .. noFile .. "/" .. nFile);
    local input, output = ReadDataset(dataList[noFile]);
    --local nBatch = 20
    for n = 1,nBatch do
      plog.info("Training Batch: " .. n .. "/" .. nBatch);
      -- prepare inputs & outputs tensors    
      --local batchSize = 30000;
      local inputs = torch.Tensor(batchSize,input:size(3));
      local targets = torch.Tensor(batchSize);
      -- process batches
      for i = 1, batchSize,1 do      
        --for j = 1, inputs:size(2), 1 do
        -- pick frame 
        local dataX=torch.random(input:size(1))
        local dataY=torch.random(input:size(2))
        --local input = datas[{{dataX},{dataY},{}}]:float():resize(datas:size(3));
        inputs[i] = input:select(1,dataX):select(1,dataY)
        targets[i] = output[dataX][dataY];
      end
      
      OptimPP(modelC, criterion, nEpoch, 0.08, inputs, targets); 
    end
  end
  -- logs & export model
    plog.info("Saving: " .. modelN);
    torch.save(settings.outputFolder .. settings.modFolder .. "/" .. modelN .. ".mod", modelC);
    --exportModel(settings.outputFolder .. settings.modFolder .. "/" .. modelN .. ".mod", settings.outputFolder .. settings.modFolder .. "/" .. modelN .. ".nnet");

  
  -- EVALUATION
  modelC:evaluate();

  local errorBatch=-1;
  for i = 1,table.getn(settings.listsTest), 1 do            
    local input, output = ReadDataset(settings.listsTest[i]);
    local err = TestNN(modelC, input, output, 10);
    log.info("Model " .. modelN .. " - err = " .. err); 
    errorBatch = err;
  end
  if extraTest == 1 then
    for i = 1,table.getn(settings.listsTestX), 1 do            
      local input, output = ReadDataset(settings.listsTestX[i]);
      local err = TestNN(modelC, input, output, 10);
      log.info("Set " .. settings.listsTestX[i] .. " - err = " .. err);  
    end
  end
  
  log.info("Train " .. modelN .. " completed in " .. sys.clock() - etime);  
  return errorBatch
end

function TrainOptimPPfullPCA(modelN,modelC, dataList, nBatch, batchSize, nEpoch, learningRate, extraTest,N)
    -- criterion 
  local criterion = nn.CrossEntropyCriterion()

  -- TRAINING
  local etime = sys.clock();  
  
  -- mode training
  modelC:training()
  -- log
  -- training per batches
  local nFile = table.getn(dataList)
  for noFile = 1,nFile do
    plog.info("Training from File: " .. noFile .. "/" .. nFile);
    local input, output = ReadDataset(dataList[noFile]);
      local s = input:size();
      input = input:resize(s[1]*s[2],s[3]);
      --X = MyPCAbatch(input2,N,10000);
      input = MyPCAbatch(input,N,10000):resize(s[1],s[2],s[3]);
    --local nBatch = 20
    for n = 1,nBatch do
      plog.info("Training Batch: " .. n .. "/" .. nBatch);
      -- prepare inputs & outputs tensors    
      --local batchSize = 30000;
      local inputs = torch.Tensor(batchSize,input:size(3));
      local targets = torch.Tensor(batchSize);
      -- process batches
      for i = 1, batchSize,1 do      
        --for j = 1, inputs:size(2), 1 do
        -- pick frame 
        local dataX=torch.random(input:size(1))
        local dataY=torch.random(input:size(2))
        --local input = datas[{{dataX},{dataY},{}}]:float():resize(datas:size(3));
        inputs[i] = input:select(1,dataX):select(1,dataY)
        targets[i] = output[dataX][dataY];
      end
      
      OptimPP(modelC, criterion, nEpoch, 0.08, inputs, targets);
    end
  end
  -- logs & export model
    plog.info("Saving: " .. modelN);
    torch.save(settings.outputFolder .. settings.modFolder .. "/" .. modelN .. ".mod", modelC);
    --exportModel(settings.outputFolder .. settings.modFolder .. "/" .. modelN .. ".mod", settings.outputFolder .. settings.modFolder .. "/" .. modelN .. ".nnet");

  
  -- EVALUATION
  modelC:evaluate();

  local errorBatch=-1;
  for i = 1,table.getn(settings.listsTest), 1 do            
    local input, output = ReadDataset(settings.listsTest[i]);
      local s = input:size();
      input = input:resize(s[1]*s[2],s[3]);
      input = MyPCAbatch(input,N,10000):resize(s[1],s[2],s[3]);
    local err = TestNN(modelC, input, output, 10);
    log.info("Model " .. modelN .. " - err = " .. err); 
    errorBatch = err;
  end
  if extraTest == 1 then
    for i = 1,table.getn(settings.listsTestX), 1 do            
      local input, output = ReadDataset(settings.listsTestX[i]);
      local s = input:size();
      input = input:resize(s[1]*s[2],s[3]);
      input = MyPCAbatch(input,N,10000):resize(s[1],s[2],s[3]);
      local err = TestNN(modelC, input, output, 10);
      log.info("Set " .. settings.listsTestX[i] .. " - err = " .. err);  
    end
  end
  
  log.info("Train " .. modelN .. " completed in " .. sys.clock() - etime);  
  return errorBatch
end