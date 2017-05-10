function ModelConvTrainPP(modelN,modelC)
-- DNN

-- criterion 
--criterion = nn.ClassNLLCriterion();
--criterion = nn.MSECriterion();
criterion = nn.CrossEntropyCriterion()

errors = torch.Tensor(settings.noEpochs)
for epoch = settings.startEpoch + 1, settings.noEpochs, 1 do  
  -- TRAINING
  --settings.gnuploting = 0;
  -- timer per epoch - start
  local etime = sys.clock();  

  -- mode training
  modelC:training()
  -- log
  log.info("Training epoch: " .. epoch .. "/" .. settings.noEpochs);
  
  -- training per batches
  --noBatchs = table.getn(settings.lists)
  --for noBatch = 1,noBatchs, 1 do
  --plog.info("Training Batch: " .. noBatch .. "/" .. noBatchs);
  noBatch=1
  dataset = Dataset(settings.lists[noBatch]);
  for n = 1,20, 1 do
    plog.info("Training Batch: " .. n .. "/" .. 20);
    -- prepare inputs & outputs tensors    
    datas = dataset.nSamplesList
    dataOutput = dataset.nSamplesListOutput
    batchSize = 100;
    inputs = torch.Tensor(batchSize,settings.inputSize,settings.SizeX,settings.SizeY);
    targets = torch.Tensor(batchSize);
    
    -- process batches
    for i = 1, batchSize,1 do      
      --for j = 1, inputs:size(2), 1 do
      -- pick frame 
      local dataX=torch.random(datas:size(1)-settings.SizeX)+settings.SizeX/2;
      local dataY=settings.SizeY/2+1;--torch.random(datas:size(2)-settings.SizeY)+settings.SizeY/2;
      --local input = datas[{{dataX},{dataY},{}}]:float():resize(datas:size(3));
      local input = datas[{{dataX-settings.SizeX/2,dataX+settings.SizeX/2-1},{dataY-settings.SizeY/2,dataY+settings.SizeY/2-1},{}}]:transpose(1,3)
      targets[i] = dataOutput[dataX][dataY];
      
      --local min = torch.min(input);
      --input = input - min;
      
      --local max = torch.max(input);
      --input = input / max;
      inputs[i]= input;
    end

    -- forward propagation
    prediction = modelC:forward(inputs)
    criterion:forward(prediction, targets);

    -- zero the accumulation of the gradients
    modelC:zeroGradParameters();

    -- back propagation
    modelC:backward(inputs, criterion:backward(prediction, targets));
    
    -- update parameters
    if (settings.lrDecayActive == 1) then 
      learningRate = settings.learningRate / (1 + (epoch - 1) * settings.lrDecay); 
      modelC:updateParameters(learningRate);
    else 
      modelC:updateParameters(settings.learningRate);  
    end 
  end

  
  -- logs & export model
  if (settings.saveEpoch == 1) then
    plog.info("Saving epoch: " .. epoch .. "/" .. settings.noEpochs);
    torch.save(settings.outputFolder .. settings.modFolder .. "/" .. epoch .. ".mod", modelC);
  end
  if (settings.exportNNET == 1) then
    exportModel(settings.outputFolder .. settings.modFolder .. "/" .. epoch .. ".mod", settings.outputFolder .. settings.modFolder .. "/" .. epoch .. ".nnet");
  end
  log.info("Epoch: " .. epoch .. "/".. settings.noEpochs .. " completed in " .. sys.clock() - etime);  
  
  -- EVALUATION
  
  --settings.gnuploting = 1;
  -- mode evaluation
  modelC:evaluate();

  plog.info("Testing epoch: " .. epoch .. "/" .. settings.noEpochs);
  err_mx = 0;
  all = 0;    
  err_all =  torch.Tensor(settings.outputSize):fill(0);   
  for i = 1,table.getn(settings.listsTest), 1 do             
    dataset = Dataset(settings.listsTest[i]);
    -- prepare inputs & outputs tensors      
    local dataa = dataset.nSamplesList;
    --local targets = torch.Tensor(settings.outputSize):fill(1);
    --targets[settings.targets[noBatch]]=settings.targets[noBatch]+1;
    --for d = 1, dataa:size(1),1 do
    local datas = dataa
    local dataOutput = dataset.nSamplesListOutput
    batchSize = 100;
    inputs = torch.Tensor(batchSize,settings.inputSize,settings.SizeX,settings.SizeY);
    targets = torch.Tensor(batchSize);
    
    -- process batches
    for i = 1, batchSize,1 do      
      --for j = 1, inputs:size(2), 1 do
      -- pick frame 
      local dataX=torch.random(datas:size(1)-settings.SizeX)+settings.SizeX/2;
      local dataY=settings.SizeY/2+1;--torch.random(datas:size(2)-settings.SizeY)+settings.SizeY/2;
      --local input = datas[{{dataX},{dataY},{}}]:float():resize(datas:size(3));
      local input = datas[{{dataX-settings.SizeX/2,dataX+settings.SizeX/2-1},{dataY-settings.SizeY/2,dataY+settings.SizeY/2-1},{}}]:transpose(1,3)
      targets[i] = dataOutput[dataX][dataY];
      
      --local min = torch.min(input);
      --input = input - min;
      
      --local max = torch.max(input);
      --input = input / max;
      inputs[i]= input;
    end
    -- forward pass
    local output = modelC:forward(inputs); 
    
    for k = 1, batchSize, 1 do
      _, mx = output[k]:max(1);   
      if (mx:squeeze() ~= targets[k]) then
        err_mx = err_mx + 1;
        err_all[targets[k]] = err_all[targets[k]] + 1;
      end
      all = all + 1;
    end      
  end
  -- save error rate for graph and log
  err = 100 * err_mx / all;
  errors[epoch] = err;
  
  for k = 1, settings.outputSize, 1 do
      plog.info("Output" .. k .. " error:" .. 100 * err_all[k] / all .. "%");
  end      
  --table.insert(errorTable[i-1], err);
  log.info("Model " .. modelN .. " - epoch: " .. epoch .. "/".. settings.noEpochs .. " - err = " .. err); 
end

  if (settings.saveEpochFull == 1) then
    plog.info("Saving model: " .. modelN);
    torch.save(settings.outputFolder .. settings.modFolder .. "/" .. modelN .. ".mod", modelC);
  end
    -- draw frame error rate graph
  if (settings.drawERRs == 1) then
    gnuplot.pngfigure(settings.outputFolder .. settings.statsFolder .. '/errs'.. modelN ..'.png'); 
    --if (#settings.lists-1 == 1) then 
    --  gnuplot.plot({settings.lists[2], torch.Tensor(errorTable[1]), '-'});
    --elseif (#settings.lists-1 == 2) then 
    --  gnuplot.plot({settings.lists[2], torch.Tensor(errorTable[1]), '-'}, {settings.lists[3], torch.Tensor(errorTable[2]), '-'});
    --elseif (#settings.lists-1 == 3) then 
    ---  gnuplot.plot({settings.lists[2], torch.Tensor(errorTable[1]), '-'}, {settings.lists[3], torch.Tensor(errorTable[2]), '-'}, {settings.lists[4], torch.Tensor(errorTable[3]), '-'});
    --else
    --  flog.error('GNUPlot: not supported');
    --end    
    gnuplot.plot(errors);
    gnuplot.title('Error rates');
    gnuplot.xlabel('epoch');
    gnuplot.ylabel('error rate [%]');
    gnuplot.plotflush();
  end
  
  --evaluate experiment
errorsExtra = torch.Tensor(table.getn(settings.listsTestX))
  for i = 1,table.getn(settings.listsTestX) do      
    err_mx = 0;
    all = 0;    
    err_all =  torch.Tensor(settings.outputSize):fill(0);          
    dataset = Dataset(settings.listsTestX[i]);
    -- prepare inputs & outputs tensors      
    local dataa = dataset.nSamplesList;
    --local targets = torch.Tensor(settings.outputSize):fill(1);
    --targets[settings.targets[noBatch]]=settings.targets[noBatch]+1;
    --for d = 1, dataa:size(1),1 do
    local datas = dataa
    local dataOutput = dataset.nSamplesListOutput
    batchSize = 100;
    inputs = torch.Tensor(batchSize,settings.inputSize,settings.SizeX,settings.SizeY);
    targets = torch.Tensor(batchSize);
    
    -- process batches
    for i = 1, batchSize,1 do      
      --for j = 1, inputs:size(2), 1 do
      -- pick frame 
      local dataX=torch.random(datas:size(1)-settings.SizeX)+settings.SizeX/2;
      local dataY=settings.SizeY/2+1;--torch.random(datas:size(2)-settings.SizeY)+settings.SizeY/2;
      --local input = datas[{{dataX},{dataY},{}}]:float():resize(datas:size(3));
      local input = datas[{{dataX-settings.SizeX/2,dataX+settings.SizeX/2-1},{dataY-settings.SizeY/2,dataY+settings.SizeY/2-1},{}}]:transpose(1,3)
      targets[i] = dataOutput[dataX][dataY];
      
      --local min = torch.min(input);
      --input = input - min;
      
      --local max = torch.max(input);
      --input = input / max;
      inputs[i]= input;
    end
    
    -- forward pass
    local output = modelC:forward(inputs); 
    
    for k = 1, batchSize, 1 do
      _, mx = output[k]:max(1);   
      if (mx:squeeze() ~= targets[k]) then
        err_mx = err_mx + 1;
        err_all[targets[k]] = err_all[targets[k]] + 1;
      end
      all = all + 1;
    end      
    
    -- save error rate for graph and log
    err = 100 * err_mx / all;
    
    for k = 1, settings.outputSize, 1 do
        plog.info("Output" .. k .. " error:" .. 100 * err_all[k] / all .. "%");
    end      
    log.info("Set " .. settings.listsTestX[i] .. " - err = " .. err);  
    errorsExtra[i] = err;  
  end  
     -- draw frame error rate graph
  if (settings.drawERRs == 1) then
    gnuplot.pngfigure(settings.outputFolder .. settings.statsFolder .. '/errsE'.. modelN ..'.png'); 
    gnuplot.plot(errorsExtra);
    gnuplot.title('Error rates');
    gnuplot.xlabel('epoch');
    gnuplot.ylabel('error rate [%]');
    gnuplot.plotflush();
  end
  return errors[errors:size(1)]
  end