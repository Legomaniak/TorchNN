
function TestNN(modelC, batchInputs, batchLabels, outputSize)
  local err_mx = 0;
  local all = 0;    
  local err_all =  torch.Tensor(outputSize):fill(0);   
  
  local batchSize = 20000;--batchInputs:size(1) * batchInputs:size(2)--10000;
  local delka = batchInputs:size(1)*batchInputs:size(2);
  local inputss = batchInputs:resize(delka,batchInputs:size(3))
  local targetss = batchLabels:resize(delka)
  local delkaN = delka/batchSize;
  for n = 1,delkaN,1 do
    local inputs = inputss[{{batchSize*(n-1)+1,batchSize*n},{}}];
    local targets = targetss[{{batchSize*(n-1)+1,batchSize*n}}];
    -- forward pass
    local output = modelC:forward(inputs); 
    for k = 1, batchSize, 1 do
      local _, mx = output[k]:max(1);   
      if (mx:squeeze() ~= targets[k]) then
        err_mx = err_mx + 1;
        err_all[targets[k]] = err_all[targets[k]] + 1;
      end
      all = all + 1;
    end     
  end
  if delka%batchSize > 0 then
  local inputs = torch.Tensor(batchSize,inputss:size(2));
  local targets = torch.Tensor(batchSize);
    for i = 1, delka%batchSize do
      inputs[i] = inputss[i];
      targets[i] = targetss[i];
    end
    for i = delka%batchSize, batchSize do
      local ind = torch.random(1,delka);
      inputs[i] = inputss:select(1,ind);
      targets[i] = targetss[ind];
    end
    -- forward pass
    local output = modelC:forward(inputs);
    for k = 1, batchSize, 1 do
      local _, mx = output[k]:max(1);   
      if (mx:squeeze() ~= targets[k]) then
        err_mx = err_mx + 1;
        err_all[targets[k]] = err_all[targets[k]] + 1;
      end
      all = all + 1;
    end    
  end
      
  -- save error rate for graph and log
  err = 100 * err_mx / all;
  
  for k = 1, outputSize, 1 do
      plog.info("Output" .. k .. " error:" .. 100 * err_all[k] / all .. "%");
      --print("Output" .. k .. " error:" .. 100 * err_all[k] / all .. "%");
  end      
  return err;
end
function TestNNconv(modelC, batchInputs, batchLabels, outputSize)
  local err_mx = 0;
  local all = 0;    
  local err_all =  torch.Tensor(outputSize):fill(0);   
  
  local batchSize = 20000;--batchInputs:size(1) * batchInputs:size(2)--10000;
  local delka = batchInputs:size(1)*batchInputs:size(2);
  local inputss = batchInputs:resize(delka,batchInputs:size(3),1)
  local targetss = batchLabels:resize(delka)
  local delkaN = delka/batchSize;
  for n = 1,delkaN,1 do
    local inputs = inputss[{{batchSize*(n-1)+1,batchSize*n},{},{}}];
    local targets = targetss[{{batchSize*(n-1)+1,batchSize*n}}];
    -- forward pass
    local output = modelC:forward(inputs); 
    for k = 1, batchSize, 1 do
      local _, mx = output[k]:max(1);   
      if (mx:squeeze() ~= targets[k]) then
        err_mx = err_mx + 1;
        err_all[targets[k]] = err_all[targets[k]] + 1;
      end
      all = all + 1;
    end     
  end
  if delka%batchSize > 0 then
  local inputs = torch.Tensor(batchSize,inputss:size(2),inputss:size(3));
  local targets = torch.Tensor(batchSize);
    for i = 1, delka%batchSize do
      inputs[i] = inputss[i];
      targets[i] = targetss[i];
    end
    for i = delka%batchSize, batchSize do
      local ind = torch.random(1,delka);
      inputs[i] = inputss:select(1,ind);
      targets[i] = targetss[ind];
    end
    -- forward pass
    local output = modelC:forward(inputs);
    for k = 1, batchSize, 1 do
      local _, mx = output[k]:max(1);   
      if (mx:squeeze() ~= targets[k]) then
        err_mx = err_mx + 1;
        err_all[targets[k]] = err_all[targets[k]] + 1;
      end
      all = all + 1;
    end    
  end
      
  -- save error rate for graph and log
  err = 100 * err_mx / all;
  
  for k = 1, outputSize, 1 do
      plog.info("Output" .. k .. " error:" .. 100 * err_all[k] / all .. "%");
      --print("Output" .. k .. " error:" .. 100 * err_all[k] / all .. "%");
  end      
  return err;
end