require 'TrainOptim'
--Input [M][InputSize] to [BatchSize][InputSize]
--Output [M][OutputSize] to [BatchSize][OutputSize]
function FillDataBatch1D(model, dataIn, dataOut, func)  
  -- prepare inputs & outputs tensors      
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
    func(model, inputs, outputs) 
  end
end

--Input [M][InputSize] to [BatchSize][InputSize]
--Output [M][OutputSize] to [BatchSize][OutputSize]
function FillDataBatch1Dcuda(model, dataIn, dataOut, func)  
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
    
    local inputs = torch.CudaTensor(torch.LongStorage(tIn))
    local outputs = torch.CudaTensor(torch.LongStorage(tOut))
    
  -- prepare inputs & outputs tensors      
  for batch = 1,Settings.BatchN do
    if Settings.DispBatch then plog.info("Training Batch: " .. batch .. "/" .. Settings.BatchN) end
    -- process batches
    for i = 1, Settings.BatchSize do      
      local dataM=torch.random(1,dataIn:size(1))
      inputs[i] = dataIn:select(1,dataM)
      outputs[i] = dataOut:select(1,dataM)
    end    
    func(model, inputs, outputs) 
  end
end
--Input [M][X][InputSize] to [BatchSize][BatchSizeX][InputSize]
--Output [M][OutputSize] to [BatchSize][OutputSize]
function FillDataBatch2D(model, dataIn, dataOut, func) 
  -- prepare inputs & outputs tensors     
  for batch = 1,Settings.BatchN do
    if Settings.DispBatch then plog.info("Training Batch: " .. batch .. "/" .. Settings.BatchN) end
    local tIn ={}
    table.insert(tIn,Settings.BatchSize)   
    table.insert(tIn,Settings.BatchSizeX)    
    local tOut ={}
    table.insert(tOut,Settings.BatchSize)   
    for i = 3,dataIn:dim() do
      table.insert(tIn,dataIn:size(i))
    end
    for i = 2,dataOut:dim() do
      table.insert(tOut,dataOut:size(i))
    end
    
    local inputs = torch.Tensor(torch.LongStorage(tIn))
    local outputs = torch.Tensor(torch.LongStorage(tOut))
    
    local SizeX=math.floor(Settings.BatchSizeX/2)
    -- process batches
    for i = 1, Settings.BatchSize do      
      local dataM=torch.random(1,dataIn:size(1))
      local dataX=torch.random(SizeX,dataIn:size(2)-SizeX)
      inputs[i] = dataIn:select(1,dataM):sub(dataX-SizeX,dataX+SizeX)
      outputs[i] = dataOut:select(1,dataM)
    end    
    func(model, inputs, outputs) 
  end
end

--Input [M][X][Y][InputSize] to [BatchSize][BatchSizeX][BatchSizeY][InputSize]
--Output [M][OutputSize] to [BatchSize][OutputSize]
function FillDataBatch3D(model, dataIn, dataOut, func) 
  -- prepare inputs & outputs tensors     
  for batch = 1,Settings.BatchN do
    if Settings.DispBatch then plog.info("Training Batch: " .. batch .. "/" .. Settings.BatchN) end
    local tIn ={}
    table.insert(tIn,Settings.BatchSize)   
    table.insert(tIn,Settings.BatchSizeX)   
    table.insert(tIn,Settings.BatchSizeY)  
    local tOut ={}
    table.insert(tOut,Settings.BatchSize)   
    for i = 4,dataIn:dim() do
      table.insert(tIn,dataIn:size(i))
    end
    for i = 2,dataOut:dim() do
      table.insert(tOut,dataOut:size(i))
    end  
    
    local inputs = torch.Tensor(torch.LongStorage(tIn))
    local outputs = torch.Tensor(torch.LongStorage(tOut))
    
    local SizeX=math.floor(Settings.BatchSizeX/2)
    local SizeY=math.floor(Settings.BatchSizeY/2)
    -- process batches
    for i = 1, Settings.BatchSize do      
      local dataM=torch.random(1,dataIn:size(1))
      local dataX=torch.random(SizeX,dataIn:size(2)-SizeX)
      local dataY=torch.random(SizeY,dataIn:size(3)-SizeY)
      inputs[i] = dataIn:select(1,dataM):sub(dataX-SizeX,dataX+SizeX,dataY-SizeY,dataY+SizeY)
      outputs[i] = dataOut:select(1,dataM)
    end
    func(model, inputs, outputs) 
  end
end

--Input [M][X][Y][Z][InputSize] to [BatchSize][BatchSizeX][BatchSizeY][BatchSizeZ][InputSize]
--Output [M][OutputSize] to [BatchSize][OutputSize]
function FillDataBatch4D(model, dataIn, dataOut, func) 
  -- prepare inputs & outputs tensors     
  for batch = 1,Settings.BatchN do
    if Settings.DispBatch then plog.info("Training Batch: " .. batch .. "/" .. Settings.BatchN) end
    local tIn ={}
    table.insert(tIn,Settings.BatchSize)   
    table.insert(tIn,Settings.BatchSizeX)   
    table.insert(tIn,Settings.BatchSizeY)    
    table.insert(tIn,Settings.BatchSizeZ)  
    local tOut ={}
    table.insert(tOut,Settings.BatchSize)   
    for i = 5,dataIn:dim() do
      table.insert(tIn,dataIn:size(i))
    end
--    for i = 2,dataOut:dim() do
--      table.insert(tOut,dataOut:size(i))
--    end  
    table.insert(tOut,Settings.OutputSize)
    
    local inputs = torch.Tensor(torch.LongStorage(tIn))
    local outputs = torch.Tensor(torch.LongStorage(tOut))
    
    local SizeX=math.floor(Settings.BatchSizeX/2)
    local SizeY=math.floor(Settings.BatchSizeY/2)
    local SizeZ=math.floor(Settings.BatchSizeZ/2)
    -- process batches
    for i = 1, Settings.BatchSize do    
      local dataM=torch.random(1,dataIn:size(1))
      local dataX=torch.random(1+SizeX,dataIn:size(2)-SizeX)
      local dataY=torch.random(1+SizeY,dataIn:size(3)-SizeY)
      local dataZ=torch.random(1+SizeZ,dataIn:size(4)-SizeZ)
      inputs[i] = dataIn:select(1,dataM):sub(dataX-SizeX,dataX+SizeX,dataY-SizeY,dataY+SizeY,dataZ-SizeZ,dataZ+SizeZ)
      outputs[i] = dataOut:select(1,dataM):select(1,dataX):select(1,dataY):select(1,dataZ)
    end
    func(model, inputs, outputs) 
  end
end

--Input [M][X][Y][Z][InputSize] to [BatchSize][BatchSizeX][BatchSizeY][BatchSizeZ][InputSize]
--Output [M][OutputSize] to [BatchSize][OutputSize]
function FillDataBatch4Dpar(model, dataIn, dataOut, func) 
  local Parallel = require "Parallel"
  local N = 8
  -- prepare inputs & outputs tensors     
  for batch = 1,Settings.BatchN do
    if Settings.DispBatch then plog.info("Training Batch: " .. batch .. "/" .. Settings.BatchN) end

    local tIn ={}
    table.insert(tIn,Settings.BatchSize)   
    table.insert(tIn,Settings.BatchSizeX)   
    table.insert(tIn,Settings.BatchSizeY)    
    table.insert(tIn,Settings.BatchSizeZ)  
    local tOut ={}
    table.insert(tOut,Settings.BatchSize)   
    for i = 5,dataIn:dim() do
      table.insert(tIn,dataIn:size(i))
    end
--    for i = 2,dataOut:dim() do
--      table.insert(tOut,dataOut:size(i))
--    end  
    table.insert(tOut,Settings.OutputSize)
    
    local inputs = torch.Tensor(torch.LongStorage(tIn))
    local outputs = torch.Tensor(torch.LongStorage(tOut))
    
    local SizeX=math.floor(Settings.BatchSizeX/2)
    local SizeY=math.floor(Settings.BatchSizeY/2)
    local SizeZ=math.floor(Settings.BatchSizeZ/2)
    
    function VYBER(i)
      local CUR = 0
      return function()
        if CUR > i-1 then return end
        CUR = CUR + 1
        return CUR
      end
    end
    local RUTINA = string.dump(
      function(thread)
      FOR(
        function(i) 
      local dataM=torch.random(1,dataIn:size(1))
      local dataX=torch.random(1+SizeX,dataIn:size(2)-SizeX)
      local dataY=torch.random(1+SizeY,dataIn:size(3)-SizeY)
      local dataZ=torch.random(1+SizeZ,dataIn:size(4)-SizeZ)
      inputs[i] = dataIn:select(1,dataM):sub(dataX-SizeX,dataX+SizeX,dataY-SizeY,dataY+SizeY,dataZ-SizeZ,dataZ+SizeZ)
      outputs[i] = dataOut:select(1,dataM):select(1,dataX):select(1,dataY):select(1,dataZ)
          return thread,i, math.pow(i, 2), math.sqrt(i) 
        end)
      end)
    
    local function VYPIS(t,i,input,output)
      inputs[i] = input
      outputs[i] = output
    end

    --Parallel.ForEach(VYBER(10), RUTINA, VYPIS, N)
    Parallel.For(1, Settings.BatchSize, RUTINA, VYPIS,N)

    func(model, inputs, outputs) 
  end
  
end
--Input [M][InputSize] to [BatchSize][InputSize]
--Output [M][OutputSize] to [BatchSize][OutputSize]
function FillDataBatchAll1D(model, dataIn, dataOut, func)
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
  
  local d = 0
  for n = 1,dataIn:size(1) do
    d=d+1
    inputs[d] = dataIn:select(1,n)
    outputs[d] = dataOut:select(1,n)
    if d==Settings.BatchSize then
      d = 0
      -- process batches    
      func(model, inputs, outputs)   
    end
  end
  if d ~= 0 then      
    for n = 1,dataIn:size(1) do
      d=d+1
      inputs[d] = dataIn:select(1,n)
      outputs[d] = dataOut:select(1,n)
      if d==Settings.BatchSize then
        -- process batch   
        func(model, inputs, outputs)   
        break
      end
    end
  end  
end  

--Input [M][InputSize] to [BatchSize][InputSize]
--Output [M][OutputSize] to [BatchSize][OutputSize]
function FillDataBatchAll1Dcuda(model, dataIn, dataOut, func)
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
  
  local inputs = torch.CudaTensor(torch.LongStorage(tIn))
  local outputs = torch.CudaTensor(torch.LongStorage(tOut))
  
  local d = 0
  for n = 1,dataIn:size(1) do
    d=d+1
    inputs[d] = dataIn:select(1,n)
    outputs[d] = dataOut:select(1,n)
    if d==Settings.BatchSize then
      d = 0
      -- process batches    
      func(model, inputs, outputs)   
    end
  end
  if d ~= 0 then      
    for n = 1,dataIn:size(1) do
      d=d+1
      inputs[d] = dataIn:select(1,n)
      outputs[d] = dataOut:select(1,n)
      if d==Settings.BatchSize then
        -- process batch   
        func(model, inputs, outputs)   
        break
      end
    end
  end  
end  

--Input [M][X][InputSize] to [BatchSize][BatchSizeX][InputSize]
--Output [M][OutputSize] to [BatchSize][OutputSize]
function FillDataBatchAll2D(model, dataIn, dataOut, func)
  if Settings.DispBatch then plog.info("Training Batch: " .. batch .. "/" .. Settings.BatchN) end
  local tIn ={}
  table.insert(tIn,Settings.BatchSize)   
  table.insert(tIn,Settings.BatchSizeX)    
  local tOut ={}
  table.insert(tOut,Settings.BatchSize)   
  for i = 3,dataIn:dim() do
    table.insert(tIn,dataIn:size(i))
  end
  for i = 2,dataOut:dim() do
    table.insert(tOut,dataOut:size(i))
  end
  
  local inputs = torch.Tensor(torch.LongStorage(tIn))
  local outputs = torch.Tensor(torch.LongStorage(tOut))
  
  local SizeX=math.floor(Settings.BatchSizeX/2)
  -- process batches
  local d = 0;
  for n = 1,dataIn:size(1) do
    for m = SizeX,dataIn:size(2)-SizeX do
      d=d+1
      inputs[d] = dataIn:select(1,n):sub(m-SizeX,m+SizeX)
      outputs[d] = dataOut:select(1,n)
      if d==Settings.BatchSize then
        d = 0
        -- process batches    
        func(model, inputs, outputs); 
      end
    end
  end
  if d ~= 0 then      
    for n = 1,dataIn:size(1) do
      for m = SizeX,dataIn:size(2)-SizeX do
        d=d+1
        inputs[d] = dataIn:select(1,n):sub(m-SizeX,m+SizeX)
        outputs[d] = dataOut:select(1,n)
        if d==Settings.BatchSize then
          -- process batch   
          func(model, inputs, outputs);    
          break
        end
      end
    end
  end  
end   

--Input [M][X][Y][InputSize] to [BatchSize][BatchSizeX][BatchSizeY][InputSize]
--Output [M][OutputSize] to [BatchSize][OutputSize]
function FillDataBatchAll3D(model, dataIn, dataOut, func)
  if Settings.DispBatch then plog.info("Training Batch: " .. batch .. "/" .. Settings.BatchN) end
  local tIn ={}
  table.insert(tIn,Settings.BatchSize)   
  table.insert(tIn,Settings.BatchSizeX)   
  table.insert(tIn,Settings.BatchSizeY)    
  local tOut ={}
  table.insert(tOut,Settings.BatchSize)   
  for i = 3,dataIn:dim() do
    table.insert(tIn,dataIn:size(i))
  end
  for i = 2,dataOut:dim() do
    table.insert(tOut,dataOut:size(i))
  end
  
  local inputs = torch.Tensor(torch.LongStorage(tIn))
  local outputs = torch.Tensor(torch.LongStorage(tOut))
  
  local SizeM=math.floor(Settings.BatchSizeM/2)
  local SizeX=math.floor(Settings.BatchSizeX/2)
  local SizeY=math.floor(Settings.BatchSizeY/2)
  -- process batches
  local d = 0;
  for n = 1,dataIn:size(1) do
    for m = SizeX,dataIn:size(2)-SizeX do
      for l = SizeY,dataIn:size(3)-SizeY do
        d=d+1
        inputs[d] = dataIn:select(1,n):sub(m-SizeX,m+SizeX,l-SizeY,l+SizeY)
        outputs[d] = dataOut:select(1,n)
        if d==Settings.BatchSize then
          d = 0
          -- process batches    
          func(model, inputs, outputs); 
        end
      end
    end
  end
  if d ~= 0 then      
    for n = 1,dataIn:size(1) do
      for m = SizeX,dataIn:size(2)-SizeX do
        for l = SizeY,dataIn:size(3)-SizeY do
          d=d+1
          inputs[d] = dataIn:select(1,n):sub(m-SizeX,m+SizeX,l-SizeY,l+SizeY)
          outputs[d] = dataOut:select(1,n)
          if d==Settings.BatchSize then
            -- process batch   
            func(model, inputs, outputs);    
            break
          end
        end
      end
    end
  end  
end 

--Input [M][X][Y][Z][InputSize] to [BatchSize][BatchSizeX][BatchSizeY][BatchSizeZ][InputSize]
--Output [M][OutputSize] to [BatchSize][OutputSize]
function FillDataBatchAll4D(model, dataIn, dataOut, func)
  if Settings.DispBatch then plog.info("Training Batch: " .. batch .. "/" .. Settings.BatchN) end
  local tIn ={}
  table.insert(tIn,Settings.BatchSize)   
  table.insert(tIn,Settings.BatchSizeX)   
  table.insert(tIn,Settings.BatchSizeY)    
  table.insert(tIn,Settings.BatchSizeZ)  
  local tOut ={}
  table.insert(tOut,Settings.BatchSize)   
  for i = 5,dataIn:dim() do
    table.insert(tIn,dataIn:size(i))
  end
--    for i = 2,dataOut:dim() do
--      table.insert(tOut,dataOut:size(i))
--    end  
  table.insert(tOut,Settings.OutputSize)
  
  local inputs = torch.Tensor(torch.LongStorage(tIn))
  local outputs = torch.Tensor(torch.LongStorage(tOut))
  
  local SizeX=math.floor(Settings.BatchSizeX/2)
  local SizeY=math.floor(Settings.BatchSizeY/2)
  local SizeZ=math.floor(Settings.BatchSizeZ/2)
  
  -- process batches
  local d = 0;
  for n = 1,dataIn:size(1) do
    for m = 1+SizeX,dataIn:size(2)-SizeX do
      for l = 1+SizeY,dataIn:size(3)-SizeY do
        for k = 1+SizeZ,dataIn:size(4)-SizeZ do
          d=d+1
          inputs[d] = dataIn:select(1,n):sub(m-SizeX,m+SizeX,l-SizeY,l+SizeY,k-SizeZ,k+SizeZ)
          outputs[d] = dataOut:select(1,n):select(1,m):select(1,l):select(1,k)
          if d==Settings.BatchSize then
            d = 0
            -- process batches    
            func(model, inputs, outputs); 
          end
        end
      end
    end
  end
  while d ~= 0 do      
    for n = 1,dataIn:size(1) do
      for m = 1+SizeX,dataIn:size(2)-SizeX do
        for l = 1+SizeY,dataIn:size(3)-SizeY do
          for k = 1+SizeZ,dataIn:size(4)-SizeZ do
            d=d+1
            inputs[d] = dataIn:select(1,n):sub(m-SizeX,m+SizeX,l-SizeY,l+SizeY,k-SizeZ,k+SizeZ)
            outputs[d] = dataOut:select(1,n):select(1,m):select(1,l):select(1,k)
            if d==Settings.BatchSize then
              -- process batch   
              func(model, inputs, outputs);    
              break
            end
          end
          if d==Settings.BatchSize then break end
        end
        if d==Settings.BatchSize then break end
      end
      if d==Settings.BatchSize then break end
    end
    if d==Settings.BatchSize then break end
  end  
end   
