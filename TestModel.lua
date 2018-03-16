--[Input] to [Output]
function TestModelOne(model, dataIn, dataOut)
  err = 0
  -- forward pass
  local output = model:forward(dataIn)
    if (output ~= dataOut) then
      err = 1
    end
  return err
end

--[X][Input] to [X][OutputSize]
function TestModelRaw(model, dataIn, dataOut)
  local err = 0
  local all = 0
  -- forward pass
  local output = model:forward(dataIn)
  --search 1 max val
  for k = 1, dataOut:size(1) do
    local _, mx = output[k]:max(1)
    local _, mxt = dataOut[k]:max(1)
    if (mx:squeeze() ~= mxt:squeeze()) then
      err = err + 1
      --err_all[targets[k]] = err_all[targets[k]] + 1
    end
    all = all + 1
  end   
  --for k = 1, dataOut:size(1) do
  --  local _, mx = output[k]:max(1)
  --  if (mx:squeeze() ~= dataOut[k]) then
  --    err = err + 1
  --    --err_all[targets[k]] = err_all[targets[k]] + 1
  --  end
  --  all = all + 1
  --end     
  return all,err
end

--[X][Input] to [X][OutputSize]
function TestModelRawX(model, dataIn, dataOut)
  local err = torch.Tensor(Settings.OutputSize,Settings.OutputSize):fill(0)
  local all = torch.Tensor(Settings.OutputSize):fill(0)
  -- forward pass
  local output = model:forward(dataIn)
  --search 1 max val
  for k = 1, dataOut:size(1) do
    local _, mx = output[k]:max(1)
    local _, mxt = dataOut[k]:max(1)
    local mxs = mx:squeeze()
    local mxts = mxt:squeeze()
    if (mxs ~= mxts) then
     err[mxts][mxs] = err[mxts][mxs]+1
      --err_all[targets[k]] = err_all[targets[k]] + 1
    end
    all[mxts] = all[mxts] + 1
  end   
  --for k = 1, dataOut:size(1) do
  --  local _, mx = output[k]:max(1)
  --  if (mx:squeeze() ~= dataOut[k]) then
  --    err = err + 1
  --    --err_all[targets[k]] = err_all[targets[k]] + 1
  --  end
  --  all = all + 1
  --end     
  return all,err
end