require 'optim'

function OptimPP(modelC, criterion, epochN, LR, batchInputs, batchLabels)
  --batchInputs = torch.reshape(batchInputs,batchInputs:size(1),batchInputs:size(2),1)
  local params, gradParams = modelC:getParameters()
  local optimState = {learningRate = LR}

  for epoch = 1, epochN do
    --plog.info("Training Epoch: " .. epoch .. "/" .. epochN);
    function feval(params)
      gradParams:zero()

      local outputs = modelC:forward(batchInputs)
      local loss = criterion:forward(outputs, batchLabels)
      local dloss_doutputs = criterion:backward(outputs, batchLabels)
      modelC:backward(batchInputs, dloss_doutputs)

      return loss, gradParams
    end
    optim.sgd(feval, params, optimState)
  end    
end