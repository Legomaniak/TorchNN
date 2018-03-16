require 'optim'

function TrainOptim(model, dataIn, dataOut)
  local params, gradParams = model:getParameters()
  local optimState = {learningRate = Settings.LearningRate}
  for epoch = 1, Settings.EpochN do
    if Settings.DispEpoch then plog.info("Training Epoch: " .. epoch .. "/" .. Settings.EpochN) end
    function feval(params)
      gradParams:zero()
      local outputs = model:forward(dataIn)
      local loss = Settings.Criterion:forward(outputs, dataOut)
      local dloss = Settings.Criterion:backward(outputs, dataOut)
      model:backward(dataIn, dloss)
      return loss, gradParams
    end
    optim.sgd(feval, params, optimState)
  end    
end