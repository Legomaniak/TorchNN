-- general libraries
require 'cunn'
require 'cutorch'

require 'paths'
require 'xlua'
require 'math'
--require 'torch-rnn'
require 'logroll'
require 'gnuplot'
require 'lfs'

-- program requires
require 'Training'
require 'TestModel'
require 'TrainOptim'
require 'Helper'

NN_CONVOLUTION_VOLUMETRIC, NN_CONVOLUTION_TEMPORAL, NN_CONVOLUTION_RESIZE, NN_CONVOLUTION_TRANSPOSE, NN_CONVOLUTION_VOLUMETRIC_MAX_POOLING, NN_LINEAR_RESIZE, NN_LINEAR = 0, 1, 2, 4, 8, 16, 32

function ModelTrain1D()

Settings.TrainFolder = Settings.ListFolder .. "/Train/"
Settings.TestFolder = Settings.ListFolder .. "/Test/"
--Settings.TestXFolder = Settings.ListFolder .. "TestX/"

Settings.ListTrain,Settings.ListTrainOut = ReadDir(Settings.TrainFolder,'.in','.out')
Settings.ListTest,Settings.ListTestOut = ReadDir(Settings.TestFolder,'.in','.out')
--Settings.ListTestX,Settings.ListTestXOut = ReadDir(Settings.TestXFolder)

Settings.OutputFolderNN = Settings.OutputFolder .. Settings.ModelName.."/"
Settings.StatsFolder = "/Stats/"
Settings.LogFolder = "/Log/"
Settings.ModFolder = "/Mod/"
CheckFolder(Settings.OutputFolderNN)
CheckFolder(Settings.OutputFolderNN..Settings.StatsFolder)
CheckFolder(Settings.OutputFolderNN..Settings.LogFolder)
CheckFolder(Settings.OutputFolderNN..Settings.ModFolder)
--os.execute(string.format('cp "%s" "%s"', Settings.ListFolder..'/Settings.lua', Settings.OutputFolderNN..Settings.StatsFolder..'Settings.txt'))
slog = io.open(Settings.OutputFolderNN..Settings.StatsFolder..'Settings.lua', "w")
  
-- initialize logs
flog = logroll.file_logger(Settings.OutputFolderNN .. Settings.LogFolder .. Settings.ModelName .. '.log')
plog = logroll.print_logger()
log = logroll.combine(flog, plog)

-- NN settings
slog:write("Settings.InputSize = "..Settings.InputSize.."\n")
slog:write("Settings.OutputSize = "..Settings.OutputSize.."\n")
slog:write("Settings.BatchSizeX = "..Settings.BatchSizeX.."\n")
slog:write("Settings.BatchSizeY = "..Settings.BatchSizeY.."\n")
slog:write("Settings.BatchSizeZ = "..Settings.BatchSizeZ.."\n")

slog:write("Settings.BatchN = "..Settings.BatchN.."\n")
slog:write("Settings.EpochN = "..Settings.EpochN.."\n")
slog:write("Settings.LearningRate = "..Settings.LearningRate.."\n")
slog:write("Settings.BatchSize = "..Settings.BatchSize.."\n")
slog:write("Settings.MaxIter = "..Settings.MaxIter.."\n")
  
slog:write("Settings.ModelType = "..Settings.ModelType.."\n")
slog:write("Settings.ModelSize = {{")
--if table.getn(Settings.ModelSize[1])>1
slog:write(Settings.ModelSize[1][1])
for j=2,table.getn(Settings.ModelSize[1]) do
  slog:write(","..Settings.ModelSize[1][j])
end
slog:write("}")
for i=2,table.getn(Settings.ModelSize) do
  slog:write(",{")
  slog:write(Settings.ModelSize[i][1])
  for j=2,table.getn(Settings.ModelSize[i]) do
    slog:write(","..Settings.ModelSize[i][j])
  end
  slog:write("}")
end
--else
--slog:write(Settings.ModelSize[1])
--for i=2,table.getn(Settings.ModelSize) do
--  slog:write(","..Settings.ModelSize[i])
--end
--end
slog:write("}\n")
slog:write("Settings.TrainSize = "..Settings.TrainSize.."\n")
slog:write("Settings.TestSize = "..Settings.TestSize.."\n")

slog:close()

--training settings
local MinError = 0.1
local err = -1
local errs = 100
local errsMin = 100
local iter = 0
local errH = torch.Tensor(Settings.MaxIter):fill(0)
local oldErrs = 0
local oldIters = 0

-- Criterions
Settings.Criterion = nn.MSECriterion()
--Settings.Criterion = nn.CrossEntropyCriterion()

local model = nn.Sequential()
for key,value in ipairs(Settings.ModelSize) do
  local v = value[1]  
  if v == NN_CONVOLUTION_VOLUMETRIC then   
    model:add(nn.VolumetricConvolution(value[2], value[3], value[4], value[5], value[6], value[7], value[8], value[9])) 
    
  elseif v == NN_CONVOLUTION_TEMPORAL then 
    model:add(nn.TemporalConvolution(value[2], value[3], value[4], value[5]))
    
  elseif v == NN_CONVOLUTION_RESIZE then 
    local s = model:forward(torch.randn(Settings.BatchSize,Settings.BatchSizeX,Settings.BatchSizeY,Settings.BatchSizeZ,Settings.InputSize)):size()
    local ssize = 1;
    for i = 2,s:size() do
      ssize = ssize * s[i]
    end
    model:add(nn.Reshape(s[1],ssize,1))
    
  elseif v == NN_CONVOLUTION_TRANSPOSE then 
    model:add(nn.Squeeze())
    model:add(nn.Transpose({2,3}))
--    output = model:forward(input)
--    print('Squeezed')
--    print(unpack(output:size():totable()))
--    model:add(nn.Unsqueeze(2))

  elseif v == NN_CONVOLUTION_VOLUMETRIC_MAX_POOLING then 
    model:add(nn.VolumetricMaxPooling(value[2], value[3], value[4]))
    
  elseif v == NN_LINEAR_RESIZE then 
    local s = model:forward(torch.randn(Settings.BatchSize,Settings.BatchSizeX,Settings.BatchSizeY,Settings.BatchSizeZ,Settings.InputSize)):size()
    local ssize = 1;
    for i = 2,s:size() do
      ssize = ssize * s[i]
    end
    model:add(nn.Reshape(s[1],ssize))  
    for i = 2,table.getn(value) do 
        model:add(nn.Linear(ssize,value[i]))
        model:add(nn.ReLU())
        ssize=value[i]
    end
    model:add(nn.Linear(ssize,Settings.OutputSize))
    
  elseif v == NN_LINEAR then 
    local ssize = Settings.BatchSizeX*Settings.BatchSizeY*Settings.BatchSizeZ*Settings.InputSize
    model:add(nn.Reshape(Settings.BatchSize,ssize))    
    for i = 2,table.getn(value) do 
        model:add(nn.Linear(ssize,value[i]))
        model:add(nn.ReLU())
        ssize=value[i]
    end
    model:add(nn.Linear(ssize,Settings.OutputSize))
  else 
  end
end

local modelType = "classic"
modelType = Settings.ModelType 
if (modelType == "convolve") then
  model:cuda()  -- convert model to CUDA
  Settings.Criterion:cuda()
  
elseif (modelType == "load") then
 model = torch.load(Settings.OutputFolderNN .. Settings.ModFolder .. "/" .. Settings.ModelName .. ".mod")
 errH = torch.load(Settings.OutputFolderNN .. Settings.ModFolder .. "/" .. Settings.ModelName .. ".errH")
 iter = errH:size(1)
 Settings.MaxIter=Settings.MaxIter+iter
 errH:expand(Settings.MaxIter)
 
--  model:cuda()  -- convert model to CUDA
--  Settings.Criterion:cuda()
elseif (modelType == "classic") then
  model:cuda()  -- convert model to CUDA
  Settings.Criterion:cuda()
else
  error('Model: not supported')
  return
end

print(model)

local etime = sys.clock()
local TrainIn = ReadDataAll(Settings.ListTrain)
local TrainOut = ReadDataAll(Settings.ListTrainOut)
local TestIn = ReadDataAll(Settings.ListTest)
local TestOut = ReadDataAll(Settings.ListTestOut)

log.info("Loaded " .. Settings.ModelName .. " completed in " .. sys.clock() - etime)
etime = sys.clock()

local cesta = Settings.OutputFolderNN .. Settings.ModFolder .. "/" .. Settings.ModelName .. "Min.mod"
if (modelType == "load" and paths.filep(cesta)) then
  local model = torch.load(cesta)
  local errMin = TestModelX(model, TestIn, TestOut, FillDataBatchAll1Dcuda, TestModelRawX) 
  errsMin = errMin:sum()/errMin:size(1) 
  log.info("Loaded Old ModelMin with error "..errsMin)  
end

repeat
  if Settings.DispIter then log.info("Iteration: " .. iter .. "/" .. Settings.MaxIter) end
  Training(model, TrainIn, TrainOut, FillDataBatch1Dcuda, TrainOptim);
  --TrainingRaw(model, TrainIn, TrainOut)  
  
  err = TestModelX(model, TestIn, TestOut, FillDataBatchAll1Dcuda, TestModelRawX)  
  
  errs = err:sum()/err:size(1)
  log.info("ErrorRate: " .. errs .. " %")
  --save minimal model
  if errs<errsMin then
    errsMin=errs
    torch.save(Settings.OutputFolderNN .. Settings.ModFolder .. "/" .. Settings.ModelName .. "Min.mod", model) 
    torch.save(Settings.OutputFolderNN .. Settings.ModFolder .. "/" .. Settings.ModelName .. "Min.err", err)
  end
  -- stop if stil the same
  if errs==oldErrs then
    oldIters = oldIters + 1    
  else 
    oldIters = 0
  end
  oldErrs = errs
  iter=iter+1;
  errH[iter]=errs
  
  -- logs & export model
  if iter%20==0 then 
    if Settings.DispSave then plog.info("Saving: " .. Settings.ModelName) end
    torch.save(Settings.OutputFolderNN .. Settings.ModFolder .. "/" .. Settings.ModelName .. ".mod", model) 
    torch.save(Settings.OutputFolderNN .. Settings.ModFolder .. "/" .. Settings.ModelName .. ".err", err)
    torch.save(Settings.OutputFolderNN .. Settings.ModFolder .. "/" .. Settings.ModelName .. ".errH", errH[{{1,iter}}])
  end
until errs <= MinError or iter >= Settings.MaxIter or oldIters >= 500
-- ukladani testu
plog.info("Saving: " .. Settings.ModelName)
torch.save(Settings.OutputFolderNN .. Settings.ModFolder .. "/" .. Settings.ModelName .. ".mod", model) 
torch.save(Settings.OutputFolderNN .. Settings.ModFolder .. "/" .. Settings.ModelName .. ".err", err)
torch.save(Settings.OutputFolderNN .. Settings.ModFolder .. "/" .. Settings.ModelName .. ".errH", errH[{{1,iter}}])

--local TestInX = ReadDataAll(Settings.ListTestX)
--local TestOutX = ReadDataAll(Settings.ListTestXOut)
--err = TestModel(model, TestInX, TestOutX, FillDataBatchAll4D, TestModelRaw)    
--print(err)
log.info("Finished " .. Settings.ModelName .. " completed in " .. sys.clock() - etime .. " with error " .. errs)
end