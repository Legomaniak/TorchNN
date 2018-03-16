-- general libraries
require 'torch'
require 'paths'
require 'xlua'
require 'math'
require 'nn'
require 'logroll'
require 'gnuplot'
require 'lfs'

-- program requires
require 'TrainingOptim'
require 'Helper'

-- initialize settings
Settings = {};
  
Settings.ListFolder = "NoveKostky/"; 
Settings.ModelName = "NoveKostky3DModel";--jmeno trenovane site
--Settings.ModelName = "NoveKostkyFullModel";--jmeno trenovane site--7.3%
Settings.Lists = {}
listIdent = 'norm'
Settings.ListsTest = {}
listTestIdent = 'OUT'
Settings.ListsTestX = {} 
listTestXIdent = 'OUTX'

for file in lfs.dir(Settings.ListFolder) do
    if lfs.attributes(Settings.ListFolder.."/"..file,"mode") == "file" then 
      if string.find(file,listTestIdent) then
        table.insert(Settings.ListsTest,file);
      elseif string.find(file,listTestXIdent) then
        table.insert(Settings.ListsTestX,file);
      elseif string.find(file,listIdent) then
        table.insert(Settings.Lists,file);
      end
    end
end

Settings.OutputFolder = "OutputFolder/" .. Settings.ModelName;
Settings.StatsFolder = "/Stats/";
Settings.LogFolder = "/Log/";
Settings.ModFolder = "/Mod/";
CheckFolder(Settings.OutputFolder..Settings.StatsFolder)
CheckFolder(Settings.OutputFolder..Settings.LogFolder)
CheckFolder(Settings.OutputFolder..Settings.ModFolder)

-- NN
Settings.BatchN=10
Settings.EpochN=10
Settings.LearningRate=0.08
Settings.BatchSize=3000 
Settings.InputSize=300  
Settings.OutputSize=4--12 

--Add NN settings
Settings.BatchSizeX=7  
Settings.BatchSizeY=7  
Settings.BatchSizeZ=7 

--Conv
Settings.SizeX=7  
Settings.SizeY=7  
Settings.SizeZ=7
Settings.dX=5  
Settings.dY=5  
Settings.dZ=5  

--Display
Settings.DispEpoch=false
Settings.DispBatch=false
Settings.DispFile=false
Settings.DispSave=false
Settings.DispIter=true

-- Criterions
--Settings.Criterion = nn.ClassNLLCriterion()
--Settings.Criterion = nn.CrossEntropyCriterion() --pro 1D vystup
Settings.Criterion = nn.MSECriterion()
  
-- initialize logs
flog = logroll.file_logger(Settings.OutputFolder .. Settings.LogFolder .. Settings.ModelName .. '.log');
plog = logroll.print_logger();
log = logroll.combine(flog, plog);

local model;
-- initialize the network
local modelType = "classic";
if (modelType == "convolve") then
  model = nn.Sequential();  
  --[[VolumetricConvolution(nInputPlane, nOutputPlane, kT, kW, kH [, dT, dW, dH, padT, padW, padH])
  forward(input) = nInputPlane x time x height x width
  nInputPlane: The number of expected input planes in the image given into forward().
  nOutputPlane: The number of output planes the convolution layer will produce.
  kT: The kernel size of the convolution in time
  kW: The kernel width of the convolution
  kH: The kernel height of the convolution
  dT: The step of the convolution in the time dimension. Default is 1.
  dW: The step of the convolution in the width dimension. Default is 1.
  dH: The step of the convolution in the height dimension. Default is 1.
  padT: Additional zeros added to the input plane data on both sides of time axis. Default is 0. (kT-1)/2 is often used here.
  padW: Additional zeros added to the input plane data on both sides of width axis. Default is 0. (kW-1)/2 is often used here.
  padH: Additional zeros added to the input plane data on both sides of height axis. Default is 0. (kH-1)/2 is often used here.]]
  model:add(nn.VolumetricConvolution(1,1,settings.sizeX,settings.sizeY,settings.sizeZ,settings.dX,settings.dY,settings.dZ,(settings.sizeX-1)/2,(settings.sizeY-1)/2,(settings.sizeZ-1)/2));
  s = model:forward(torch.randn(settings.batchSize,1,480,settings.sizeY,settings.sizeZ)):size()
  ssize= s[2]*s[3]*s[4]*s[5];
  model:add(nn.Reshape(s[1],ssize))
  
  -- output layer
  ll = nn.Linear(ssize, settings.outputSize);   -- layer size  
  ll.weight:zero();   -- default weights
  ll.bias:zero();     -- default bias
  model:add(ll);
  
elseif (modelType == "load") then
 model = torch.load(Settings.OutputFolder .. Settings.ModFolder .. "/" .. Settings.ModelName .. ".mod");
elseif (modelType == "classic") then
  model = nn.Sequential();  
  
  
  model:add(nn.Linear(Settings.InputSize,200));
  model:add(nn.Linear(200,50));
  model:add(nn.Linear(50,100));
  
  --  layer
  ll = nn.Linear(100,Settings.OutputSize);   -- layer size  
  --ll.weight = torch.randn(300, Settings.OutputSize);
  --ll.bias= torch.randn(Settings.OutputSize);     -- default bias
  model:add(ll);
    
else
  error('Model: not supported');
  return
end

print(model)

--training settings
local maxIter = 200
local MinError = 0.1

local etime = sys.clock()
local dataIn = ReadDataAll(Settings.Lists)
local dataOut = ReadDataAll(Settings.ListsTest)
--local dataX = ReadDataAll(Settings.ListsTestX)

--experiment loading
local s = dataIn:size();
dataIn=dataIn:resize(s[1],s[2],s[3]*s[4]):transpose(2,3)--1D
dataIn=dataIn[{{},{},{1,Settings.InputSize}}]

--dataIn=dataIn:transpose(2,3):transpose(3,4)--2D
--dataIn=dataIn[{{},{},{},{1,Settings.InputSize}}]

s = dataOut:size();
dataOut=dataOut:resize(s[1],s[2]*s[3])--1D
local dataOut2=torch.Tensor(s[1],s[2]*s[3],Settings.OutputSize) 
--local dataOut2=torch.Tensor(s[1],s[2],s[3],Settings.OutputSize) 

for f=1,dataOut:size(1) do
  --dataOut:select(1,f):mul(f)
  --dataOut:select(1,f):add(1)
  
  --dataOut2[{{f},{},{f}}]=dataOut:select(1,f)
  
  for i=1,dataOut:size(2) do --1D
    if dataOut[f][i]==0 then
    dataOut2[f][i][1]=1
    else
    dataOut2[f][i][f+1]=1
    end
  end
  --for i=1,dataOut:size(2) do  --2D 
  --  for j=1,dataOut:size(3) do
  --    if dataOut[f][i][j]==0 then
  --    dataOut2[f][i][j][1]=1
  --    else
  --    dataOut2[f][i][j][f+1]=1
  --    end
  --  end
  --end

end
dataOut=dataOut2

log.info("Loaded " .. Settings.ModelName .. " completed in " .. sys.clock() - etime);  
etime = sys.clock(); 
local err = -1
local iter = 0;
repeat
  if Settings.DispIter then plog.info("Iteration: " .. iter .. "/" .. maxIter) end
  TrainingOptim(model, dataIn, dataOut, FillDataBatch1D);
  err = TestModel(model, dataIn, dataOut, FillDataBatchAll1D)  
  iter=iter+1;
until err < MinError or iter > maxIter
--local err = TestConv(ModelName, model, settings.listsTestX, batchSize)  
print(err)
log.info("Finished " .. Settings.ModelName .. " completed in " .. sys.clock() - etime .. " with error " .. err);  

