require 'torch'
require 'paths'
require 'xlua'
require 'math'
require 'nn'
require 'logroll'
require 'gnuplot'
require 'optim'

require 'TensorSaveLoad'

Settings={}
Settings.BatchSize=50
Settings.InputSize=300  
Settings.OutputSize=4--12 

--Add NN settings
Settings.BatchSizeX=1
Settings.BatchSizeY=5
Settings.BatchSizeZ=5
print("Input size "..Settings.BatchSizeY*Settings.BatchSizeZ*Settings.InputSize*2 .. "B")



NN_CONVOLUTION_VOLUMETRIC, NN_CONVOLUTION_TEMPORAL, NN_CONVOLUTION_RESIZE, NN_CONVOLUTION_TRANSPOSE, NN_CONVOLUTION_VOLUMETRIC_MAX_POOLING, NN_LINEAR_RESIZE, NN_LINEAR = 0, 1, 2, 4, 8, 16, 32


Settings.ModelSize = {{NN_CONVOLUTION_VOLUMETRIC,1,50,3,3,3,1,1,1},{NN_CONVOLUTION_VOLUMETRIC_MAX_POOLING,1,2,1},{NN_CONVOLUTION_VOLUMETRIC,50,50,3,3,3,1,1,1},{NN_CONVOLUTION_VOLUMETRIC_MAX_POOLING,1,2,1},{NN_CONVOLUTION_TRANSPOSE},{NN_CONVOLUTION_TEMPORAL,50,5,1,1},{NN_CONVOLUTION_TEMPORAL,5,5,1,1},{NN_LINEAR_RESIZE, 10}}

--input = torch.randn(Settings.BatchSize,1,Settings.BatchSizeY,Settings.BatchSizeZ,Settings.InputSize)
input = torch.randn(Settings.BatchSize,Settings.BatchSizeX,Settings.BatchSizeY,Settings.BatchSizeZ,Settings.InputSize)

model = nn.Sequential()

local model = nn.Sequential()
for key,value in ipairs(Settings.ModelSize) do
  local v = value[1]  
  if v == NN_CONVOLUTION_VOLUMETRIC then   
    model:add(nn.VolumetricConvolution(value[2], value[3], value[4], value[5], value[6], value[7], value[8], value[9])) 
    
  elseif v == NN_CONVOLUTION_TEMPORAL then 
    model:add(nn.TemporalConvolution(value[2], value[3], value[4]))
    
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

print(model)