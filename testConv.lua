require 'torch'
require 'paths'
require 'xlua'
require 'math'
require 'nn'
require 'logroll'
require 'gnuplot'
require 'optim'
require 'dp'

require 'TensorSaveLoad'


delkaVstupu = 1
pocetVstupu = 1
pocetVystupu = 1
delkaKernel = 5

input = torch.DoubleTensor(pocetVstupu,delkaKernel*2,delkaKernel*2)
i=0
input:apply(function()
  i = i + 1
  return i
end)
input:fill(1)

--SpatialConvolution
model = nn.Sequential()
conv = nn.SpatialConvolution(pocetVstupu,pocetVystupu,delkaKernel,delkaKernel)
--conv = nn.SpatialConvolution(pocetVstupu,pocetVystupu,5,5,1,1,2,2)
conv.bias:fill(0)
conv.weight:fill(1/delkaKernel/delkaKernel)
model:add(conv)

output = conv:forward(input)
print(unpack(output:size():totable()))


