require 'torch'
require 'ModelTrain1D'

local OutFolder='Out/IndianPinesNew_5'

cmd = torch.CmdLine()
cmd:option('-lf',OutFolder,'ListFolder')
params = cmd:parse(arg)
-- initialize settings
Settings = {};
Settings.ListFolder = params.lf; 
--Settings.ListFolder = "IndianPinesPCA_3_50"; 

--if Settings.ListFolder == nill then print("No input defined") return end

require(Settings.ListFolder..'.Settings')
--NN_CONVOLUTION_VOLUMETRIC - nInputPlane, nOutputPlane, kT, kW, kH , dT, dW, dH
--NN_CONVOLUTION_TEMPORAL - inputFrameSize, outputFrameSize, kW
--NN_CONVOLUTION_RESIZE - resize after volumetric convolution 
--NN_CONVOLUTION_TRANSPOSE - resize with transpose after volumetric convolution for temporal convolution
--NN_CONVOLUTION_VOLUMETRIC_MAX_POOLING 
--NN_LINEAR_RESIZE - linear layer after convolution
--NN_LINEAR - common Linear NN

--local V = {{5},{10},{15},{20},{30},{40},{50},{75},{100}}
--local V = {{{5}},{{15}},{{30}},{{50}},{{100}}} --Model
--local V = {{{5,5}},{{5,15}},{{15,15}},{{15,30}},{{30,5}},{{30,15}},{{30,30}},{{30,50}},{{50,30}},{{50,50}},{{50,100}},{{100,5}},{{100,20}},{{100,50}},{{100,100}}} --model2D
--local V = {{{30,30,30}},{{100,30,100}},{{100,100,100}}}--model3D
--local V = {{{30,30,30,30}},{{100,30,100,100}},{{100,100,100,100}}}--model4D
--local V = {{5},{15},{30},{50},{100}}
--local V = {{{1,1,3,30,3,1,50,1},{10}},{{1,1,3,3,3,1,1,1},{10}},{{1,1,3,30,3,1,50,1},{1,1,3,3,3,1,1,1},{10}}}

--3x3
--local V = {{{1,5,3,3,3,1,1,1},{-2},{5,1,1,1},{10}},{{1,10,3,3,3,1,1,1},{-2},{10,1,1,1},{10}},{{1,5,3,3,3,1,1,1},{-2},{5,1,1,1},{20}}}--Model2C
--local V = {{{1,5,3,3,3,1,1,1},{-2},{5,5,1,1},{5,1,1,1},{10}},{{1,10,3,3,3,1,1,1},{-2},{10,10,1,1},{10,1,1,1},{10}},{{1,5,3,3,3,1,1,1},{-2},{5,5,1,1},{5,1,1,1},{20}},{{1,20,3,3,3,1,1,1},{-2},{20,10,1,1},{10,1,1,1},{10}},{{1,50,3,3,3,1,1,1},{-2},{50,25,1,1},{25,1,1,1},{10}}}--Model3C
--local V = {{{1,5,3,3,3,1,1,1},{-2},{5,5,1,1},{5,5,1,1},{5,5,1,1},{5,5,1,1},{10}},{{1,10,3,3,3,1,1,1},{-2},{10,10,1,1},{10,5,1,1},{5,5,1,1},{5,5,1,1},{10}}}--Model5C

--5x5
--local V = {{{1,5,3,3,3,1,1,1},{5,5,3,3,3,1,1,1},{-2},{5,1,1,1},{10}},{{1,5,3,3,3,1,1,1},{5,5,3,3,3,1,1,1},{-2},{5,5,1,1},{5,5,1,1},{10}},{{1,50,3,3,3,1,1,1},{1,2,1},{50,50,3,3,3,1,1,1},{1,2,1},{-2},{50,5,1,1},{5,5,1,1},{10}}}--Model2C
--local V = {{{NN_CONVOLUTION_VOLUMETRIC,1,5,3,3,3,1,1,1},{NN_CONVOLUTION_VOLUMETRIC,5,5,3,3,3,1,1,1},{NN_CONVOLUTION_TRANSPOSE},{NN_CONVOLUTION_TEMPORAL,5,1,1,1},{NN_LINEAR_RESIZE, 10}},{{NN_CONVOLUTION_VOLUMETRIC,1,5,3,3,3,1,1,1},{NN_CONVOLUTION_VOLUMETRIC,5,5,3,3,3,1,1,1},{NN_CONVOLUTION_TRANSPOSE},{NN_CONVOLUTION_TEMPORAL,5,5,1,1},{NN_CONVOLUTION_TEMPORAL,5,5,1,1},{NN_LINEAR_RESIZE, 10}},{{NN_CONVOLUTION_VOLUMETRIC,1,50,3,3,3,1,1,1},{NN_CONVOLUTION_VOLUMETRIC_MAX_POOLING,1,2,1},{NN_CONVOLUTION_VOLUMETRIC,50,50,3,3,3,1,1,1},{NN_CONVOLUTION_VOLUMETRIC_MAX_POOLING,1,2,1},{NN_CONVOLUTION_TRANSPOSE},{NN_CONVOLUTION_TEMPORAL,50,5,1,1},{NN_CONVOLUTION_TEMPORAL,5,5,1,1},{NN_LINEAR_RESIZE, 10}}}--Model5C
--local V = {{{NN_CONVOLUTION_VOLUMETRIC,1,50,3,3,3,1,1,1},{NN_CONVOLUTION_VOLUMETRIC,50,50,3,3,3,1,1,1},{NN_CONVOLUTION_TRANSPOSE},{NN_CONVOLUTION_TEMPORAL,50,50,1,1},{NN_CONVOLUTION_TEMPORAL,50,50,1,1},{NN_LINEAR_RESIZE,100,100}}}--Model1C

--7x7
local V = {
  {{NN_CONVOLUTION_VOLUMETRIC,1,50,3,3,3,1,1,1},{NN_CONVOLUTION_VOLUMETRIC,50,50,5,5,5,1,1,1},{NN_CONVOLUTION_TRANSPOSE},{NN_CONVOLUTION_TEMPORAL,50,50,1,1},{NN_LINEAR_RESIZE,100}},
  {{NN_CONVOLUTION_VOLUMETRIC,1,20,3,3,3,1,1,1},{NN_CONVOLUTION_VOLUMETRIC,20,20,5,5,5,1,1,1},{NN_CONVOLUTION_TRANSPOSE},{NN_CONVOLUTION_TEMPORAL,20,20,1,1},{NN_LINEAR_RESIZE,100}},
  {{NN_CONVOLUTION_VOLUMETRIC,1,20,5,5,5,1,1,1},{NN_CONVOLUTION_VOLUMETRIC,20,20,3,3,3,1,1,1},{NN_CONVOLUTION_TRANSPOSE},{NN_CONVOLUTION_TEMPORAL,20,20,1,1},{NN_LINEAR_RESIZE,100}}
  }--Model1C

Settings.ModelType  = "convolve"
--Settings.ModelType  = "classic"
for i = 1,table.getn(V) do
  Settings.ModelSize=V[i]
  Settings.ModelName="Model1C"..i
  ModelTrain1D()  
end
