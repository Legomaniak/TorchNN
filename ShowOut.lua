-- general libraries
require 'cunn'
require 'cutorch'

require 'paths'
require 'xlua'
require 'math'
require 'logroll'
require 'gnuplot'
require 'lfs'

-- program requires
require 'Helper'
require 'TensorSaveLoad'


cmd = torch.CmdLine()
cmd:option('-f','IndianPinesNew_5','Folder')
cmd:option('-m','Model1C1','Model')
params = cmd:parse(arg)

ListFolder = params.f
ModelName = params.m
--Source data
SFname="HyperKostky"
ListsName="Indian_pines.in"
listIdent = '.in'
ListsTestName="Indian_pines.out"
listTestIdent = '.out'

SourceFolder = SFname.."/"
Lists = {ListsName}
ListsTest = {ListsTestName}


--NN
CestaRoot = "Out/"--/home/legomaniak/site3D/Out/
--ListFolder = "IndianPinesNew_3"
--ModelName = "Model3D3"
OutputFolderNN = CestaRoot..ListFolder.."/"..ModelName.."/"
StatsFolder = "Stats"
LogFolder = "Log/"
ModFolder = "Mod/"

Settings={}
os.rename(OutputFolderNN..StatsFolder.."/Settings.txt", OutputFolderNN..StatsFolder.."/Settings.lua")
require(OutputFolderNN..StatsFolder..'.Settings')

Settings.BatchSize=2

for file in lfs.dir(SourceFolder) do
    if lfs.attributes(SourceFolder.."/"..file,"mode") == "file" then 
      if string.find(file,listTestIdent) then
        table.insert(ListsTest,file);
      elseif string.find(file,listIdent) then
        table.insert(Lists,file);
      end
    end
  end


for ListN = 1,1 do
  local dataIn = NactiData(SourceFolder .. '/' .. Lists[ListN])
  dataIn=dataIn:transpose(1,3)
  --dataIn=dataIn:transpose(1,3):transpose(1,2)
  local s = dataIn:size()
--  --norm  
--  for x = 1,s[1] do
--  for y = 1,s[2] do
--    local input = dataIn:select(1,x):select(1,y)
--    input = input-input:min()
--    input = input/input:max()
--    dataIn[x][y]=input
--  end
--  end
  
  local dataOutR = NactiData(SourceFolder .. '/' .. ListsTest[ListN])
  
  local SizeX=math.floor(Settings.BatchSizeX/2)
  local SizeY=math.floor(Settings.BatchSizeY/2)
  local SizeZ=math.floor(Settings.BatchSizeZ/2)
    
  local dataOut = torch.Tensor(s[1],s[2]):fill(0)
  --load model
  local model = torch.load(OutputFolderNN .. ModFolder .. "/" .. ModelName .. "Min.mod")
  print(model)
  model:replace(function(module)
   if torch.typename(module) == 'nn.Reshape' then
      return nn.Reshape(Settings.BatchSize,module.size[2])
   else
      return module
   end
  end)  
  model:cuda()  -- convert model to CUDA
  print(model)
  
  local err = torch.Tensor(Settings.OutputSize,Settings.OutputSize):fill(0)
  local all = torch.Tensor(Settings.OutputSize):fill(0)
  
  for x = 1+SizeY,s[1]-SizeY do
  for y = 1+SizeZ,s[2]-SizeZ do
    if dataOutR[x][y] >0 then
      --local input = dataIn:select(1,x):select(1,y):double
      --input
      local input = torch.Tensor(Settings.BatchSize,Settings.BatchSizeX,Settings.BatchSizeY,Settings.BatchSizeZ,Settings.InputSize):fill(0)
      input[1][1] = dataIn:sub(x-SizeY,x+SizeY,y-SizeZ,y+SizeZ)
      input[2][1] = dataIn:sub(x-SizeY,x+SizeY,y-SizeZ,y+SizeZ)
      --local input = dataIn:sub(x-SizeY,x+SizeY,y-SizeZ,y+SizeZ)
      
      --norm
      input = input-input:min()
      input = input/input:max()
--      input = input-torch.mean(input:double())
      
      input = input:cuda()
      
      local output = model:forward(input)
      output=output[1]
      --local mxn, mx = output:max(2)
      local mxn, mx = output:max(1)
      local mxs = mx:squeeze()
      dataOut[x][y]=mxs
      local mxr = dataOutR[x][y]
      
      if mxs~=mxr then
       err[mxr][mxs] = err[mxr][mxs]+1
      end
      all[mxr] = all[mxr] + 1
    end
  end
end

  for i=1,err:size(1) do
    if all[i]==0 then
      err[i]=0
    else
      --err[i]=100*err[i]:sum()/all[i]
      err[i]=100*err[i]/all[i]
    end
  end
  print("Errors")
  print(err:sum(2))
  print("Average Error "..err:sum()/err:size(1).."%")
  gnuplot.imagesc(dataOut,'color')
--  gnuplot.figure()
--  gnuplot.imagesc(dataOutR,'color')
 end
