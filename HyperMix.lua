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
require 'Helper'
require 'TensorSaveLoad'
require 'MyPCA'

SFname="HyperKostky"
OFName="IndianPinesNew"
ListsName="Indian_pines.in"
listIdent = '.in'
ListsTestName="Indian_pines.out"
listTestIdent = '.out'

function HyperMix(V)
  -- initialize settings
  Settings = {}
    
  Settings.SourceFolder = SFname.."/"
  Settings.OutputFolder = "Out/"..OFName.."_"..V.."/"
  Settings.Lists = {ListsName}
  Settings.ListsTest = {ListsTestName}

  for file in lfs.dir(Settings.SourceFolder) do
      if lfs.attributes(Settings.SourceFolder.."/"..file,"mode") == "file" then 
        if string.find(file,listTestIdent) then
          table.insert(Settings.ListsTest,file);
        elseif string.find(file,listIdent) then
          table.insert(Settings.Lists,file);
        end
      end
  end

  Settings.TrainFolder = Settings.OutputFolder..'/'.."Train/"; 
  Settings.TestFolder = Settings.OutputFolder..'/'.."Test/"; 
  Settings.TestXFolder = Settings.OutputFolder..'/'.."TestX/"; 
  CheckFolder(Settings.OutputFolder)
  CheckFolder(Settings.TrainFolder)
  CheckFolder(Settings.TestFolder)
  CheckFolder(Settings.TestXFolder)

  Settings.ModelName="Model"

  -- NN settings
  Settings.OutputSize=16--+1
  Settings.BatchSizeX=1  
  Settings.BatchSizeY=V  
  Settings.BatchSizeZ=V 
  print("BatchSize "..Settings.BatchSizeZ)

  Settings.BatchN=10
  Settings.EpochN=3
  Settings.LearningRate=0.08
  Settings.BatchSize=50
  Settings.MaxIter=3000
  
  Settings.ModelType="classic"
  
  --Parse position
  TrainingRatio = 1/20

  show = false;
  save = true;

  for ListN = 1,1 do
    local dataOut = NactiData(Settings.SourceFolder .. '/' .. Settings.ListsTest[ListN])
    if show then
      gnuplot.figure(ListN)
      gnuplot.imagesc(dataOut,'color')
    end  
    local dataIn = NactiData(Settings.SourceFolder .. '/' .. Settings.Lists[ListN])
    dataIn=dataIn:transpose(1,3)
    --dataIn=dataIn:transpose(1,3):transpose(1,2)
    local s = dataIn:size()
    --experiment loading
    local SizeX=math.floor(Settings.BatchSizeX/2)
    local SizeY=math.floor(Settings.BatchSizeY/2)
    local SizeZ=math.floor(Settings.BatchSizeZ/2)
    
    local ssize = (s[1]-2*SizeY)*(s[2]-2*SizeZ)
    local TrainSize = 0
    local TestSize = 0 
    Settings.InputSize=s[3]
    
    for o = 1,Settings.OutputSize do
      for i = 1+SizeY,s[1]-SizeY do
        for j = 1+SizeZ,s[2]-SizeZ do
          if dataOut[i][j]==o then
            if (TrainSize+TestSize)%(1/TrainingRatio)==0 then
              TrainSize=TrainSize+1
            else
              TestSize=TestSize+1
            end  
          end
        end
      end
    end
    slog = io.open(Settings.OutputFolder .. 'Settings.lua', "w")

    slog:write("--Start mixing\n")
    slog:write("--With parameters:\n")
    slog:write("Settings.ModelName = \""..Settings.ModelName.."\"\n")
    slog:write("Settings.SourceFolder = \""..Settings.SourceFolder.."\"\n")
    slog:write("Settings.OutputFolder = \""..Settings.OutputFolder.."\"\n")

    -- NN settings
    slog:write("Settings.InputSize = "..Settings.InputSize.."\n")
    slog:write("Settings.OutputSize = "..Settings.OutputSize.."\n")
    slog:write("Settings.BatchSizeX = "..Settings.BatchSizeX.."\n")
    slog:write("Settings.BatchSizeY = "..Settings.BatchSizeY.."\n")
    slog:write("Settings.BatchSizeZ = "..Settings.BatchSizeZ.."\n")
    slog:write("--Add NN settings\n")
    slog:write("Settings.BatchN = "..Settings.BatchN.."\n")
    slog:write("Settings.EpochN = "..Settings.EpochN.."\n")
    slog:write("Settings.LearningRate = "..Settings.LearningRate.."\n")
    slog:write("Settings.BatchSize = "..Settings.BatchSize.."\n")
    slog:write("Settings.MaxIter = "..Settings.MaxIter.."\n")

    slog:write("--Display\n")
    slog:write("Settings.DispEpoch = false\n")
    slog:write("Settings.DispBatch = false\n")
    slog:write("Settings.DispFile = true\n")
    slog:write("Settings.DispSave = false\n")
    slog:write("Settings.DispIter = true\n")

    slog:write("--Model\n")
    slog:write("--classic,convolve,load\n")
    slog:write("Settings.ModelType = \""..Settings.ModelType.."\"\n")
    if Settings.ModelType=="classic" then
      slog:write("Settings.ModelSize = {10,20,30}\n")
    end
    --Parse position
    slog:write("--TrainingRatio = "..TrainingRatio.."\n")
    print("TrainSize "..TrainSize)
    slog:write("Settings.TrainSize = "..TrainSize.."\n")
    print("TestSize "..TestSize)
    slog:write("Settings.TestSize = "..TestSize.."\n")
    slog:close()
      
    dataInTest=torch.Tensor(TestSize,Settings.BatchSizeX,Settings.BatchSizeY,Settings.BatchSizeZ,Settings.InputSize):fill(0)  
    dataInTrain=torch.Tensor(TrainSize,Settings.BatchSizeX,Settings.BatchSizeY,Settings.BatchSizeZ,Settings.InputSize):fill(0) 
   
    --s = dataOut:size();
    dataOutTest=torch.Tensor(TestSize,Settings.OutputSize):fill(0)
    dataOutTrain=torch.Tensor(TrainSize,Settings.OutputSize):fill(0) 
      
    local d1=0
    local d2=0
    for o = 1,Settings.OutputSize do
        --gnuplot.figure(o)
      for i = 1+SizeY,s[1]-SizeY do
        for j = 1+SizeZ,s[2]-SizeZ do
          if dataOut[i][j]==o then
            --input
            local input = torch.Tensor(Settings.BatchSizeX,Settings.BatchSizeY,Settings.BatchSizeZ,Settings.InputSize):fill(0)
            input[1] = dataIn:sub(i-SizeY,i+SizeY,j-SizeZ,j+SizeZ)
            --input[1] = (dataIn:sub(i-SizeY,i+SizeY,j-SizeZ,j+SizeZ):double():resize(Settings.BatchSizeY*Settings.BatchSizeZ,s[3])*dataInPCA):resize(Settings.BatchSizeY*Settings.BatchSizeZ*Settings.InputSize)          
            --norm
            input = input-input:min()
            input = input/input:max()
            --input = input-input:mean()
            --resize
            --input = input:resize(Settings.BatchSizeX,Settings.BatchSizeY,Settings.BatchSizeZ,Settings.InputSize)         
            --input[1] = MyPCA(dataIn:sub(i-SizeY,i+SizeY,j-SizeZ,j+SizeZ):resize(Settings.BatchSizeY*Settings.BatchSizeZ,s[3]):t():double(),Settings.InputSize):resize(Settings.BatchSizeY*Settings.BatchSizeZ*Settings.InputSize)
            --output
            local output = torch.Tensor(Settings.OutputSize):fill(0)
            output[o]=1
            if (d1+d2)%(1/TrainingRatio)==0 then
              d1=d1+1
              dataInTrain[d1]=input
              dataOutTrain[d1]=output
    --          print("d1 "..d1)
            else
              d2=d2+1
              dataInTest[d2]=input
              dataOutTest[d2]=output 
    --          print("d2 "..d2)
            end  
          end
        end
      end
    end
    
    --Save data
    if save then 
      --train
      UlozData(Settings.TrainFolder..'/All.in',dataInTrain)
      UlozData(Settings.TrainFolder..'/All.out',dataOutTrain)
  --    UlozData(Settings.TrainFolder..'/'..Settings.Lists[ListN],dataInTrain)
  --    UlozData(Settings.TrainFolder..'/'..Settings.ListsTest[ListN],dataOutTrain)
      --test
      UlozData(Settings.TestFolder..'/All.in',dataInTest)
      UlozData(Settings.TestFolder..'/All.out',dataOutTest)
  --    UlozData(Settings.TestFolder..'/'..Settings.Lists[ListN],dataInTest)
  --    UlozData(Settings.TestFolder..'/'..Settings.ListsTest[ListN],dataOutTest)
      --testX
  --      UlozData(Settings.TestXFolder..'/'..Settings.Lists[ListN],dataInTestX)
  --      UlozData(Settings.TestXFolder..'/'..Settings.ListsTest[ListN],dataOutTestX)
    end
  end
end