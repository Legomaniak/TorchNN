require 'torch'
require 'paths'
require 'xlua'
require 'math'
require 'nn'
require 'logroll'
require 'gnuplot'
require 'optim'
require 'dp'

MyFolder = '/home/legomaniak/Dokumenty/site2/'
TrainFileData = 'trainData.txt'
DL_Sufix = '.ShortTensor'


function NactiData(cesta)
  cestaT = cesta..DL_Sufix
  if paths.filep(cestaT) then
    file1 = torch.DiskFile(cestaT,'r')
    output1 = file1:readObject()
    file1:close()
    return output1
  else
    if paths.filep(cesta) == false then return nil end    
    file = io.open(cesta,'r')
    --outputTrain = file:readDouble(Delka)
    output = file:read("*a")
    file:close()
    --output = output:split(" \n")
    torch.DoubleTensor(output:split(" "))
    --outputTrainMax=torch.Tensor(outputTrain):max()+1
    
    file = torch.DiskFile(cestaT,'w')
    file:writeObject(output)
    file:close()
    return output
  end
end


a = NactiData(MyFolder..TrainFileData)
print(a)