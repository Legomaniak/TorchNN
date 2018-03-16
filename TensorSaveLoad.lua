require 'torch'
require 'paths'
require 'gnuplot'
--obecne nacteni zdrojovych dat

--MyFolder = '/home/legomaniak/Dokumenty/site/Kostky/'
--TrainFileData = 'BB300P10F0000'

function NactiData(cesta)
  if paths.filep(cesta) then
    local file = torch.DiskFile(cesta,'r')
    output = file:readObject()
    file:close()
    return output
  end
end
function UlozData(cesta,data)
    local file = torch.DiskFile(cesta,'w')
    file:writeObject(data)
    file:close()
end

function VytvorData(velikostX,velikostY,velikostZ)
x = torch.DoubleTensor(velikostX,velikostY,velikostZ)
i=0
x:apply(function()
  i = i + 1
  return i
end)
return x
end

function VytvorData(velikostX,velikostY)
x = torch.DoubleTensor(velikostX,velikostY)
i=0
x:apply(function()
  i = i + 1
  return i
end)
return x
end

--UlozData(MyFolder..TrainFileData..'2x3x4',VytvorData(2,3,4))
--UlozData(MyFolder..TrainFileData..'10x5x2',VytvorData(10,5,2))
--UlozData(MyFolder..TrainFileData..'5x10',VytvorData(5,10))
--a = NactiData(MyFolder..TrainFileData)
--print(a)
--velikosti = a:size()
--gnuplot.plot(a[1][1])

