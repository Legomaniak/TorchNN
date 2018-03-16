-- general libraries
require 'torch'
require 'paths'
require 'nn'
require 'gnuplot'
require 'lfs'

-- program requires
require 'Helper'
require 'TensorSaveLoad'

cmd = torch.CmdLine()
cmd:option('-f','IndianPinesNew_3','Folder')
cmd:option('-m','Model2D','Model')
params = cmd:parse(arg)

CestaRoot = "Out/"--/home/legomaniak/site3D/Out/
--CestaFolders = {"IndianPinesNew_1"}
CestaFolder = params.f--"IndianPinesNew_1"
CestaName = params.m--"Model2D"
CestaNames = {}
for i=1,20 do
  CestaNames[i]=CestaName..i
end
CestaFile = "/Mod/"
CestaFileEx = ".err"
i = 0
--for CestaFolder in pairs(CestaFolders) do
  for CestaName in pairs(CestaNames) do
    --Cesta = CestaRoot..CestaFolders[CestaFolder].."/"..CestaNames[CestaName]..CestaFile..CestaNames[CestaName]..CestaFileEx
    Cesta = CestaRoot..CestaFolder.."/"..CestaNames[CestaName]..CestaFile..CestaNames[CestaName]..CestaFileEx
    print(Cesta)
      if paths.filep(Cesta) then
    data = torch.load(Cesta)
    gnuplot.figure(i)
    gnuplot.title(CestaFolder.."/"..CestaNames[CestaName])
    gnuplot.imagesc(data)
    i=i+1
    print("Minimum "..torch.min(data))
    print("Maximum "..torch.max(data))
  else print("Neplatna cesta")
  end
  end
--end