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
cmd:option('-f','IndianPines1_1','Folder')
params = cmd:parse(arg)

CestaRoot = "Out/"--/home/legomaniak/site3D/Out/
--CestaFolders = {"IndianPines1_1"}
CestaFolder = params.f
CestaSubFolders = {"/Train","/Test","/TestX"}

ListsInTrain = {}
listsInTest = {}
listsInTestX = {}
ListsOutTrain = {}
listsOutTest = {}
listsOutTestX = {}
--for CestaFolder in pairs(CestaFolders) do
for CestaSubFolder in pairs(CestaSubFolders) do
--listFolder=CestaRoot..CestaFolders[CestaFolder]..CestaSubFolders[CestaSubFolder]
listFolder=CestaRoot..CestaFolder..CestaSubFolders[CestaSubFolder]
for file in lfs.dir(listFolder) do
    if lfs.attributes(listFolder.."/"..file,"mode") == "file" then 
      if string.find(file,".in") then
        if CestaSubFolder==1 then
          table.insert(ListsInTrain,file);
        elseif CestaSubFolder==2 then
          table.insert(listsInTest,file);
        else 
          table.insert(listsInTestX,file);
        end
      elseif string.find(file,".out")  then
        if CestaSubFolder==1 then
          table.insert(ListsOutTrain,file);
        elseif CestaSubFolder==2 then
          table.insert(listsOutTest,file);
        else 
          table.insert(listsOutTestX,file);
        end
      end
    end
end
end
end
function ShowListData(ListIn,ListOut)
    local DataIn = NactiData(ListIn):squeeze();    
    local DataOut = NactiData(ListOut);
    local t = {}
    for i=1,DataIn:size(1) do
      table.insert(t,{DataIn:select(1,i),'-'})
    end
    
    gnuplot.figure()
    gnuplot.title(ListIn)
    gnuplot.plot(t)
    
    gnuplot.figure()
    gnuplot.title(ListOut)
    gnuplot.imagesc(DataOut,'color')
end
function ShowOutputData(ListIn,ListOut)
    local DataIn = NactiData(ListIn):squeeze();    
    local DataOut = NactiData(ListOut);
    for o=1,DataOut:size(2) do  
      local t = {}
      for i=1,DataIn:size(1) do
        if DataOut[i][o]==1 then
          local d = DataIn:select(1,i)
          local s = d:size()
          ssize = 1;
          for i = 1,s:size() do
            ssize = ssize * s[i]
          end
        table.insert(t,{d:resize(ssize,1) ,'-'})
        end
      end      
      if table.getn(t)>0 then
        gnuplot.figure()
        gnuplot.title(ListIn)
        gnuplot.plot(t)   
      end
    end
    gnuplot.figure()
    gnuplot.title(ListOut)
    gnuplot.imagesc(DataOut,'color')
end

for noLists in pairs(ListsInTrain) do
  listFolder=CestaRoot..CestaFolder..CestaSubFolders[1].."/"
  --ShowListData(listFolder..ListsInTrain[noLists],listFolder..ListsOutTrain[noLists])
  ShowOutputData(listFolder..ListsInTrain[noLists],listFolder..ListsOutTrain[noLists])
end