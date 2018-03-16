require 'TensorSaveLoad'
require 'gnuplot'
require 'lfs'

lists = {}
listsNorm = {}
listsTest = {}
listsTestX = {}
listFolder = "NoveKostky";
fi = 1; 
show = true;
save = false;

for file in lfs.dir(listFolder) do
    if lfs.attributes(listFolder.."/"..file,"mode") == "file" then 
      --print("found file, "..file)
      if string.find(file,"OUT") then
        table.insert(listsTest,file);
      elseif string.find(file,"norm") then
        table.insert(listsNorm,file);
      else 
        table.insert(lists,file);
      end
    end
end
for nolistsTest = 1,table.getn(listsTest) do
    local data = NactiData(listFolder..'/'..listsTest[nolistsTest]);
    if show then
      gnuplot.figure(i)
      gnuplot.imagesc(data,'color')
      fi=fi+1;
    end
end
CelaData = {}
for nolists = 1,table.getn(lists) do
    local data = NactiData(listFolder..'/'..lists[nolists]);
    table.insert(CelaData,data)
    if show then
      gnuplot.figure(i)
      gnuplot.imagesc(data:select(1,10))
      fi=fi+1;
    end
    data[{{1},{},{}}]=data:select(1,2)--[[
    for x = 1,data:size(2) do
      for y = 1,data:size(3) do
        local d = data:select(2,x):select(2,y)
        local dmin = d:min()
        data[{{},{x},{y}}]=(d-dmin)/(d:max()-dmin)
      end
    end]]
    if show then
      gnuplot.figure(i)
      gnuplot.plot(data:select(2,data:size(2)/2):select(2,data:size(3)/2))
      fi=fi+1;
    end
    if save then 
      UlozData(listFolder..'/'..lists[nolists]..'norm',data)
    end
end
print(table.getn(CelaData))
