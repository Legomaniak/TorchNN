require 'TensorSaveLoad'
require 'gnuplot'
function ReadDataset(fname)
  local DoPCA = false
    
  --local file = NactiData(settings.listFolder .. fname..'BatchIn');
  local file = NactiData(settings.listFolder .. fname);
  local input = file--:float();

  --file = NactiData(settings.listFolder .. fname..'BatchOut');
  file = NactiData(settings.listFolder .. fname..'_OUT');
  local output = file--:float();
  
  local min = torch.min(input);
  input = input - min;
  
  local max = torch.max(input);
  input = input / max;
        
  if (DoPCA) then
    local wavelength = torch.load(settings.listFolder .. 'DataMixedPCA');
    local data = torch.Tensor(dataset.nSamplesList:size(1),dataset.nSamplesList:size(2),wavelength:size(1))
  
    for v = 1, wavelength:size(1), 1 do
     data[{{},{},{v}}] = input:select(3,wavelength[v][1])
    end  
    input = data;
  end
    input = input:transpose(1,3);
    input = input:transpose(2,3);
  return input,output;  
end

function ReadDataAllOld(list)
  local t ={}
  local pocet = table.getn(list)
  table.insert(t,pocet)
  local file = NactiData(Settings.ListFolder .. list[1]);
  for i = 1,file:dim() do
    table.insert(t,file:size(i))
  end
  local data = torch.Tensor(torch.LongStorage(t))
  data[1] = file
  for i = 2,pocet do
    file = NactiData(Settings.ListFolder .. list[i]);
    data[i] = file
  end
  return data
end

--function ReadDataAll(list)
--  local t ={}
--  local pocet = table.getn(list)
--  table.insert(t,pocet)
--  local file = NactiData(list[1]);
--  for i = 1,file:dim() do
--    table.insert(t,file:size(i))
--  end
--  local data = torch.Tensor(torch.LongStorage(t))
--  data[1] = file
--  for i = 2,pocet do
--    file = NactiData(list[i]);
--    data[i] = file
--  end
--  return data
--end
function ReadDataAll(list)
  local data ={}
  local pocet = table.getn(list)
  for i = 1,pocet do
   local file = NactiData(list[i]);
   table.insert(data,file)
  end
  return data
end
--function ReadDir(dir)  
--Lists = {}
--listIdent = 'norm'
--ListsOut = {}
--listOutIdent = 'OUT'

--for file in lfs.dir(dir) do
--    if lfs.attributes(dir..file,"mode") == "file" then 
--      if string.find(file,listIdent) then
--        table.insert(Lists,dir..file);
--      elseif string.find(file,listOutIdent) then
--        table.insert(ListsOut,dir..file);
--      end
--    end
--end
--return Lists,ListsOut
--end

function ReadDir(dir,listIdent,listOutIdent)  
Lists = {}
ListsOut = {}

for file in lfs.dir(dir) do
    if lfs.attributes(dir..file,"mode") == "file" then 
      if string.find(file,listIdent) then
        table.insert(Lists,dir..file);
      elseif string.find(file,listOutIdent) then
        table.insert(ListsOut,dir..file);
      end
    end
end
return Lists,ListsOut
end