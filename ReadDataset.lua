require 'TensorSaveLoad'
require 'gnuplot'
function ReadDataset(fname)
  local DoPCA = false
    
  local file = NactiData(settings.listFolder .. fname..'BatchIn');
  local input = file--:float();

  file = NactiData(settings.listFolder .. fname..'BatchOut');
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
    
  return input,output;
  
end
