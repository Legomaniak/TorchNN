require 'TensorSaveLoad'
require 'gnuplot'
-- LM --- Datasets

function Dataset(fname)
  
  local convolve = false
  -- initialization
  local dataset = {}
  
  -- logs
  local flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. '/' .. fname .. '.log');
  local plog = logroll.print_logger();
  local log = logroll.combine(flog, plog);
    
  -- set tensors to float to handle data
  --torch.setdefaulttensortype('torch.FloatTensor');
  
  -- log & timer
  
  log.info('Preparing dataset: ' .. fname);    
  local begin = sys.clock();
  
  
  -- initialize sample count
  local totalSamples = 0;
    
  local file = NactiData(settings.listFolder .. fname..'BatchIn');
  -- log
  flog.info('Processing file: ' .. fname);
  local input = file--:transpose(1,3)--:float();

  file = NactiData(settings.listFolder .. fname..'BatchOut');
  local output = file:float();
  
  local min = torch.min(input);
  input = input - min;
  
  local max = torch.max(input);
  input = input / max;
  
  if(convolve) then
    -- dataset initialization
    local delkaVstupu = 1
    local pocetVstupu = 1
    local pocetVystupu = 1
    local delkaKernel = 5
    
    input = input:transpose(1,3)--:float();
    dataset.nSamplesList = torch.Tensor(input:size(1),input:size(2)-delkaKernel+1,input:size(3)-delkaKernel+1)
    
    dataset.nSamplesListOutput = output
    dataset.nSamples = dataset.nSamplesList:size(3)*dataset.nSamplesList:size(2);
    
    --SpatialConvolution
    --conv = nn.SpatialConvolution(pocetVstupu,pocetVystupu,delkaKernel,delkaKernel,1,1,2,2)
    local conv = nn.SpatialConvolution(pocetVstupu,pocetVystupu,delkaKernel,delkaKernel)
    conv.bias:fill(0)
    conv.weight:fill(1/delkaKernel/delkaKernel)

    for i = 1, input:size(1) do        
        local outputConv = conv:forward(input[{{i},{},{}}])
        dataset.nSamplesList[i]=outputConv
    end       
    dataset.nSamplesList = dataset.nSamplesList:transpose(1,3) 
  else 
    --[[
    delkaKernel = 5
    dataset.nSamplesList = torch.Tensor(input:size(1)*delkaKernel*delkaKernel,input:size(2),input:size(3))
    dataset.nSamples = input:size(1)*delkaKernel*delkaKernel;
     delkaKernelPul = (delkaKernel-1)/2
    for i = 1, dataset.nSamplesList:size(1) do  
      for j = delkaKernelPul+1, dataset.nSamplesList:size(2)-delkaKernelPul do   
        for k = delkaKernelPul+1, dataset.nSamplesList:size(3)-delkaKernelPul do  
          for l = 0, delkaKernel-1 do  
            for ll = 0, delkaKernel-1 do  
              dataset.nSamplesList[i+l*delkaKernel+ll][j][k]=input[i][j+l-delkaKernelPul][k+ll-delkaKernelPul]
            end  
          end  
        end      
      end      
    end ]]
    dataset.nSamplesList = input;
    dataset.nSamplesListOutput = output
    dataset.nSamples = input:size(1);
  end  
  if (false) then
    local wavelength = torch.load(settings.listFolder .. 'DataMixedPCA');
    local data = torch.Tensor(dataset.nSamplesList:size(1),dataset.nSamplesList:size(2),wavelength:size(1))
  
    for v = 1, wavelength:size(1), 1 do
     data[{{},{},{v}}] = dataset.nSamplesList:select(3,wavelength[v][1])
    end  
  dataset.nSamplesList = data
  end
  
  --[[dataset.nSamplesList = dataset.nSamplesList[{{},{},{},{5,dataset.nSamplesList:size(4)}}]
  if(settings.gnuploting==1) then
    gnuplot.figure()
    if(dataset.nSamplesListOutput:nDimension()==3) then
    gnuplot.imagesc(dataset.nSamplesListOutput:select(3,1),'color')
    else
    gnuplot.imagesc(dataset.nSamplesListOutput,'color')
    end
    for i = dataset.nSamplesList:size(3)/3, dataset.nSamplesList:size(3),(dataset.nSamplesList:size(3)-dataset.nSamplesList:size(3)/3)/3 do
      gnuplot.figure()
      gnuplot.imagesc(dataset.nSamplesList:select(3,i),'color')
    end
    gnuplot.close()
  end]]
  
  return dataset;
  
end
