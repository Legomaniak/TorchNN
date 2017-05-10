require 'TensorSaveLoad'
-- LM --- Stats

function Stats(fname)
  
  -- initialization
  local stats = {}
  
  -- logs
  flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. '/stats.log');
  plog = logroll.print_logger();
  log = logroll.combine(flog, plog);

  -- set tensors to float to calculate stats
  torch.setdefaulttensortype('torch.FloatTensor');
  
  -- log & timer
  log.info('Computing mean and variance for ' .. fname);
  local begin = sys.clock();
  
  -- stats initialization
  stats.mean = 1;
  stats.var = 1;
  stats.size = 0;
  
  -- load filelist
  local fvec = NactiData(settings.listFolder .. fname)--info soubor
  
  -- log
  flog.info('Processing file: ' .. fname);
  
  local nSamples, sampPeriod, sampSize, parmKind, data
        
        
--predvypocet v matlabu
  -- compute global stats
  stats.size = fvec:size();
  stats.nSamples = torch.sum(torch.Tensor(fvec:size()));
  stats.sum = torch.sum(fvec);
  stats.mean = torch.mean(fvec);
  stats.min = torch.min(fvec);
  stats.max = torch.max(fvec);
  stats.var = torch.var(fvec);
  --stats.var = torch.sum(torch.pow(fvec, 2))/stats.nSamples;
  --flog.info('var ' .. var);
  
  -- compute global number of frames
  --stats.nSamples = stats.nSamples + fvec:size(1);

  -- compute global stats
  --stats.mean:div(stats.nSamples);
  --stats.var:div(stats.nSamples);
  --stats.var:add(-torch.pow(stats.mean, 2));
  --stats.var:sqrt();
  --stats.mean = stats.mean:float();
  --stats.var = stats.var:float();
  
  -- log time needed for computation
  log.info('Mean and variance completed in ' .. sys.clock() - begin);

  return stats;
  
end
