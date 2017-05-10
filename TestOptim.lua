require 'OptimPP'
require 'ReadDataset'
require 'TestNN'
require 'nn'

function TestOptim(modelN)
  -- TRAINING
  local etime = sys.clock();  


  -- logs & export model
  --plog.info("Loading: " .. modelN);
  modelC = torch.load(settings.outputFolder .. settings.modFolder .. modelN .. ".mod");


  -- EVALUATION
  modelC:evaluate();

  local errorBatch=-1;
  for i = 1,table.getn(settings.listsTest), 1 do            
    local input, output = ReadDataset(settings.listsTest[i]);
    local err = TestNN(modelC, input, output, 10);
    log.info("Model " .. modelN .. " - err = " .. err); 
    --print("Model " .. modelN .. " - err = " .. err); 
    errorBatch = err;
  end

  for i = 1,table.getn(settings.listsTestX), 1 do            
    local input, output = ReadDataset(settings.listsTestX[i]);
    local err = TestNN(modelC, input, output, settings.outputSize);
    log.info("Set " .. settings.listsTestX[i] .. " - err = " .. err);  
    --print("Set " .. settings.listsTestX[i] .. " - err = " .. err);  
  end
    
  log.info("Test " .. modelN .. " completed in " .. sys.clock() - etime); 
  --print("Test " .. modelN .. " completed in " .. sys.clock() - etime);  
  return errorBatch
end