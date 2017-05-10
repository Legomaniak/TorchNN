
function Settings()
    
  local settings = {};
  
  settings.lists = {'NewDataMixed'}; 
  settings.listsTest = {'NewBatchTest'}; 

  settings.listsTestX = {'MultyMix_60_40BatchTest','MultyMix_70_30BatchTest','MultyMix_80_20BatchTest','MultyMix_90_10BatchTest'}; 

 
  settings.learningRate = 0.08;--0.08;
  settings.activationFunction = "relu";      -- relu / tanh / sigmoid
  
  
  settings.startEpoch = 0
  settings.noEpochs = 50
  settings.saveEpoch = 0
  settings.saveEpochFull = 1
  
  -- other settings
  settings.gnuploting = 0;
  settings.cuda = 0;
  settings.shuffle = 1;
  settings.exportNNET = 0;
  settings.drawERRs = 1;
  settings.inputView = 0;      
  
  settings.listFolder = "Kostky/"; 
  --settings.modelName = "TestModelName";
  settings.modelName = "ChemModelName";
  settings.outputFolder = "outputFolder/" .. settings.modelName;
  settings.statsFolder = "/stats/";
  settings.logFolder = "/log/";
  settings.modFolder = "/mod/";
  settings.logPath = "settings.log";

  -- log
  --flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. settings.logPath);
  --flog.info(settings);
  
  -- create output folders
  --os.execute("mkdir -p " .. settings.outputFolder .. settings.statsFolder);
  --os.execute("mkdir -p " .. settings.outputFolder .. settings.modFolder);
  
  return settings;
    
end
