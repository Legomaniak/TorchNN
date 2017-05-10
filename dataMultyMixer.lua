require 'TensorSaveLoad'

MyFolder = "Kostky/"
TrainFileData = {'BB300P10F0000', 'BB300P10F7230', 'BB300P10F7730', 'BB300P10F8248', 'BB300P10F9530', 'BB300P10F9640', 'BB300P10F10350', 'BB300P10F10500', 'BB300P10F11660', 'BB300P10Fkonst'}
PocetSouboru = table.getn(TrainFileData)
VelikostX=271
VelikostY=41
VelikostZ=299
Velikost = 100

DataMixedTestIn= torch.Tensor(PocetSouboru*PocetSouboru*Velikost,VelikostY,VelikostZ)
DataMixedTestOut= torch.Tensor(PocetSouboru*PocetSouboru*Velikost,VelikostY)

for i=1,PocetSouboru do
  DataIn = NactiData(MyFolder..TrainFileData[i])
  --DataOut= torch.Tensor(VelikostX,VelikostY):fill(i);
  print(i);
  for p = 1,PocetSouboru do
  DataMix1 = NactiData(MyFolder..TrainFileData[p])
  --DataMix2 = NactiData(MyFolder..TrainFileData[torch.random(PocetSouboru)])  
    for j=1,Velikost do
      vyberDataIn = torch.random(VelikostX)
      vyberDataMix1 = torch.random(VelikostX)
      --vyberDataMix2 = torch.random(VelikostX)
      --DataMixedTestIn[{{(i-1)*Velikost+j},{},{}}]=DataIn:select(1,vyberDataIn)*0.6+DataMix1:select(1,vyberDataMix1)*0.4;
      local ind = (i-1)*PocetSouboru*Velikost+(p-1)*Velikost+j
      DataMixedTestIn[{{ind},{},{}}]=DataIn:select(1,vyberDataIn)*0.6+DataMix1:select(1,vyberDataMix1)*0.4;--+DataMix2:select(1,vyberDataMix2)*0.1;
      --DataMixedTestOut[(i-1)*Velikost+j]=DataOut[vyberDataIn]
      DataMixedTestOut[ind]= i
    end
  end
end
jmeno = 'MultyMix_60_40'
UlozData(MyFolder..jmeno..'BatchTest'..'BatchIn',DataMixedTestIn);
UlozData(MyFolder..jmeno..'BatchTest'..'BatchOut',DataMixedTestOut);

print("Hotovo " .. jmeno)

for i=1,PocetSouboru do
  DataIn = NactiData(MyFolder..TrainFileData[i])
  --DataOut= torch.Tensor(VelikostX,VelikostY):fill(i);
  print(i);
  for p = 1,PocetSouboru do
  DataMix1 = NactiData(MyFolder..TrainFileData[p])
  --DataMix2 = NactiData(MyFolder..TrainFileData[torch.random(PocetSouboru)])  
    for j=1,Velikost do
      vyberDataIn = torch.random(VelikostX)
      vyberDataMix1 = torch.random(VelikostX)
      --vyberDataMix2 = torch.random(VelikostX)
      --DataMixedTestIn[{{(i-1)*Velikost+j},{},{}}]=DataIn:select(1,vyberDataIn)*0.6+DataMix1:select(1,vyberDataMix1)*0.4;
      local ind = (i-1)*PocetSouboru*Velikost+(p-1)*Velikost+j
      DataMixedTestIn[{{ind},{},{}}]=DataIn:select(1,vyberDataIn)*0.7+DataMix1:select(1,vyberDataMix1)*0.3;--+DataMix2:select(1,vyberDataMix2)*0.1;
      --DataMixedTestOut[(i-1)*Velikost+j]=DataOut[vyberDataIn]
      DataMixedTestOut[ind]= i
    end
  end
end
jmeno = 'MultyMix_70_30'
UlozData(MyFolder..jmeno..'BatchTest'..'BatchIn',DataMixedTestIn);
UlozData(MyFolder..jmeno..'BatchTest'..'BatchOut',DataMixedTestOut);

print("Hotovo " .. jmeno)

for i=1,PocetSouboru do
  DataIn = NactiData(MyFolder..TrainFileData[i])
  --DataOut= torch.Tensor(VelikostX,VelikostY):fill(i);
  print(i);
  for p = 1,PocetSouboru do
  DataMix1 = NactiData(MyFolder..TrainFileData[p])
  --DataMix2 = NactiData(MyFolder..TrainFileData[torch.random(PocetSouboru)])  
    for j=1,Velikost do
      vyberDataIn = torch.random(VelikostX)
      vyberDataMix1 = torch.random(VelikostX)
      --vyberDataMix2 = torch.random(VelikostX)
      --DataMixedTestIn[{{(i-1)*Velikost+j},{},{}}]=DataIn:select(1,vyberDataIn)*0.6+DataMix1:select(1,vyberDataMix1)*0.4;
      local ind = (i-1)*PocetSouboru*Velikost+(p-1)*Velikost+j
      DataMixedTestIn[{{ind},{},{}}]=DataIn:select(1,vyberDataIn)*0.8+DataMix1:select(1,vyberDataMix1)*0.2;--+DataMix2:select(1,vyberDataMix2)*0.1;
      --DataMixedTestOut[(i-1)*Velikost+j]=DataOut[vyberDataIn]
      DataMixedTestOut[ind]= i
    end
  end
end
jmeno = 'MultyMix_80_20'
UlozData(MyFolder..jmeno..'BatchTest'..'BatchIn',DataMixedTestIn);
UlozData(MyFolder..jmeno..'BatchTest'..'BatchOut',DataMixedTestOut);

print("Hotovo " .. jmeno)

for i=1,PocetSouboru do
  DataIn = NactiData(MyFolder..TrainFileData[i])
  --DataOut= torch.Tensor(VelikostX,VelikostY):fill(i);
  print(i);
  for p = 1,PocetSouboru do
  DataMix1 = NactiData(MyFolder..TrainFileData[p])
  --DataMix2 = NactiData(MyFolder..TrainFileData[torch.random(PocetSouboru)])  
    for j=1,Velikost do
      vyberDataIn = torch.random(VelikostX)
      vyberDataMix1 = torch.random(VelikostX)
      --vyberDataMix2 = torch.random(VelikostX)
      --DataMixedTestIn[{{(i-1)*Velikost+j},{},{}}]=DataIn:select(1,vyberDataIn)*0.6+DataMix1:select(1,vyberDataMix1)*0.4;
      local ind = (i-1)*PocetSouboru*Velikost+(p-1)*Velikost+j
      DataMixedTestIn[{{ind},{},{}}]=DataIn:select(1,vyberDataIn)*0.9+DataMix1:select(1,vyberDataMix1)*0.1;--+DataMix2:select(1,vyberDataMix2)*0.1;
      --DataMixedTestOut[(i-1)*Velikost+j]=DataOut[vyberDataIn]
      DataMixedTestOut[ind]= i
    end
  end
end
jmeno = 'MultyMix_90_10'
UlozData(MyFolder..jmeno..'BatchTest'..'BatchIn',DataMixedTestIn);
UlozData(MyFolder..jmeno..'BatchTest'..'BatchOut',DataMixedTestOut);

print("Hotovo " .. jmeno)