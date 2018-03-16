require 'torch'
require 'HyperMixPCA'
require 'HyperMix'
require 'DataMixerFilter'

--local IS = {10,15,20,30,50}
--local V = {1,3,5,7,9}

--for i = 1,table.getn(IS) do
--  for j = 1,table.getn(V) do
--    HyperMixPCA(V[j],IS[i])
--  end
--end

local V = {1,3,5,7}
local V = {9}

for j = 1,table.getn(V) do
  HyperMix(V[j])
  --DataMixerFilter(V[j])
end