-------------------------------------------------------------------------------
--
-- Active Training Image Creation (Round 2)
--
-- Description:
-- > enforces the exact same number of human supervision labels, no batch
-- > round 1 of training produces the final synth images
-- > round 2 uses human labels on those images
-- > use half of real pairs for bootstrapping
--
-- Note: Search for the "MOD" tag for list of paths to change
-- 
-------------------------------------------------------------------------------

require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'image'
require 'pl'
require 'paths'
require 'hdf5'
require 'loadcaffe'
require 'gnuplot'
require 'stn'
require 'distributions'


--------------------------
-- LAYERS :: Attr2Image --
--------------------------
require 'libs.attr2img.layers.LinearWP'
require 'libs.attr2img.layers.LinearGaussian'
require 'libs.attr2img.layers.LinearMix'
require 'libs.attr2img.layers.LinearMix2'
require 'libs.attr2img.layers.Reparametrize'
require 'libs.attr2img.layers.GaussianCriterion'
require 'libs.attr2img.layers.KLDCriterion'
require 'libs.attr2img.utils.scaled_zap50k'
optim1_utils = require 'libs.attr2img.utils.adam_v2'


-----------------------
-- LAYERS :: DeepSTN --
-----------------------
require 'libs.deepstn.utils'
require 'libs.deepstn.layers.IndexPrime'
require 'libs.deepstn.layers.MeanSub'
require 'libs.deepstn.layers.AddConstantT'
require 'libs.deepstn.layers.MulConstantT'


----------------------------------
-- INPUTS :: Control Parameters --
----------------------------------

cmd = torch.CmdLine()
cmd:option('-attrID', 1, 'Attribute ID')
cmd:option('-trainIter', 100, '# of Training Epoches')
cmd:option('-expName', 'attic_main', 'Experiment Base Name')

cmd:option('-realFact', 1, 'Real Factor')
cmd:option('-synthFact', 1, 'Synth Factor')
cmd:option('-pregenLR', 0.001, 'Pre-Generator Learning Rate')
cmd:option('-pregenLRD', 0.0001, 'Pre-Generator Learning Rate Decay')
cmd:option('-pregenWD', 0.01, 'Pre-Generator Weight Decay (Regularization)')
cmd:option('-pregenMOM', 0.9, 'Pre-Generator Momentum')

cmd:option('-saveModel', 0, 'Save Final Model')
cmd:option('-saveModelGap', 100, 'Gaps between saving Trained Model File')
cmd:option('-saveLogs', 1, 'Save Performance Log (Flag)')
cmd:option('-saveSynthImg', 0, 'Save Synthetic Images (Flag)')
cmd:option('-saveSynthImgGap', 10, 'Gaps between saving Synthetic Images')

params = cmd:parse(arg)

local attr_id = tonumber(params.attrID)
local num_train_epoch = tonumber(params.trainIter)
local exp_name_full = params.expName

local real_fact = tonumber(params.realFact)
local synth_fact = tonumber(params.synthFact)
local pregen_lr = tonumber(params.pregenLR)
local pregen_lrd = tonumber(params.pregenLRD)
local pregen_wd = tonumber(params.pregenWD)
local pregen_mom = tonumber(params.pregenMOM)

local save_model = tonumber(params.saveModel)
local save_model_gap = tonumber(params.saveModelGap)
local save_logs = tonumber(params.saveLogs)
local save_synth_images = tonumber(params.saveSynthImg)
local save_synth_images_gap = tonumber(params.saveSynthImgGap)


-- Visual Display of Inputs
print('Attr #: ' .. attr_id)
print('# of Epoch: ' .. num_train_epoch)

print('Real Factor: ' .. real_fact)
print('Synth Factor: ' .. synth_fact)
print('PG Learning Rate: ' .. pregen_lr)
print('PG Learning Rate Decay: ' .. pregen_lrd)
print('PG Weight Decay: ' .. pregen_wd)
print('PG Momentum: ' .. pregen_mom)

print('Save Model Flag: ' .. save_model)
print('Save Model Gap: ' .. save_model_gap)
print('Save Logs: ' .. save_logs)
print('Save Images:' .. save_synth_images)
print('Save Images Gap: ' .. save_synth_images_gap)

print('Experiment Name: ' .. exp_name_full .. '\n')


--------------------------
-- INPUTS :: Attr2Image --
--------------------------
opts1 = {}
opts1.model_dir = 'learned_model/pretrained/zap50k_attr2img/'  -- MOD: path to pretained image generator model
opts1.model_name = 'net-epoch-100.t7'
opts1.stats_file = 'libs/attr2img/zap50k_train_stats.t7'

opts1.test_dir = 'images/'


-----------------------
-- INPUTS :: DeepSTN --
-----------------------
opts2 = {}

opts2.lr = 0.001
opts2.lrlocal = 0.1
opts2.batchSize = 25
opts2.scaleInit = 0.4
opts2.scaleRatio =  0.05
opts2.numIter = num_train_epoch
opts2.saveFreq = model_save_freq
opts2.modelType = 2   -- always train combined model (2)

opts2.attrNum = attr_id
opts2.attrNameAll = {'casual', 'comfort', 'simple', 'sporty', 'colorful', 'durable', 'supportive', 'bold', 'sleek', 'open'}
opts2.attrName = opts2.attrNameAll[opts2.attrNum]

opts2.setupName = exp_name_full

opts2.modelFile = 'libs/deepstn/arch_deepSTN.lua'   -- mode definition for DeepSTN

opts2.dataDirPath = ''  -- MOD: path to pretained image generator model

-- MOD: Add labeled synth pairs to original real pairs
opts2.trainDataReal = 'train-comb-real-sudo-' .. opts2.attrName .. '.h5'   -- real + labeled synth training pairs
opts2.testDataReal = 'test-zap50k-' .. opts2.attrName .. '.h5'             -- real testing pairs
opts2.valDataReal = 'val-zap50k-' .. opts2.attrName .. '.h5'               -- real validation pairs

opts2.pretrainModelName = 'learned_model_200.dat'   -- one round localization for DeepSTN

-- Pretrained Alexnet Path
opts2.alexnet_model_path = 'libs/deepstn/bvlc_caffenet/bvlc_reference_caffenet.caffemodel'
opts2.alexnet_prototxt_path = 'libs/deepstn/bvlc_caffenet/deploy.prototxt'

-- Real Pairs Data File
opts2.train_data_real = opts2.dataDirPath .. '/new_shoes_full_' .. opts2.attrName .. '/' .. opts2.trainDataReal
opts2.test_data_real = opts2.dataDirPath .. '/new_shoes_full_' .. opts2.attrName .. '/' .. opts2.testDataReal
opts2.val_data_real = opts2.dataDirPath .. '/new_shoes_full_' .. opts2.attrName .. '/' .. opts2.valDataReal

-- Output directory path
opts2.output_dir_path = 'models/' .. opts2.setupName

-- Localization network path
opts2.loc_network_path = 'models/pretrained/zap50k_deepstn_loc/' .. opts2.attrName

-- Construct Directory to Store Outputs
base_dir, im_folder = construct_directory(opts2.output_dir_path, opts2.attrName)


------------------------
-- INPUTS :: Combined --
------------------------
-- Set CUDA for GPU Usage
cutorch.setDevice(1)   -- 0:CPU 1:GPU
torch.setdefaulttensortype('torch.FloatTensor')

-- ImageNet Mean in BGR Order
local imagenet_bgr_mean = torch.Tensor(3, 227, 227):fill(0)
imagenet_bgr_mean[1] = 104
imagenet_bgr_mean[2] = 117
imagenet_bgr_mean[3] = 122

-- Load Pre-Computed Latent Variable Statistics (from 38K training images for generator)
image_stats = torch.load(opts1.stats_file)

-- Set Manual Seed
SEED = 1234
torch.manualSeed(SEED + attr_id)
cutorch.manualSeed(SEED + attr_id)


----------------------------
-- SETUP :: Pre-Generator --
----------------------------
local method = 'kaiming'

-- Top Branch
local pgTopBranchZ = nn.Sequential()
pgTopBranchZ:add(nn.View(-1, 2, 256))
pgTopBranchZ:add(nn.Select(2, 1))

local pgTopBranchY = nn.Sequential()
pgTopBranchY:add(require('libs.deepstn.weight-init')(nn.Linear(512, 256), method))
pgTopBranchY:add(cudnn.ReLU())
pgTopBranchY:add(require('libs.deepstn.weight-init')(nn.Linear(256, 50), method))
pgTopBranchY:add(nn.BatchNormalization(50))

pgTopBranchY:add(nn.MulConstantT(image_stats.var))
pgTopBranchY:add(nn.AddConstantT(image_stats.mean))


local pgTopBranch = nn.ParallelTable()
pgTopBranch:add(pgTopBranchZ)
pgTopBranch:add(pgTopBranchY)

-- Bottom Branch
local pgBotBranchZ = nn.Sequential()
pgBotBranchZ:add(nn.View(-1, 2, 256))
pgBotBranchZ:add(nn.Select(2, 2))

local pgBotBranchY = nn.Sequential()
pgBotBranchY:add(require('libs.deepstn.weight-init')(nn.Linear(512, 256), method))
pgBotBranchY:add(cudnn.ReLU())
pgBotBranchY:add(require('libs.deepstn.weight-init')(nn.Linear(256, 50), method))
pgBotBranchY:add(nn.BatchNormalization(50))

pgBotBranchY:add(nn.MulConstantT(image_stats.var))
pgBotBranchY:add(nn.AddConstantT(image_stats.mean))


local pgBotBranch = nn.ParallelTable()
pgBotBranch:add(pgBotBranchZ)
pgBotBranch:add(pgBotBranchY)

-- Join Sub-Branch
local splitBranch = nn.ConcatTable()
splitBranch:add(pgTopBranch)
splitBranch:add(pgBotBranch)

-- Initial Split
local branchZ = nn.Copy()

local branchY = nn.Sequential()
branchY:add(require('libs.deepstn.weight-init')(nn.Linear(512, 512), method))
branchY:add(cudnn.ReLU())

local initBranch = nn.ConcatTable()
initBranch:add(branchZ)
initBranch:add(branchY)

-- Main Branch
netPreGen = nn.Sequential()
netPreGen:add(initBranch)
netPreGen:add(splitBranch)

netPreGen:cuda()
netPreGen:training()

-- Configuration for the PreGen
local config_pregen = {learningRate = pregen_lr, learningRateDecay = pregen_lrd, weightDecay = pregen_wd, momentum = pregen_mom}


-------------------------
-- SETUP :: Attr2Image --
-------------------------

local ts_attr2image = os.clock()

-- Load Pre-Trained Attr2Image Model
local model_path_attr2img = opts1.model_dir .. opts1.model_name   -- path to pre-trained models
att2img = torch.load(model_path_attr2img)
encoder = att2img.encoder
decoder = att2img.decoder
print('Loaded pre-trained GENERATIVE model from: ' .. model_path_attr2img)

-- Save Original Decoder for Visualization
netViz = decoder:clone()
local last_conv = netViz:get(22):get(1):get(1):clone()
netViz:remove(22)
netViz:add(last_conv):add(nn.Tanh())
netViz:cuda()
netViz:evaluate()

-- Creates the Generator Model
local last_conv = decoder:get(22):get(1):get(1):clone()
decoder:remove(22)
decoder:add(last_conv):add(nn.Tanh())
decoder:add(nn.AddConstant(1)):add(nn.MulConstant(0.5))                  -- post-processing from generator
decoder:add(nn.MulConstant(255))                                         -- convert from 0~1 to 0~255
decoder:add(nn.IndexPrime(2, torch.LongTensor{3,2,1}))                   -- convert from RGB to BGR
decoder:add(nn.SpatialUpSamplingBilinear({oheight=227, owidth=227}))     -- upscale to 227 to match netRank's input
decoder:add(nn.MeanSub(imagenet_bgr_mean))                               -- subtract ImageNet mean

local decoderp = decoder:clone('weight', 'bias', 'gradWeight', 'gradBias')  -- duplicate net, share params

netGen = nn.ParallelTable()
netGen:add(decoder)
netGen:add(decoderp)

netGen:cuda()
netGen:evaluate()

print(string.format('Pre-Trained Attr2Image Model Loading Took %.2f sec\n', os.clock() - ts_attr2image))


----------------------
-- SETUP :: DEEPSTN --
----------------------

local ts_deepstn = os.clock()

-- Load DeepSTN Architecture (loads pre-trained localization model)
local zap50k_deepstn_module = dofile(opts2.modelFile)
netRank = zap50k_deepstn_module.create_network(opts2)

netRank:cuda()
netRank:training()

-- Set Up Learning Rate
local params_lr_ranker = setup_learningrate(netRank, ind_list, opts2.lrlocal)

-- Configuration for the Learner
local config_ranker = {learningRate = opts2.lr, learningRates = params_lr_ranker, weightDecay = 0.0005, momentum = 0.9, learningRateDecay = 0}

print(string.format('Pre-Trained DeepSTN Model Loading Took %.2f sec\n', os.clock() - ts_deepstn))


------------------
-- LOAD DATASET --
------------------
-- A and B represents one image pair: real_data_a[1] vs. real_data_b[1] is one pair

local ts_dataset = os.clock()

-- Factor of Real Pairs Used as "Base" for Fair Experiment
local real_base_fact = 0.5

-- Original # of Real Pairs
-- MOD: adjust according to # of real pairs
local real_pairs_orig = torch.Tensor({649, 863, 854, 659, 1051, 861, 898, 969, 818, 744})

-- Real Training Pairs
local real_train_data_a, real_train_data_b, real_train_data_labels = load_data(opts2.train_data_real)
local real_val_data_a, real_val_data_b, real_val_data_labels = load_data(opts2.val_data_real)
local real_test_data_a, real_test_data_b, real_test_data_labels = load_data(opts2.test_data_real)

-- Modify # of Real Training Pairs
num_real_train_full = torch.ceil(real_train_data_labels:size()[1] * real_fact)

-- NEW: Base # of Real Training Pairs for Synthetic Active Batch
num_real_train = torch.ceil(real_pairs_orig[opts2.attrNum] * real_base_fact)

-- Remove Redundant Real Pairs
real_train_a = real_train_data_a:sub(1, num_real_train)
real_train_b = real_train_data_b:sub(1, num_real_train)
real_train_labels = real_train_data_labels:sub(1, num_real_train)

-- Modify # of Synth Training Pairs
num_synth_train = num_real_train

-- Validation Set
real_val_a = real_val_data_a
real_val_b = real_val_data_b
real_val_labels = real_val_data_labels

num_real_val = real_val_labels:size()[1]

-- Test Set
real_test_a = real_test_data_a
real_test_b = real_test_data_b
real_test_labels = real_test_data_labels

num_real_test = real_test_labels:size()[1]

-- Handle Corner Case for netRank where BatchSize =/= 1
if math.fmod(num_real_train, opts2.batchSize) == 1 then num_real_train = num_real_train - 1 end
if math.fmod(num_synth_train, opts2.batchSize) == 1 then num_synth_train = num_synth_train - 1 end
if math.fmod(num_real_val, opts2.batchSize) == 1 then num_real_val = num_real_val - 1 end
if math.fmod(num_real_test, opts2.batchSize) == 1 then num_real_test = num_real_test - 1 end

print(string.format('Dataset Loading Took %.2f sec\n', os.clock() - ts_dataset))

-- Random Inputs for Pre-Gen
pregen_inputs = torch.randn(num_synth_train, 512)

print('Num Real Train Total: ' .. num_real_train_full .. ' pairs')
print('Num Real Train: ' .. num_real_train .. ' pairs')
print('Num Synth Train: ' .. num_synth_train .. ' pairs')
print('Num Real Val: ' .. num_real_val .. ' pairs')
print('Num Real Test: ' .. num_real_test .. ' pairs')


--------------
-- TRAINING --
--------------
-- Get Global Model Parameters
local paramsPreGen, gradParamsPreGen = netPreGen:getParameters()
local paramsGen, gradParamsGen = netGen:getParameters()
local paramsRank, gradParamsRank = netRank:getParameters()

-- Initialize Performance Counter
local count_real_right = 0
local count_synth_right = 0
local count_val_right = 0
local count_test_right = 0

-- Cumulative Loss/Epoch
local loss_real_epoch = 0
local loss_synth_epoch = 0
local loss_pregen_epoch = 0

-- Initialize Performance Log
local loss_real_log = torch.Tensor(opts2.numIter, 1):fill(0)
local loss_synth_log = torch.Tensor(opts2.numIter, 1):fill(0)
local loss_pregen_log = torch.Tensor(opts2.numIter, 1):fill(0)

local acc_real_log = torch.Tensor(opts2.numIter, 1):fill(0) 
local acc_synth_log = torch.Tensor(opts2.numIter, 1):fill(0)
local acc_val_log = torch.Tensor(opts2.numIter, 1):fill(0)
local acc_test_log = torch.Tensor(opts2.numIter, 1):fill(0)

-- Initialize Tensors
local batch_real = {}           -- input real batch
local batch_input_synth = {}    -- input synth batch in {y,z} form
local batch_synth = {}          -- input synth batch in image form
local batch_val = {}            -- validation real batch
local batch_test = {}           -- test real batch

local grad_real = {}
local grad_synth = {}
local grad_pregen = {}

local batch_synth_targets = {}    -- dynamically determined ground truth w.r.t. output of pregen


-----------------------------------------------------
-- (Closure) Train DeepSTN Ranker using Real Pairs --
-----------------------------------------------------
local fRankX_Real = function(x)
  
  collectgarbage()
  if x ~= paramsRank then
    print('<WARNING> PARAMETERS DO NOT MATCH!')
    paramsRank:copy(x)
  end
  
  -- Reset Loss & Gradients for given Batch
  netRank:zeroGradParameters()
  
  local batch_loss_ranker_real = 0
  local grad1_real = torch.Tensor(mini_batchsize, 1):zero()
  local grad2_real = torch.Tensor(mini_batchsize, 1):zero()
  
  -- Forward Pass of the Batch Image Pairs
  local output_real = netRank:forward(batch_real)
  
  -- Criterion: Compute RankLoss and Gradients given Batch
  grad1_real, grad2_real, batch_loss_ranker_real = compute_rankloss(output_real, batch_real_targets, mini_batchsize)
  grad_real = {grad1_real:cuda(), grad2_real:cuda()}
  
  -- Pass Back Gradients from Real Pairs
  netRank:backward(batch_real, grad_real)
  
  -- Add Batch Loss into Cumulative Loss
  loss_real_epoch = loss_real_epoch + batch_loss_ranker_real
  
  -- Normalize Gradients and Loss
  gradParamsRank:div(mini_batchsize)
  batch_loss_ranker_real = batch_loss_ranker_real / mini_batchsize
  
  -- Track Performance on Real Training Pairs
  count_real_right = count_real_right + count_correct(output_real, batch_real_targets)
  
  return batch_loss_ranker_real, gradParamsRank
  
end


------------------------------------------------------
-- (Closure) Train DeepSTN Ranker using Synth Pairs --
------------------------------------------------------
local fRankX_Synth = function(x)
  
  collectgarbage()
  if x ~= paramsRank then
    print('<WARNING> PARAMETERS DO NOT MATCH!')
    paramsRank:copy(x)
  end
  
  -- Reset Loss & Gradients for given Batch
  netRank:zeroGradParameters()
  
  local batch_loss_ranker_synth = 0
  local grad1_synth = torch.Tensor(mini_batchsize, 1):zero()
  local grad2_synth = torch.Tensor(mini_batchsize, 1):zero()
  
  -- Forward Pass to Generate and use Synthetic Images Pairs
  local batch_input_synth = netPreGen:forward(batch_input_pregen)    -- generate synth parameters {y,z}
  local batch_synth = netGen:forward(batch_input_synth)              -- generate synth images
  output_synth = netRank:forward(batch_synth)                        -- predicted attribute scores

  -- Obtain Ground Truth Synth Labels based on Attribute Vector
  local topY = batch_input_synth[1][2][{{},{opts2.attrNum+40}}]
  local botY = batch_input_synth[2][2][{{},{opts2.attrNum+40}}]
  batch_synth_targets = ((torch.sign(topY-botY) + 1) / 2):double()   -- 1 for "more, 0 for "less", 0.5 for "equal", assumes perfect generator
  
  -- Criterion: Compute RankLoss and Gradients given Batch
  grad1_synth, grad2_synth, batch_loss_ranker_synth = compute_rankloss(output_synth, batch_synth_targets, mini_batchsize)
  grad_synth = {grad1_synth:cuda(), grad2_synth:cuda()}
  
  -- Pass Back Gradients from Synth Pairs
  netRank:backward(batch_synth, grad_synth)
  
  -- Add Batch Loss into Cumulative Loss
  loss_synth_epoch = loss_synth_epoch + batch_loss_ranker_synth
  
  -- Normalize Gradients and f(X)
  gradParamsRank:div(mini_batchsize)
  batch_loss_ranker_synth = batch_loss_ranker_synth / mini_batchsize
  
  -- Track Performance on Synthetic Training Pairs
  count_synth_right = count_synth_right + count_correct(output_synth, batch_synth_targets)
  
  return batch_loss_ranker_synth, gradParamsRank
  
end


-----------------------------------------------------
-- (Closure) Train Pre-Generator using Synth Pairs --
-----------------------------------------------------
local fPreGenX = function(x)
  
  collectgarbage()
  if x ~= paramsPreGen then
    print('<WARNING> PARAMETERS DO NOT MATCH!')
    paramsPreGen:copy(x)
  end
  
  -- Reset Loss & Gradients for given Batch
  netPreGen:zeroGradParameters()
  netGen:zeroGradParameters()
  netRank:zeroGradParameters()
  
  local batch_loss_pregen = 0
  local grad1_pregen = torch.Tensor(mini_batchsize, 1):zero()
  local grad2_pregen = torch.Tensor(mini_batchsize, 1):zero()
  
  -- Forward Pass to Generate and use Synthetic Images Pairs
  local batch_input_synth_pg = netPreGen:forward(batch_input_pregen)    -- generate synth parameters (y,z)
  local batch_synth_pg = netGen:forward(batch_input_synth_pg)           -- generate synth images
  output_synth_pg = netRank:forward(batch_synth_pg)                     -- predicted attribute scores
  
  -----------------
  --| RANK LOSS |--
  -----------------
  
  -- Flip Ground Truth for PreGen Training (1 -> 0, 0 -> 1)
  batch_synth_targets = torch.abs(batch_synth_targets - 1)
  
  -- Criterion: Compute RankLoss and Gradients given Batch
  grad1_pregen, grad2_pregen, batch_loss_pregen = compute_rankloss(output_synth_pg, batch_synth_targets, mini_batchsize)
  grad_pregen = {grad1_pregen:cuda(), grad2_pregen:cuda()}
  
  -- Pass Back Gradients from Synth Pairs all the way to PreGen
  dRank_dGenOut = netRank:updateGradInput(batch_synth_pg, grad_pregen)
  dGenOut_dPreGenOut = netGen:updateGradInput(batch_input_synth_pg, dRank_dGenOut)
  netPreGen:backward(batch_input_pregen, dGenOut_dPreGenOut)
  
  -- Add Batch Loss into Cumulative Loss
  loss_pregen_epoch = loss_pregen_epoch + batch_loss_pregen
  
  -- Normalize Gradients and f(X)
  batch_loss_pregen = batch_loss_pregen / mini_batchsize
  gradParamsPreGen:div(mini_batchsize)
  
  return batch_loss_pregen, gradParamsPreGen
  
end


-------------------
-- Training Loop --
-------------------
snapshot = {}

state_pregen = {}
state_pregen.evalCounter = 0     -- reset pregen counter

synth_abatch_final_a = torch.Tensor(num_synth_train, 3, 227, 227):fill(0)
synth_abatch_final_b = torch.Tensor(num_synth_train, 3, 227, 227):fill(0)
synth_abatch_final_targets = torch.Tensor(num_synth_train, 1):fill(0)

for epoch = 1, opts2.numIter do
  
  local start_time = os.clock()
  
  -- Reset Performance Counters
  count_real_right = 0
  count_synth_right = 0
  count_val_right = 0
  count_test_right = 0
  
  loss_real_epoch = 0
  loss_synth_epoch = 0
  loss_pregen_epoch = 0
  
  -- Randomize Order of Training Pairs
  rnd_order_real = torch.randperm(num_real_train)
  rnd_order_synth = torch.randperm(num_synth_train)
  rnd_order_val = torch.linspace(1, num_real_val, num_real_val)
  rnd_order_test = torch.linspace(1, num_real_test, num_real_test)
  
  
  ----------------------------
  -- Training on Real Pairs --
  ----------------------------
  netRank:training()
  
  -- Real Mini-Batch (images)
  for t = 1, num_real_train, opts2.batchSize do
    mini_batchsize = math.min(t + opts2.batchSize - 1, num_real_train) - t + 1
    batch_real, batch_real_targets = create_batch_ranker(real_train_a, real_train_b, real_train_labels, rnd_order_real, mini_batchsize, t)
    optim.sgd(fRankX_Real, paramsRank, config_ranker)
  end
  
  -- Log Results on Real Pairs
  loss_real_log[epoch] = loss_real_epoch / num_real_train
  acc_real_log[epoch] = count_real_right / num_real_train
  
  
  -----------------------------
  -- Training on Synth Pairs --
  -----------------------------
  -- Synth Mini-Batch (latent variables {z,y})
  for t = 1, num_synth_train, opts2.batchSize do
    
    mini_batchsize = math.min(t + opts2.batchSize - 1, num_synth_train) - t + 1
    batch_input_pregen = create_batch_pregen(pregen_inputs, rnd_order_synth, mini_batchsize, t)
    
    netPreGen:evaluate()
    optim.sgd(fRankX_Synth, paramsRank, config_ranker)
    
    netPreGen:training()
    optim.sgd(fPreGenX, paramsPreGen, config_pregen, state_pregen)
    
  end
  
  -- Log Results on Synth Pairs
  loss_synth_log[epoch] = loss_synth_epoch / num_synth_train
  loss_pregen_log[epoch] = loss_pregen_epoch / num_synth_train
  
  acc_synth_log[epoch] = count_synth_right / num_synth_train
  
  
  ----------------------------------
  -- Evaluate on Validation Pairs --
  ----------------------------------
  netRank:evaluate()
  
  -- Real Mini-Batch for Validation (images)
  for t = 1, num_real_val, opts2.batchSize do
    mini_batchsize = math.min(t + opts2.batchSize - 1, num_real_val) - t + 1
    batch_val, batch_val_targets = create_batch_ranker(real_val_a, real_val_b, real_val_labels, rnd_order_val, mini_batchsize, t)
    
    -- Forward Pass into Updated Ranker
    local output_val = netRank:forward(batch_val)
    count_val_right = count_val_right + count_correct(output_val, batch_val_targets)
  end
  
  -- Log Results on Validation Pairs
  acc_val_log[epoch] = count_val_right / num_real_val
  
  
  ----------------------------
  -- Evaluate on Test Pairs --
  ----------------------------
  -- Real Mini-Batch for Evaluation (images)
  for t = 1, num_real_test, opts2.batchSize do
    mini_batchsize = math.min(t + opts2.batchSize - 1, num_real_test) - t + 1
    batch_test, batch_test_targets = create_batch_ranker(real_test_a, real_test_b, real_test_labels, rnd_order_test, mini_batchsize, t)
    
    -- Forward Pass into Updated Ranker
    local output_test = netRank:forward(batch_test)
    count_test_right = count_test_right + count_correct(output_test, batch_test_targets)
  end
  
  -- Log Results on Validation Pairs
  acc_test_log[epoch] = count_test_right / num_real_test
  
  
  -------------
  -- Logging --
  -------------
  -- Real + Synth w/ PreGen
  print(string.format('(Epoch %3.0f) R-Acc: %0.3f (%d/%d) || R-Loss: %0.3f || S-Acc: %0.3f (%d/%d) || S-Loss: %0.3f || PG-Loss: %0.3f ||| V-Acc: %0.3f (%d/%d) ||| T-Acc: %0.3f (%d/%d) ||| Elapsed %.2f sec', epoch, acc_real_log[epoch][1], count_real_right, num_real_train, loss_real_log[epoch][1], acc_synth_log[epoch][1], count_synth_right, num_synth_train, loss_synth_log[epoch][1], loss_pregen_log[epoch][1], acc_val_log[epoch][1], count_val_right, num_real_val, acc_test_log[epoch][1], count_test_right, num_real_test, os.clock() - start_time))
  
  -- Save Performance Logs
  if save_logs then
    save_stats_full(epoch, 1, base_dir, im_folder, loss_real_log, loss_synth_log, acc_real_log, acc_synth_log, acc_val_log, acc_test_log)
  end
  
  -- Save Models
  if ( (save_model == 1) and (math.fmod(epoch, save_model_gap) == 0) ) then
    
    -- Save Additional Variables
    snapshot.seed = SEED
    snapshot.epoch = epoch
    snapshot.imagenet_bgr_mean = imagenet_bgr_mean
    snapshot.opt1 = opt1
    snapshot.opt2 = opt2
    
    snapshot.config_pregen = config_pregen
    snapshot.config_ranker = config_ranker
    
    snapshot.pregen_inputs = pregen_inputs
    
    local snapshot_params = base_dir .. 'nobatch_part2_snapshot_params.dat'
    torch.save(snapshot_params, snapshot)
    print(string.format('\nParameters Saved @ Epoch %d', epoch))
    
    -- Save Individual Models
    save_abatch_models(abatch_id, epoch, base_dir, netPreGen, netGen, netRank)
    print(string.format('\nModel Saved @ Epoch %d', epoch))
    
  end
  
  -- Extra Feature #1: Save Generated Synthetic Images in Appropriate Epoch
  if ( (save_synth_images == 1) and ( (math.fmod(epoch, save_synth_images_gap) == 0) or epoch == 1 ) ) then
  
    -- Store Image on Vision Projects Page
    local tmp_expdir = ''   -- MOD: path to save the synthetic images
    paths.mkdir(tmp_expdir)
    local image_outdir = tmp_expdir .. 'epoch' .. epoch .. '/'
    paths.mkdir(image_outdir)
    
    netPreGen:evaluate()
    
    for v = 1, num_synth_train do
      
      local image_params = netPreGen:forward(pregen_inputs[v]:cuda())
      local imageA_yz = {image_params[1][1], image_params[1][2]}
      local imageB_yz = {image_params[2][1], image_params[2][2]}
      
      local imageA_mat = netViz:forward(imageA_yz)
      save_image(imageA_mat, image_outdir, v, 1, epoch)
      
      local imageB_mat = netViz:forward(imageB_yz)
      save_image(imageB_mat, image_outdir, v, 2, epoch)

    end
  
  end
  
end


-------------------
-- Post Learning --
-------------------
-- Finalize Synth Images
synth_abatch_final_a, synth_abatch_final_b, synth_abatch_final_labels = finlaize_synth_pairs(pregen_inputs, num_synth_train, opts2.batchSize)

print('\nSaving finalized synth pairs...')

local fsynth_name = base_dir .. 'abatch2_fsynth_data.h5'

local myFile = hdf5.open(fsynth_name, 'w')
myFile:write('data_a', synth_abatch_final_a:float())
myFile:write('data_b', synth_abatch_final_b:float())
myFile:write('labels', synth_abatch_final_labels:float())
myFile:close()
