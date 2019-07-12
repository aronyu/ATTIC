
-- all the utility functions

-------------------------------------
-- Initialize Localization Network --
-------------------------------------

-- Localization Network that has been fully trained
function initialize_localization_network(loc_network_path, atr_name, ind_list, siamese_1_combine, mul_factor, pretrain_model_name)
  
  -- Use Pre-Trained Models
  local loc_model_path = loc_network_path .. '/' .. pretrain_model_name
  local tmp_loc_model = torch.load(loc_model_path):double()
  print('Loaded pre-train STN patch localization model from: ' .. loc_model_path)

  for i = 1, #ind_list do
    siamese_1_combine:get(1):get(1):get(1):get(1):get(2):get(1):get(ind_list[i]).weight:copy(tmp_loc_model:get(1):get(1):get(1):get(2):get(1):get(ind_list[i]).weight)
    siamese_1_combine:get(1):get(1):get(1):get(1):get(2):get(1):get(ind_list[i]).bias:copy(tmp_loc_model:get(1):get(1):get(1):get(2):get(1):get(ind_list[i]).bias)
  end
  
  local old_mul_factor = tmp_loc_model:get(1):get(1):get(1):get(2):get(1):get(22).weight[1]
  siamese_1_combine:get(1):get(1):get(1):get(1):get(2):get(1):get(21).weight[1]:mul(old_mul_factor/mul_factor)
  siamese_1_combine:get(1):get(1):get(1):get(1):get(2):get(1):get(21).bias[1] = siamese_1_combine:get(1):get(1):get(1):get(1):get(2):get(1):get(21).bias[1] * (old_mul_factor/mul_factor)

end


--------------------------
-- Load Real Image Data --
--------------------------

function load_data(f_name)

  print('loading: '..f_name)
  local myFile = hdf5.open(f_name,'r')
  local mydata = myFile:read('data'):all()
  local mydatap = myFile:read('datap'):all()
  local mylabel = myFile:read('label'):all()
  mylabel = mylabel:cuda()

  return mydata, mydatap, mylabel
  
end

function load_data_synth(f_name)

  print('loading: '..f_name)
  local myFile = hdf5.open(f_name,'r')
  local mydatatop = myFile:read('dataTop'):all()
  local mydatabot = myFile:read('dataBot'):all()
  local mylabelhuman = myFile:read('labelsHuman'):all()
  local mylabelauto = myFile:read('labelsAuto'):all()
  mylabelhuman = mylabelhuman:cuda()
  mylabelauto = mylabelauto:cuda()

  return mydatatop, mydatabot, mylabelhuman, mylabelauto
  
end


-----------------------
-- Set Learning Rate --
-----------------------

function setup_learningrate(siamese_net, ind_list, loc_lr)
  
  local params_lr_m = siamese_net:clone()
  local params_lr = params_lr_m:getParameters()
  params_lr:fill(1)

  for i = 1, #ind_list do
    params_lr_m:get(1):get(1):get(1):get(1):get(1):get(2):get(1):get(ind_list[i]).weight:fill(loc_lr)
    params_lr_m:get(1):get(1):get(1):get(1):get(1):get(2):get(1):get(ind_list[i]).bias:fill(loc_lr)
  end
  params_lr_m:get(1):get(1):get(1):get(1):get(1):get(2):get(1):get(22).weight:fill(0)

  return params_lr

end


--------------------------------------
-- Create Directory/Sub-Directories --
--------------------------------------

function construct_directory(output_dir_path, atr_name, num_samples, modeltype)
  
  local base_dir = ''
  base_dir = output_dir_path..'/models_combined/attribute='..atr_name..'/'
  paths.mkdir(output_dir_path..'/models_combined/')
  paths.mkdir(base_dir)
  
  local im_folder = 'images/'
  paths.mkdir(base_dir .. im_folder)
  
  return base_dir, im_folder

end


---------------------------
-- Save Synthetic Images --
---------------------------

function save_image(image_mat, image_outdir, pair_num, image_num, epoch)
  
  local image_mat = torch.squeeze(image_mat)
  image_mat:add(1):mul(0.5)
  
  local image_formatted = image.toDisplayTensor(image_mat)
  image_formatted = image_formatted:double()
  image_formatted:mul(255)
  image_formatted = image_formatted:byte()
  
  image.save(image_outdir .. string.format('pair%d_id%d_epoch%d.jpg', pair_num, image_num, epoch), image_formatted)
  
end


-------------------------------------------
-- Initialize Synthetic Image Parameters --
-------------------------------------------

-- Sample Y using multi-variate gaussian. Order them 1 or 2 based on strength of target attribute.

function init_synth_params(num_synth, attrID, image_stats)
  
  local target_attrID = attrID + 40   -- specifically for zap50k
  
  -- Z is a normal distribution
  local z1_init = torch.rand(num_synth, 256):normal(0,1):cuda()
  local z2_init = torch.rand(num_synth, 256):normal(0,1):cuda()

  -- Y is based on prior distribution from real training set
  local y1_init = torch.Tensor(num_synth, 50):zero():cuda()
  local y2_init = torch.Tensor(num_synth, 50):zero():cuda()


  for p = 1, num_synth do
    
    local gen_y1 = distributions.mvn.rnd(image_stats.mean, image_stats.cov):cuda()
    local gen_y2 = distributions.mvn.rnd(image_stats.mean, image_stats.cov):cuda()
    
    -- Always Put Larger Y in (1)
    if gen_y1[target_attrID] > gen_y2[target_attrID] then
      y1_init[p] = gen_y1
      y2_init[p] = gen_y2
    else
      y1_init[p] = gen_y2
      y2_init[p] = gen_y1
    end
    
  end
  
  local input_synth_a = {z1_init, y1_init}
  local input_synth_b = {z2_init, y2_init}
  local input_synth = {input_synth_a, input_synth_b}
  --local synth_labels = torch.Tensor(num_synth, 1):fill(1):cuda()
  
  return input_synth
  
end

-- same function for LFW faces (73 attributes)
function init_synth_params_lfw(num_synth, attrID, image_stats, attrMap)
  
  local target_attrID = attrMap[attrID]   -- specifically for lfw
  
  -- Z is a normal distribution
  local z1_init = torch.rand(num_synth, 256):normal(0,1):cuda()
  local z2_init = torch.rand(num_synth, 256):normal(0,1):cuda()

  -- Y is based on prior distribution from real training set
  local y1_init = torch.Tensor(num_synth, 73):zero():cuda()
  local y2_init = torch.Tensor(num_synth, 73):zero():cuda()


  for p = 1, num_synth do
    
    local gen_y1 = distributions.mvn.rnd(image_stats.mean, image_stats.cov):cuda()
    local gen_y2 = distributions.mvn.rnd(image_stats.mean, image_stats.cov):cuda()
    
    -- Always Put Larger Y in (1)
    if gen_y1[target_attrID] > gen_y2[target_attrID] then
      y1_init[p] = gen_y1
      y2_init[p] = gen_y2
    else
      y1_init[p] = gen_y2
      y2_init[p] = gen_y1
    end
    
  end
  
  local input_synth_a = {z1_init, y1_init}
  local input_synth_b = {z2_init, y2_init}
  local input_synth = {input_synth_a, input_synth_b}
  --local synth_labels = torch.Tensor(num_synth, 1):fill(1):cuda()
  
  return input_synth
  
end



--------------------------------
-- Compute RankLoss Criterion --
--------------------------------

function compute_rankloss(scores, targets, batch_size)
  
  local batch_loss = 0
  local grads1 = torch.Tensor(batch_size, 1):zero()
  local grads2 = torch.Tensor(batch_size, 1):zero()
  
  for i = 1, batch_size do
    
    local loss = 0
    local grad1 = torch.Tensor(1):zero()
    local grad2 = torch.Tensor(1):zero()
    
    local score_pair = torch.Tensor(2)
    score_pair[1] = scores[1][i][1]
    score_pair[2] = scores[2][i][1]
    
    -- Sigmoid of Score Difference
    local score_diff = score_pair[1] - score_pair[2]
    local prob = 1 / (1 + torch.exp(-1 * score_diff))
    local target = targets[i]
    
    -- Loss Function
    loss = -1 * target * score_diff + torch.log(1 + torch.exp(score_diff))
    
    -- Gradients
    grad1 = -1 * target + prob
    grad2 = target - prob
    
    -- Update Loss and Gradients
    batch_loss = batch_loss + loss
    grads1[i] = grad1
    grads2[i] = grad2
    
  end

  return grads1, grads2, batch_loss
  
end


-----------------------
-- Create Mini Batch --
-----------------------

-- Create Batch for Inputs to Ranker
function create_batch_ranker(mydata, mydatap, mylabel, rnd_order, batch_size, t)
  
  local batch_order_rnd = rnd_order:sub(t, t+batch_size-1)
  
  local inputA = torch.Tensor(batch_size, 3, 227, 227):cuda()
  local inputB = torch.Tensor(batch_size, 3, 227, 227):cuda()
  local batch_targets = torch.DoubleTensor(batch_size, 1)

  for s = 1, batch_size do
    local index_rnd = batch_order_rnd[s]
    inputA[s] = image.scale(mydata[index_rnd], 227, 227)
    inputB[s] = image.scale(mydatap[index_rnd], 227, 227)
    batch_targets[s] = mylabel[index_rnd][1]
  end
  local batch_input_ranker = {inputA, inputB}
  
  return batch_input_ranker, batch_targets

end

-- Create Batch for Inputs to PreGen
function create_batch_pregen(pregen_inputs, rnd_order, batch_size, t)

  local batch_order_rnd = rnd_order:sub(t, t+batch_size-1)
  
  local batch_input_pregen = torch.Tensor(batch_size, 512):fill(0):cuda()

  for s = 1, batch_size do
    local index_rnd = batch_order_rnd[s]
    batch_input_pregen[s] = pregen_inputs[index_rnd]
  end
  
  return batch_input_pregen

end

-- Create Batch for Synth Images in the form of {z,y}
function create_batch_yz(input_synth, rnd_order, batch_size, t)

  local batch_order_rnd = rnd_order:sub(t, batch_size+t-1)
    
  local batch_z1 = torch.Tensor(batch_size, 256):fill(0):cuda()
  local batch_y1 = torch.Tensor(batch_size, 50):fill(0):cuda()
  local batch_z2 = torch.Tensor(batch_size, 256):fill(0):cuda()
  local batch_y2 = torch.Tensor(batch_size, 50):fill(0):cuda()
  --local targets = torch.Tensor(tonumber(batch_size), 1):cuda()

  for s = 1, batch_size do
    local index_rnd = batch_order_rnd[s]
    batch_z1[s] = input_synth[1][1][index_rnd]
    batch_y1[s] = input_synth[1][2][index_rnd]
    batch_z2[s] = input_synth[2][1][index_rnd]
    batch_y2[s] = input_synth[2][2][index_rnd]
    --targets[s] = mylabel[index_rnd]
  end
  local batch_input_synth_a = {batch_z1:cuda(), batch_y1:cuda()}
  local batch_input_synth_b = {batch_z2:cuda(), batch_y2:cuda()}
  local batch_input_synth = {batch_input_synth_a, batch_input_synth_b}
  
  return batch_input_synth

end

-- same code for LFW faces
function create_batch_yz_lfw(input_synth, rnd_order, batch_size, t)

  local batch_order_rnd = rnd_order:sub(t, batch_size+t-1)
    
  local batch_z1 = torch.Tensor(batch_size, 256):fill(0):cuda()
  local batch_y1 = torch.Tensor(batch_size, 73):fill(0):cuda()
  local batch_z2 = torch.Tensor(batch_size, 256):fill(0):cuda()
  local batch_y2 = torch.Tensor(batch_size, 73):fill(0):cuda()
  --local targets = torch.Tensor(tonumber(batch_size), 1):cuda()

  for s = 1, batch_size do
    local index_rnd = batch_order_rnd[s]
    batch_z1[s] = input_synth[1][1][index_rnd]
    batch_y1[s] = input_synth[1][2][index_rnd]
    batch_z2[s] = input_synth[2][1][index_rnd]
    batch_y2[s] = input_synth[2][2][index_rnd]
    --targets[s] = mylabel[index_rnd]
  end
  local batch_input_synth_a = {batch_z1:cuda(), batch_y1:cuda()}
  local batch_input_synth_b = {batch_z2:cuda(), batch_y2:cuda()}
  local batch_input_synth = {batch_input_synth_a, batch_input_synth_b}
  
  return batch_input_synth

end


-- Compare Predicted and Target Labels
function count_correct(scores_pred, targets)
  local preds = ((torch.sign(scores_pred[1]-scores_pred[2]) + 1) / 2):double()
  local num_correct = torch.sum((targets - preds):eq(0))
  return num_correct
end

-- Get Raw Test Results
function get_raw_results(scores_pred, targets)
  local preds = ((torch.sign(scores_pred[1]-scores_pred[2]) + 1) / 2):double()
  local raw_output = ((targets - preds):eq(0)):transpose(1,2):double()
  return raw_output
end


-- Finalize Converged Synth Pairs
function finlaize_synth_pairs(pregen_inputs, abatchSize, batchSize)
  
  local synth_abatch_single_a = torch.Tensor(abatchSize, 3, 227, 227):fill(0)
  local synth_abatch_single_b = torch.Tensor(abatchSize, 3, 227, 227):fill(0)
  local synth_abatch_single_targets = torch.Tensor(abatchSize, 1):fill(0)
  
  -- Move Converged Synth Pairs into Statis Real Pairs
  for t = 1, abatchSize, batchSize do
    
    local mini_batchsize = math.min(t + batchSize - 1, abatchSize) - t + 1
    local batch_input_pg_single = pregen_inputs:sub(t, t+mini_batchsize-1)
    
    local batch_input_synth_pg_single = netPreGen:forward(batch_input_pg_single:cuda())
    local batch_synth_pg_single = netGen:forward(batch_input_synth_pg_single)
    
    -- Obtain Ground Truth Synth Labels based on Attribute Vector
    local topY = batch_input_synth_pg_single[1][2][{{},{opts2.attrNum+40}}]
    local botY = batch_input_synth_pg_single[2][2][{{},{opts2.attrNum+40}}]
    local batch_synth_pg_targets_single = ((torch.sign(topY-botY) + 1) / 2):double()
    
    -- Save Generated Images into New Tensor
    synth_abatch_single_a[{{t, t+mini_batchsize-1}, {}}] = batch_synth_pg_single[1]:float()
    synth_abatch_single_b[{{t, t+mini_batchsize-1}, {}}] = batch_synth_pg_single[2]:float()
    synth_abatch_single_targets[{{t, t+mini_batchsize-1}}] = batch_synth_pg_targets_single
    
  end
  
  return synth_abatch_single_a, synth_abatch_single_b, synth_abatch_single_targets
  
end


-- Save Loss and Accuray
function save_stats(epoch, save_stats_gap, base_dir, im_folder, loss_real, loss_synth, acc_real, acc_synth, acc_val)

  if((epoch-1)%save_stats_gap == 0) then
  
    local epoch_loss_real = torch.Tensor(epoch)
    local epoch_loss_synth = torch.Tensor(epoch)
    local epoch_acc_real = torch.Tensor(epoch)
    local epoch_acc_synth = torch.Tensor(epoch)
    local epoch_acc_val = torch.Tensor(epoch)
    
    local filename_loss_real = base_dir .. 'loss_real.txt'
    local filename_loss_synth = base_dir .. 'loss_synth.txt'
    local filename_acc_real = base_dir .. 'acc_real.txt'
    local filename_acc_synth = base_dir .. 'acc_synth.txt'
    local filename_acc_val = base_dir .. 'acc_val.txt'
    
    r_loss = io.open(filename_loss_real, 'w')
    s_loss = io.open(filename_loss_synth, 'w')
    r_acc = io.open(filename_acc_real, 'w')
    s_acc = io.open(filename_acc_synth, 'w')
    v_acc = io.open(filename_acc_val, 'w')
    
    for k = 1, epoch do
    
      epoch_loss_real[k] = torch.mean(loss_real[k])
      epoch_loss_synth[k] = torch.mean(loss_synth[k])
      epoch_acc_real[k] = acc_real[k]
      epoch_acc_synth[k] = acc_synth[k]
      epoch_acc_val[k] = acc_val[k]
      
      r_loss:write(tostring(epoch_loss_real[k]), '\n')
      s_loss:write(tostring(epoch_loss_synth[k]), '\n')
      
      r_acc:write(tostring(epoch_acc_real[k]), '\n')
      s_acc:write(tostring(epoch_acc_synth[k]), '\n')
      v_acc:write(tostring(epoch_acc_val[k]), '\n')
      
    end
    
    r_loss:close()
    s_loss:close()
    r_acc:close()
    s_acc:close()
    v_acc:close()
    
  end

end


-- Save Loss and Accuray
function save_stats_real(epoch, save_stats_gap, base_dir, im_folder, loss_real, acc_real, acc_val, acc_test)

  if((epoch-1)%save_stats_gap == 0) then
  
    local epoch_loss_real = torch.Tensor(epoch)
    local epoch_acc_real = torch.Tensor(epoch)
    local epoch_acc_val = torch.Tensor(epoch)
    local epoch_acc_test = torch.Tensor(epoch)
    
    local filename_loss_real = base_dir .. 'loss_real.txt'
    local filename_acc_real = base_dir .. 'acc_real.txt'
    local filename_acc_val = base_dir .. 'acc_val.txt'
    local filename_acc_test = base_dir .. 'acc_test.txt'
    
    r_loss = io.open(filename_loss_real, 'w')
    r_acc = io.open(filename_acc_real, 'w')
    v_acc = io.open(filename_acc_val, 'w')
    t_acc = io.open(filename_acc_test, 'w')
    
    for k = 1, epoch do
    
      epoch_loss_real[k] = torch.mean(loss_real[k])
      epoch_acc_real[k] = acc_real[k]
      epoch_acc_val[k] = acc_val[k]
      epoch_acc_test[k] = acc_test[k]
      
      r_loss:write(tostring(epoch_loss_real[k]), '\n')
      r_acc:write(tostring(epoch_acc_real[k]), '\n')
      v_acc:write(tostring(epoch_acc_val[k]), '\n')
      t_acc:write(tostring(epoch_acc_test[k]), '\n')
      
    end
    
    r_loss:close()
    r_acc:close()
    v_acc:close()
    t_acc:close()
    
  end

end


-- Save Loss and Accuray Full Version
function save_stats_full(epoch, save_stats_gap, base_dir, im_folder, loss_real, loss_synth, acc_real, acc_synth, acc_val, acc_test)

  if((epoch-1)%save_stats_gap == 0) then
  
    local epoch_loss_real = torch.Tensor(epoch)
    local epoch_loss_synth = torch.Tensor(epoch)
    local epoch_acc_real = torch.Tensor(epoch)
    local epoch_acc_synth = torch.Tensor(epoch)
    local epoch_acc_val = torch.Tensor(epoch)
    local epoch_acc_test = torch.Tensor(epoch)
    
    local filename_loss_real = base_dir .. 'loss_real.txt'
    local filename_loss_synth = base_dir .. 'loss_synth.txt'
    local filename_acc_real = base_dir .. 'acc_real.txt'
    local filename_acc_synth = base_dir .. 'acc_synth.txt'
    local filename_acc_val = base_dir .. 'acc_val.txt'
    local filename_acc_test = base_dir .. 'acc_test.txt'
    
    r_loss = io.open(filename_loss_real, 'w')
    s_loss = io.open(filename_loss_synth, 'w')
    r_acc = io.open(filename_acc_real, 'w')
    s_acc = io.open(filename_acc_synth, 'w')
    v_acc = io.open(filename_acc_val, 'w')
    t_acc = io.open(filename_acc_test, 'w')
    
    for k = 1, epoch do
    
      epoch_loss_real[k] = torch.mean(loss_real[k])
      epoch_loss_synth[k] = torch.mean(loss_synth[k])
      epoch_acc_real[k] = acc_real[k]
      epoch_acc_synth[k] = acc_synth[k]
      epoch_acc_val[k] = acc_val[k]
      epoch_acc_test[k] = acc_test[k]
      
      r_loss:write(tostring(epoch_loss_real[k]), '\n')
      s_loss:write(tostring(epoch_loss_synth[k]), '\n')
      
      r_acc:write(tostring(epoch_acc_real[k]), '\n')
      s_acc:write(tostring(epoch_acc_synth[k]), '\n')
      v_acc:write(tostring(epoch_acc_val[k]), '\n')
      t_acc:write(tostring(epoch_acc_test[k]), '\n')
      
    end
    
    r_loss:close()
    s_loss:close()
    r_acc:close()
    s_acc:close()
    v_acc:close()
    t_acc:close()
    
  end

end

-- Save Raw Test Results
function save_raw_test(epoch, save_stats_gap, base_dir, im_folder, raw_test, num_test)

  if((epoch-1)%save_stats_gap == 0) then
  
    local filename_raw_test = base_dir .. 'raw_test.txt'
    
    raw_test_f = io.open(filename_raw_test, 'w')
    
    for k = 1, epoch do
      --raw_test_f:write(tostring(raw_test[{{k}}]), '\n')
      for t = 1, num_test do
        raw_test_f:write(tostring(raw_test[k][t]))
      end
      raw_test_f:write('\n')
    end
    
    raw_test_f:close()
    
  end

end

-- Save Learned Models
function save_abatch_models(abatch_id, epoch, base_dir, netPreGen, netGen, netRank)
  
  -- ARON: Not saving Generator since it's frozen anyway
  
  local netPreGenLoc = base_dir .. 'abatch' .. abatch_id .. '_model_pregen.dat'
  --local netGenLoc = base_dir .. 'abatch' .. abatch_id .. '_model_gen.dat'
  local netRankLoc = base_dir .. 'abatch' .. abatch_id .. '_model_rank.dat'
  
  torch.save(netPreGenLoc, netPreGen:clearState())
  --torch.save(netGenLoc, netGen)
  torch.save(netRankLoc, netRank:clearState())
  
  -- NOTE: Can't do clearState on netGen
  --torch.save(netPreGenLoc, netPreGen:clearState())
  --torch.save(netGenLoc, netGen:clearState())
  --torch.save(netRankLoc, netRank:clearState())
  
end

function save_trained_models(epoch, base_dir, netPreGen, netGen, netRank)

  local netPreGenLoc = base_dir .. '/learned_model_pregen' .. (epoch) .. '.dat'
  local netGenLoc = base_dir .. '/learned_model_gen' .. (epoch) .. '.dat'
  local netRankLoc = base_dir .. '/learned_model_rank' .. (epoch) .. '.dat'
  
  torch.save(netPreGenLoc, netPreGen:clearState())
  torch.save(netGenLoc, netGen)
  torch.save(netRankLoc, netRank:clearState())
  
end

function save_trained_models_ranker(epoch, base_dir, netRank)

  local netRankLoc = base_dir .. '/learned_model_rank' .. (epoch) .. '.dat'
  torch.save(netRankLoc, netRank:clearState())
  
end

