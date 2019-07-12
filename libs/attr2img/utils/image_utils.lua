-- code adapted from eyescream project
require 'torch'
require 'image'

local image_utils = {}

function image_utils.normalize(data, mean_, std_)
  local mean = mean_ or data:mean(1)
  local std = std_ or data:std(1, true)
  local eps = 1e-7
  for i=1,data:size(1) do
    data[i]:add(-1, mean)
    data[i]:cdiv(std + eps)
  end
  return mean, std
end

function image_utils.normalizeGlobal(data, mean_, std_)
  local std = std_ or data:std()
  local mean = mean_ or data:mean()
  data:add(-mean)
  data:mul(1/std)
  return mean, std
end

function image_utils.contrastNormalize(data, new_min, new_max, old_min_, old_max_)
  local old_max = old_max_ or data:max(1)
  local old_min = old_min_ or data:min(1)
  local eps = 1e-7
  for i=1,data:size(1) do
    data[i]:add(-1, old_min)
    data[i]:mul(new_max - new_min)
    data[i]:cdiv(old_max - old_min + eps)
    data[i]:add(new_min)
  end
  return old_min, old_max
end

function image_utils.flip(data, labels)
  local n = data:size(1)
  local N = n*2
  local new_data = torch.Tensor(N, data:size(2), data:size(3), data:size(4)):typeAs(data)
  local new_labels = torch.Tensor(N)
  new_data[{{1,n}}] = data
  new_labels[{{1,n}}] = labels:clone()
  new_labels[{{n+1,N}}] = labels:clone()
  for i = n+1,N do
    new_data[i] = image.hflip(data[i-n])
  end
  local rp = torch.LongTensor(N)
  rp:randperm(N)
  return new_data:index(1, rp), new_labels:index(1, rp)
end

function image_utils.translate2(data, w, labels)
  local n = data:size(1)
  local N = n*5
  local ow = data:size(3)
  local new_data = torch.Tensor(N, data:size(2), w, w):typeAs(data)
  local new_labels = torch.Tensor(N, labels:size(2))
  local d = ow - w + 1
  local m1 = (ow - w) / 2 + 1
  local m2 = ow - ((ow - w) / 2)
  local x1 = {1, d, 1, d, m1} 
  local x2 = {w, ow, w, ow, m2}
  local y1 = {1, 1, d, d, m1}
  local y2 = {w, w, ow, ow, m2}
  local k = 1
  for i = 1,n do
    for j = 1,5 do
      new_data[k] = data[{ i, {}, {y1[j], y2[j]}, {x1[j], x2[j]} }]:clone()
      new_labels[k] = labels[{i, {}}]
      k = k + 1
    end
  end
  local rp = torch.LongTensor(N)

  rp:randperm(N)
  return new_data:index(1, rp), new_labels:index(1, rp)
  --return new_data:index(1, rp), new_labels:index(1, rp)
end

--function image_utils.translate(data, w, labels)
function image_utils.translate(data, w)
  local n = data:size(1)
  local N = n*5
  local ow = data:size(3)
  local new_data = torch.Tensor(N, data:size(2), w, w):typeAs(data)
  --local new_labels = torch.Tensor(N, labels:size(2))
  local d = ow - w + 1
  local m1 = (ow - w) / 2 + 1
  local m2 = ow - ((ow - w) / 2)
  local x1 = {1, d, 1, d, m1} 
  local x2 = {w, ow, w, ow, m2}
  local y1 = {1, 1, d, d, m1}
  local y2 = {w, w, ow, ow, m2}
  local k = 1
  for i = 1,n do
    for j = 1,5 do
      new_data[k] = data[{ i, {}, {y1[j], y2[j]}, {x1[j], x2[j]} }]:clone()
      --new_labels[k] = labels[{i, {}}]
      k = k + 1
    end
  end
  local rp = torch.LongTensor(N)
  if perm == 0 then
    rp:randperm(N)
    return new_data:index(1, rp)
  else
    return new_data
  end
  --return new_data:index(1, rp), new_labels:index(1, rp)
end

function image_utils.coloraug(data, beginv, endv, stepv)
  local n = data:size(1)
  local w = data:size(3)
  local nsample = math.floor((endv - beginv)/stepv) + 1
  local N = n * nsample * nsample * nsample
  local new_data = torch.FloatTensor(N, data:size(2), w, w):typeAs(data)
  local k = 1
  for i= 1, n do
    local img = data[{{i}, {}, {}, {}}]:float():clone()
    img = img:squeeze()
    for jR = 1, nsample do
      for jG = 1, nsample do
        for jB = 1, nsample do
          local img2 = img:clone()
          img2[{{1}, {}, {}}]:mul((jR-1)*stepv+beginv)
          img2[{{2}, {}, {}}]:mul((jG-1)*stepv+beginv)
          img2[{{3}, {}, {}}]:mul((jB-1)*stepv+beginv)
          new_data[{{k}, {}, {}, {}}] = img2
          k = k + 1
        end
      end
    end
  end
  local rp = torch.LongTensor(N)
  rp:randperm(N)
  return new_data:index(1, rp)
end

function image_utils.sharpen(data, beginv, endv, stepv)
  local n = data:size(1)
  local w = data:size(3)
  local nsample = math.floor((endv - beginv)/stepv) + 1
  local N = n * nsample
  local new_data = torch.FloatTensor(N, data:size(2), w, w):typeAs(data)
  local k = 1
  local Kernel = torch.FloatTensor(5, 5):fill(1/25)
  for i = 1, n do
    local img = data[{{i}, {}, {}, {}}]:float():clone()
    img = img:squeeze()
    local img_blurred = image.convolve(img, Kernel, 'same')
    local residue = torch.add(img, -1, img_blurred)
    for j = 1, nsample do
      local ratio = beginv + (j-1)*stepv
      local img_filtered = torch.add(img, ratio, residue)
      new_data[{{k}, {}, {}, {}}] = img_filtered
      k = k + 1
    end
  end
  local rp = torch.LongTensor(N)
  rp:randperm(N)
  return new_data:index(1, rp)
end

function image_utils.crop(data, w)
  local N = data:size(1)
  local ow = data:size(3)
  local new_data = torch.Tensor(N, data:size(2), w, w):typeAs(data)
  local d = ow - w + 1
  local m1 = (ow - w) / 2 + 1
  local m2 = ow - ((ow - w) / 2)
  local k = 1
  for i = 1, N do
    new_data[k] = data[{i, {}, {m1, m2}, {m1, m2}}]:clone()
    k = k + 1
  end
  return new_data
end

return image_utils
