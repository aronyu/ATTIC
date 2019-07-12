require 'paths'
require 'image'
zap50k = {}

zap50k.path_dataset = ''  -- MOD: path to training data for image generator
zap50k.scale = 64

function zap50k.setScale(scale)
  zap50k.scale = scale
end

function zap50k.loadTrainSet(start, stop)
  return zap50k.loadDataset(start, stop)
end

function zap50k.loadDataset(start, stop)

  datafile = torch.load(zap50k.path_dataset .. 'zap50k_train.t7')
  data = datafile.train_images
  attr = datafile.train_attributes

  local start = start or 1
  local stop = stop or data:size(1)
  data = data[{ {start, stop} }]
  attr = attr[{ {start, stop} }]
  local N = stop - start + 1

  local dataset = {}
  dataset.data = data:float()
  dataset.attr = attr:float()

  function dataset:scaleData()
    local N = dataset.data:size(1)
    dataset.scaled = torch.FloatTensor(N, 3, zap50k.scale, zap50k.scale)
    for n = 1, N do
      dataset.scaled[n] = image.scale(dataset.data[n], zap50k.scale, zap50k.scale)
    end
  end

  function dataset:size()
    local N = dataset.data:size(1)
    return N
  end

  setmetatable(dataset, {__index = function(self, index)
    local example = {self.scaled[index], self.attr[index]}
    return example
  end})

  return dataset
end

