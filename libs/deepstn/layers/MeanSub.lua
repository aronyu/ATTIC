local MeanSub, parent = torch.class('nn.MeanSub', 'nn.Module')

-- Subtract the mean scalar from each color channel

function MeanSub:__init(mean_tensor)
   parent.__init(self)
   self.mean = mean_tensor
end

-- Forward Pass (subtract directly)
function MeanSub:updateOutput(input)
  
  -- Duplicate 3D Mean Tensor w.r.t. Batch Size
  local inputSize = input:size()
  local batch_mean = self.mean:repeatTensor(inputSize[1],1,1,1):reshape(inputSize)

  self.output:resizeAs(input)
  self.output:copy(input)
  self.output:add(-batch_mean)
  return self.output
end

-- Backward Pass (keep gradient the same)
function MeanSub:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput)
  self.gradInput:copy(gradOutput)
  return self.gradInput
end