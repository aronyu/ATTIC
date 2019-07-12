-- Code adapted from https://github.com/y0ast/VAE-Torch
-- Based on JoinTable module

require 'nn'
require 'cunn'
local Reparametrize, parent = torch.class('nn.Reparametrize', 'nn.Module')

function Reparametrize:__init(dimension)
    parent.__init(self)
    self.size = torch.LongStorage()
    self.dimension = dimension
    self.gradInput = {}
    self.rng = torch.Tensor(1, self.dimension):randn(1, self.dimension)
end 

function Reparametrize:updateOutput(input)
    --Different eps for whole batch, or one and broadcast?
    self.eps = torch.randn(input[2]:size(1),self.dimension)
    
    -- ARON MOD (12/12/17)
    if input[2]:type() == "torch.CudaTensor" then
      self.output = torch.mul(input[2]:float(),0.5):exp():cmul(self.eps):cuda()
    else
      self.output = torch.mul(input[2],0.5):exp():cmul(self.eps)
    end
    --self.output = torch.mul(input[2],0.5):exp():cmul(self.eps)
    
    -- Add the mean
    self.output:add(input[1])

    return self.output
end

function Reparametrize:updateGradInput(input, gradOutput)
    -- Derivative with respect to mean is 1
    self.gradInput[1] = gradOutput:clone()
    
    -- ARON MOD (12/12/17)
    if input[2]:type() == "torch.CudaTensor" then
      self.gradInput[2] = torch.mul(input[2]:float(),0.5):exp():mul(0.5):cmul(self.eps):cuda()
      self.gradInput[2]:cmul(gradOutput)
    else
      self.gradInput[2] = torch.mul(input[2],0.5):exp():mul(0.5):cmul(self.eps)
      self.gradInput[2]:cmul(gradOutput)
    end
    
    --Not sure if this gradient is right
    --self.gradInput[2] = torch.mul(input[2],0.5):exp():mul(0.5):cmul(self.eps)
    --self.gradInput[2]:cmul(gradOutput)

    return self.gradInput
end
