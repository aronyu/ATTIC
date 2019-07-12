local AddConstantT, parent = torch.class('nn.AddConstantT', 'nn.Module')

-- Modified version of AddConstant, where a constant tensor is added as an input

function AddConstantT:__init(constant_tensor)
  parent.__init(self)
  self.constant_tensor = constant_tensor
end

function AddConstantT:updateOutput(input)
  local add_matrix = torch.repeatTensor(self.constant_tensor, input:size(1), 1)   -- duplicate constant 2D tensor
  self.output:resizeAs(input)
  self.output:copy(input)
  self.output:add(add_matrix) 
  return self.output
end

function AddConstantT:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput)
  self.gradInput:copy(gradOutput)
  return self.gradInput
end