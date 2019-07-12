local MulConstantT, parent = torch.class('nn.MulConstantT', 'nn.Module')

-- Modified version of MulConstant, where a constant tensor is multiplied as an input

function MulConstantT:__init(constant_tensor)
  parent.__init(self)
  self.constant_tensor = constant_tensor
end

function MulConstantT:updateOutput(input)
  self.mul_matrix = torch.repeatTensor(self.constant_tensor, input:size(1), 1)   -- duplicate constant 2D tensor
  self.output:resizeAs(input)
  self.output:copy(input)
  self.output:cmul(self.mul_matrix)
  return self.output
end

function MulConstantT:updateGradInput(input, gradOutput)
  if self.gradInput then
    self.gradInput:resizeAs(gradOutput)
    self.gradInput:copy(gradOutput)
    self.gradInput:cmul(self.mul_matrix)
    return self.gradInput
  end
end