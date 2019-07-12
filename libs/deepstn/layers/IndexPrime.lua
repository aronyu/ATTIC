local IndexPrime, parent = torch.class('nn.IndexPrime', 'nn.Module')

-- Modified version of Index, where the index parameter is passed in during initialization

function IndexPrime:__init(dimension, index)
    parent.__init(self)
    self.dimension = dimension
    self.index = index
end

-- Perform Index Switching (called on forward pass)
function IndexPrime:updateOutput(input)
    self.output:index(input, self.dimension, self.index)
    return self.output
end

function IndexPrime:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input):zero()
    self.gradInput:indexAdd(self.dimension, self.index, gradOutput)
    return self.gradInput
end

function IndexPrime:clearState()
    self.gradInput:set()
    self.output:set()
    return self
end