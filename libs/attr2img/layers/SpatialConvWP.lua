local SpatialConvWP, parent = torch.class('nn.SpatialConvWP', 'nn.SpatialConvolution')

function SpatialConvWP:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
end

function SpatialConvWP:initialize(winit, binit, fanin)
  self.weight:normal(0,1):mul(winit):div(math.sqrt(fanin))
  self.bias:normal(0,1):mul(binit)
end

function SpatialConvWP:updateOutput(input)
  parent.updateOutput(input)
end
