local SpatialConvWP, parent = torch.class('cudnn.SpatialConvWP', 'cudnn.SpatialConvolution')

function SpatialConvWP:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
end

function SpatialConvWP:initialize(winit, binit, fanin)
  self.weight:normal(0,1):mul(winit):div(math.sqrt(fanin))
  self.bias:normal(0,1):mul(binit)
end


