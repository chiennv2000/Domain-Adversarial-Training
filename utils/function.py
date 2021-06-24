from torch.autograd import Function

class ReversalGradient(Function):
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grads):
        output = grads.neg() * ctx.alpha
        return output, None