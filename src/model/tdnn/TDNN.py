import torch
import torch.nn as nn
import torch.nn.functional as F

class TDNN(nn.Module):
    
    def __init__(
                    self, 
                    input_dim=23, 
                    output_dim=512,
                    context_size=5,
                    stride=1,
                    dilation=1,
                    batch_norm=True,
                    dropout_p=0.0
                ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

        Affine transformation not applied globally to all frames but smaller windows with local context

        batch_norm: True to include batch normalisation after the non linearity
        
        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
      
        # self.kernel = nn.Linear(input_dim*context_size, output_dim)
        assert context_size % 2 == 1
        self.kernel = nn.Conv1d(input_dim, output_dim, kernel_size=context_size, stride=stride, 
                                padding=(context_size-1) // 2 * dilation, dilation=dilation) #add padding small fix for alignment
        self.nonlinearity = nn.ReLU()

        self.bn = nn.BatchNorm1d(output_dim) if self.batch_norm else lambda x: x
        self.drop = nn.Dropout(p=self.dropout_p) if self.dropout_p else lambda x: x
        
    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''
        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)

        x = x.transpose(1,2)
        x = self.kernel(x)
        x = self.nonlinearity(x)
        
        x = self.drop(x)
        x = self.bn(x)
        
        x = x.transpose(1,2)
        return x

    
if __name__=='__main__':
    inp = torch.rand(8, 500, 23)

    tdnn_layers = nn.Sequential(
        TDNN(23, 512, context_size=5, stride=1,dilation=1),
        TDNN(512, 512, context_size=3, stride=1,dilation=2), 
        TDNN(512, 512, context_size=3, stride=1,dilation=3), 
        TDNN(512, 128, context_size=1, stride=1,dilation=1)
    )
    inp = tdnn_layers(inp)
    print(inp.shape)
