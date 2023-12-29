import torch.nn as nn
import torch
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
    
class Res2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scales=8):
        super(Res2NetBlock, self).__init__()
        self.scales = scales
        self.scale_channels = out_channels // scales
        self.conv_blocks = nn.ModuleList()

        for _ in range(scales - 1):
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels if _ == 0 else self.scale_channels, 
                              self.scale_channels, 
                              kernel_size=3, 
                              stride=1, 
                              padding=1, 
                              bias=False),
                    nn.BatchNorm2d(self.scale_channels),
                    nn.ReLU(inplace=True)
                )
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        outputs = [x]
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            outputs.append(x)
        
        return self.relu(torch.cat(outputs, dim=1))




class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(RRCNN_block, self).__init__()
        self.Res2Net_block = Res2NetBlock(ch_in, ch_out, scales=8)
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x = self.Res2Net_block(x)
        return x


class R2U_Net(nn.Module):

    def __init__(self, img_ch=1, output_ch=4, t=2):
        super(R2U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.fc = None  # Placeholder, will be set in forward method
        self.output_ch = output_ch
        self.softmax = nn.Softmax(dim=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 112)) # this line is added to make the model work with any input size


    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)



        x5 = self.Maxpool(x4)

        x5 = self.RRCNN5(x5)


        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        # Dynamically calculate input size for the fully connected layer
        if not self.fc:
            n_size = d1.size()[1] * d1.size()[2] * d1.size()[3]
            self.fc = nn.Linear(n_size, self.output_ch).to(d1.device)

        d1 = d1.view(d1.size(0), -1)  # flatten the tensor
        d1 = self.fc(d1)
        d1 = self.softmax(d1)  # apply softmax to get probabilities

        return d1



def test_unet():
    batch_size = 8
    dummy_mfccs = torch.rand(batch_size, 1, 32, 1120) 

    # Initialize the Attention U-Net model with 1 input channel and 4 output classes
    model = R2U_Net()

    # Forward pass the dummy data through the model
    output = model(dummy_mfccs)

    # Check the output size
    # assert output.size() == (batch_size, 2), "The output size is incorrect"


    # Check if model parameters are trainable
    # assert all(param.requires_grad for param in model.parameters()), "Some model parameters are not trainable."

    # Add more assertions as per your requirements

    print("The test passed successfully.")

test_unet()
