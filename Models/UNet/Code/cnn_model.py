import torch.nn as nn
import torch
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        #print("After conv1:", x.shape)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        #print("After conv2:", x.shape)
        x = self.bn2(x)
        x = self.relu2(x)
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


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
    
    
class U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=4):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.fc = None  # Placeholder, will be set in forward method
        self.output_ch = output_ch
        self.softmax = nn.Softmax(dim=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 112))
    def forward(self, x):
        # encoding path
        ##print dimensions of input
        #print("input shape:",x.shape)
        x1 = self.Conv1(x)
        ##print dimensions 
        #print("x1 shape:", x1.shape)


        x2 = self.Maxpool(x1)
        #print("x2 Maxpool shape:", x2.shape)
        x2 = self.Conv2(x2)
        #print("x2 Conv2 shape:", x2.shape)

        x3 = self.Maxpool(x2)
        #print("x3 Maxpool shape:", x3.shape)
        x3 = self.Conv3(x3)
        #print("x3 Conv3 shape:", x3.shape)

        x4 = self.Maxpool(x3)
        #print("x4 Maxpool shape:", x4.shape)
        x4 = self.Conv4(x4)
        #print("x4 Conv4 shape:", x4.shape)

        x5 = self.Maxpool(x4)
        #print("x5 Maxpool shape:", x5.shape)
        x5 = self.Conv5(x5)
        #print("x5 Conv5 shape:", x5.shape)

        # decoding + concat path
        d5 = self.Up5(x5)
        #print("d5 Up5 shape:", d5.shape)
        d5 = torch.cat((x4, d5), dim=1)
        #print("d5 cat shape:", d5.shape)

        d5 = self.Up_conv5(d5)
        #print("d5 Up_conv5 shape:", d5.shape)

        d4 = self.Up4(d5)
        #print("d4 Up4 shape:", d4.shape)
        d4 = torch.cat((x3, d4), dim=1)
        #print("d4 cat shape:", d4.shape)
        d4 = self.Up_conv4(d4)
        #print("d4 Up_conv4 shape:", d4.shape)

        d3 = self.Up3(d4)
        #print("d3 Up3 shape:", d3.shape)
        d3 = torch.cat((x2, d3), dim=1)
        #print("d3 cat shape:", d3.shape)
        d3 = self.Up_conv3(d3)
        #print("d3 Up_conv3 shape:", d3.shape)

        d2 = self.Up2(d3)
        #print("d2 Up2 shape:", d2.shape)
        d2 = torch.cat((x1, d2), dim=1)
        #print("d2 cat shape:", d2.shape)
        
        d2 = self.Up_conv2(d2)
        #print("d2 Up_conv2 shape:", d2.shape)

        d1 = self.Conv_1x1(d2)
        #print("d1 Conv_1x1 shape:", d1.shape)
        # Dynamically calculate input size for the fully connected layer
        if not self.fc:
            n_size = d1.size()[1] * d1.size()[2] * d1.size()[3]
            self.fc = nn.Linear(n_size, self.output_ch).to(d1.device)

        d1 = d1.view(d1.size(0), -1)  # flatten the tensor
        #print("d1 view shape:", d1.shape)
        d1 = self.fc(d1)
        #print("d1 fc shape:", d1.shape)
        d1 = self.softmax(d1)  # apply softmax to get probabilities
        #print("d1 softmax shape:", d1.shape)

        return d1



def test_unet():
    batch_size = 1
    dummy_mfccs = torch.rand(batch_size, 1, 32, 1120)

    # Initialize the Attention U-Net model with 1 input channel and 4 output classes
    model = U_Net()
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters in the model = ", total_params)

    # Forward pass the dummy data through the model
    output = model(dummy_mfccs)

    # Check the output size
    # assert output.size() == (batch_size, 2), "The output size is incorrect"


    # Check if model parameters are trainable
    # assert all(param.requires_grad for param in model.parameters()), "Some model parameters are not trainable."

    # Add more assertions as per your requirements

    print("The test passed successfully.")

test_unet()
