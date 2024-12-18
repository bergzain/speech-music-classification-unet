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
    
    
class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1

class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1
    
class R2U_Net(nn.Module):

    def __init__(self, img_ch=1, output_ch=2, t=2):
        super(R2U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

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

class R2AttU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=2,t=2):
        super(R2AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.fc = None  # Placeholder, will be set in forward method
        self.output_ch = output_ch
        self.softmax = nn.Softmax(dim=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 112)) # this line is added to make the model work with any input size



    def forward(self,x):
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
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
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


class AttentionUNet(nn.Module):
    def __init__(self, mfcc_dim=32, output_ch=2):
        super(AttentionUNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=1, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.fc = None  # Placeholder, will be set in forward method
        self.output_ch = output_ch
        self.softmax = nn.Softmax(dim=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 112))

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        # Dynamically calculate input size for the fully connected layer
        if not self.fc:
            n_size = d1.size()[1] * d1.size()[2] * d1.size()[3]
            self.fc = nn.Linear(n_size, self.output_ch).to(d1.device)

        d1 = d1.view(d1.size(0), -1)  # flatten the tensor
        d1 = self.fc(d1)
        d1 = self.softmax(d1)  # apply softmax to get probabilities

        return d1



class U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=2):
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
        # print("d1 softmax shape:", d1.shape)

        return d1



def test_models():
    batch_size = 1
    dummy_input = torch.rand(batch_size, 1, 80, 112)  # Adjust the input dimensions as needed

    models = {
        "U_Net": U_Net(),
        "R2U_Net": R2U_Net(),
        "R2AttU_Net": R2AttU_Net(),
        "AttentionUNet": AttentionUNet()
    }

    for model_name, model in models.items():
        print(f"Testing {model_name}...")

        # Print total number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters in {model_name} = {total_params}")

        # Forward pass the dummy data through the model
        output = model(dummy_input)

        # Print the output size
        print(f"Output shape of {model_name}: {output.shape}")

        # Check if model parameters are trainable
        assert all(param.requires_grad for param in model.parameters()), f"Some model parameters in {model_name} are not trainable."

        print(f"The test for {model_name} passed successfully.\n")

if __name__ == "__main__":
    test_models()
