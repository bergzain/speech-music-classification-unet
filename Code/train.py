from torch import nn
from torchsummary import summary

class CnnNetowrk(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(CnnNetowrk, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.batchnorm6 = nn.BatchNorm2d(512)
        self.batchnorm7 = nn.BatchNorm2d(1024)
        self.batchnorm8 = nn.BatchNorm2d(512)
        self.batchnorm9 = nn.BatchNorm2d(256)
        self.batchnorm10 = nn.BatchNorm2d(128)
        self.batchnorm11 = nn.BatchNorm2d(64)
        self.batchnorm12 = nn.BatchNorm2d(32)
        self.batchnorm13 = nn.BatchNorm2d(16)
        self.batchnorm14 = nn.BatchNorm2d(out_channels)
        self.t_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.t_conv5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.t_conv6 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.t_conv7 = nn.ConvTranspose2d(16, out_channels, kernel_size=2, stride=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #encoder
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.batchnorm3(self.conv3(x)))
        x = self.relu(self.batchnorm4(self.conv4(x)))
        x = self.pool(x)
        x = self.relu(self.batchnorm5(self.conv5(x)))
        x = self.relu(self.batchnorm6(self.conv6(x)))
        x = self.pool(x)
        x = self.relu(self.batchnorm7(self.conv7(x)))
        x = self.relu(self.batchnorm8(self.conv8(x)))
        x = self.pool(x)
        x = self.relu(self.batchnorm9(self.conv9(x)))
        x = self.relu(self.batchnorm10(self.conv10(x)))
        x = self.pool(x)
        x = self.relu(self.batchnorm11(self.conv11(x)))
        x = self.relu(self.batchnorm12(self.conv12(x)))
        x = self.pool(x)
        x = self.relu(self.batchnorm13(self.conv13(x)))
        x = self.relu(self.batchnorm14(self.conv14(x)))
        x = self.pool(x)
        x = self.softmax(x)
        return x

    #decoder
    def backward(self, x):
        x = self.relu(self.t_conv1(x))
        x = self.relu(self.t_conv2(x))
        x = self.relu(self.t_conv3(x))
        x = self.relu(self.t_conv4(x))
        x = self.relu(self.t_conv5(x))
        x = self.relu(self.t_conv6(x))
        x = self.relu(self.t_conv7(x))
        return x

    def predict(self, x):
        #predict the class of the input
        x = self.forward(x)
        x = torch.argmax(x, dim=1)
        return x

    def save(self, path):
        #save the model
        torch.save(self.state_dict(), path)



if __name__ == "main":
    cnn = CnnNetowrk()
    summary(cnn, (1, 64, 44))
