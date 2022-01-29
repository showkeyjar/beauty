import sys
sys.path.append('./kaffe')
sys.path.append('./kaffe/tensorflow')
#from kaffe.tensorflow.network_allNonTrain import Network
from network_shape import Network_Shape



class ResNet_101(Network_Shape):
    def setup(self):
        (self.feed('input')
             .conv(7, 7, 64, 2, 2, biased=False, relu=False, name='conv1')
             .batch_normalization(relu=True, name='bn_conv1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
             .batch_normalization(name='bn2a_branch1'))

        (self.feed('pool1')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
             .batch_normalization(relu=True, name='bn2a_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
             .batch_normalization(relu=True, name='bn2a_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
             .batch_normalization(name='bn2a_branch2c'))

        (self.feed('bn2a_branch1', 
                   'bn2a_branch2c')
             .add(name='res2a')
             .relu(name='res2a_relu') # batch_size x 56 x 56 x 256
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a') # batch_size x 56 x 56 x 64
             .batch_normalization(relu=True, name='bn2b_branch2a') # batch_size x 56 x 56 x 64
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b') # batch_size x 56 x 56 x 64
             .batch_normalization(relu=True, name='bn2b_branch2b') # batch_size x 56 x 56 x 64
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c') # batch_size x 56 x 56 x 256
             .batch_normalization(name='bn2b_branch2c')) # batch_size x 56 x 56 x 256

        (self.feed('res2a_relu', # batch_size x 56 x 56 x 256
                   'bn2b_branch2c') # batch_size x 56 x 56 x 256
             .add(name='res2b')
             .relu(name='res2b_relu') # batch_size x 56 x 56 x 256
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
             .batch_normalization(relu=True, name='bn2c_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
             .batch_normalization(relu=True, name='bn2c_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c')
             .batch_normalization(name='bn2c_branch2c'))

        (self.feed('res2b_relu', 
                   'bn2c_branch2c')
             .add(name='res2c')
             .relu(name='res2c_relu') # batch_size x 56 x 56 x 256
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1')
             .batch_normalization(name='bn3a_branch1'))

        (self.feed('res2c_relu') # batch_size x 56 x 56 x 256
             .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a')
             .batch_normalization(relu=True, name='bn3a_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
             .batch_normalization(relu=True, name='bn3a_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
             .batch_normalization(name='bn3a_branch2c'))

        (self.feed('bn3a_branch1', 
                   'bn3a_branch2c')
             .add(name='res3a')
             .relu(name='res3a_relu') # batch_size x 28 x 28 x 512
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2a')
             .batch_normalization(relu=True, name='bn3b1_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2b')
             .batch_normalization(relu=True, name='bn3b1_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b1_branch2c')
             .batch_normalization(name='bn3b1_branch2c'))

        (self.feed('res3a_relu', 
                   'bn3b1_branch2c')
             .add(name='res3b1')
             .relu(name='res3b1_relu') # batch_size x 28 x 28 x 512
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2a')
             .batch_normalization(relu=True, name='bn3b2_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2b')
             .batch_normalization(relu=True, name='bn3b2_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b2_branch2c')
             .batch_normalization(name='bn3b2_branch2c'))

        (self.feed('res3b1_relu', 
                   'bn3b2_branch2c')
             .add(name='res3b2')
             .relu(name='res3b2_relu') # batch_size x 28 x 28 x 512
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2a')
             .batch_normalization(relu=True, name='bn3b3_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2b')
             .batch_normalization(relu=True, name='bn3b3_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b3_branch2c')
             .batch_normalization(name='bn3b3_branch2c'))

        (self.feed('res3b2_relu', 
                   'bn3b3_branch2c')
             .add(name='res3b3')
             .relu(name='res3b3_relu') # batch_size x 28 x 28 x 512
             .conv(1, 1, 1024, 2, 2, biased=False, relu=False, name='res4a_branch1')
             .batch_normalization(name='bn4a_branch1'))

        (self.feed('res3b3_relu') # batch_size x 28 x 28 x 512
             .conv(1, 1, 256, 2, 2, biased=False, relu=False, name='res4a_branch2a')
             .batch_normalization(relu=True, name='bn4a_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4a_branch2b')
             .batch_normalization(relu=True, name='bn4a_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c')
             .batch_normalization(name='bn4a_branch2c'))

        (self.feed('bn4a_branch1', 
                   'bn4a_branch2c')
             .add(name='res4a')
             .relu(name='res4a_relu') # batch_size x 14 x 14 x 1024
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b1_branch2a')
             .batch_normalization(relu=True, name='bn4b1_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b1_branch2b')
             .batch_normalization(relu=True, name='bn4b1_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b1_branch2c')
             .batch_normalization(name='bn4b1_branch2c'))

        (self.feed('res4a_relu', 
                   'bn4b1_branch2c')
             .add(name='res4b1')
             .relu(name='res4b1_relu') # batch_size x 14 x 14 x 1024
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b2_branch2a')
             .batch_normalization(relu=True, name='bn4b2_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b2_branch2b')
             .batch_normalization(relu=True, name='bn4b2_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b2_branch2c')
             .batch_normalization(name='bn4b2_branch2c'))

        (self.feed('res4b1_relu', 
                   'bn4b2_branch2c')
             .add(name='res4b2')
             .relu(name='res4b2_relu') # batch_size x 14 x 14 x 1024
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b3_branch2a')
             .batch_normalization(relu=True, name='bn4b3_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b3_branch2b')
             .batch_normalization(relu=True, name='bn4b3_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b3_branch2c')
             .batch_normalization(name='bn4b3_branch2c'))

        (self.feed('res4b2_relu', 
                   'bn4b3_branch2c')
             .add(name='res4b3')
             .relu(name='res4b3_relu') # batch_size x 14 x 14 x 1024
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b4_branch2a')
             .batch_normalization(relu=True, name='bn4b4_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b4_branch2b')
             .batch_normalization(relu=True, name='bn4b4_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b4_branch2c')
             .batch_normalization(name='bn4b4_branch2c'))

        (self.feed('res4b3_relu', 
                   'bn4b4_branch2c')
             .add(name='res4b4')
             .relu(name='res4b4_relu') # batch_size x 14 x 14 x 1024
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b5_branch2a')
             .batch_normalization(relu=True, name='bn4b5_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b5_branch2b')
             .batch_normalization(relu=True, name='bn4b5_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b5_branch2c')
             .batch_normalization(name='bn4b5_branch2c'))

        (self.feed('res4b4_relu', 
                   'bn4b5_branch2c')
             .add(name='res4b5')
             .relu(name='res4b5_relu') # batch_size x 14 x 14 x 1024
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b6_branch2a')
             .batch_normalization(relu=True, name='bn4b6_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b6_branch2b')
             .batch_normalization(relu=True, name='bn4b6_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b6_branch2c')
             .batch_normalization(name='bn4b6_branch2c'))

        (self.feed('res4b5_relu', 
                   'bn4b6_branch2c')
             .add(name='res4b6')
             .relu(name='res4b6_relu') # batch_size x 14 x 14 x 1024
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b7_branch2a')
             .batch_normalization(relu=True, name='bn4b7_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b7_branch2b')
             .batch_normalization(relu=True, name='bn4b7_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b7_branch2c')
             .batch_normalization(name='bn4b7_branch2c'))

        (self.feed('res4b6_relu', 
                   'bn4b7_branch2c')
             .add(name='res4b7')
             .relu(name='res4b7_relu') # batch_size x 14 x 14 x 1024
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b8_branch2a')
             .batch_normalization(relu=True, name='bn4b8_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b8_branch2b')
             .batch_normalization(relu=True, name='bn4b8_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b8_branch2c')
             .batch_normalization(name='bn4b8_branch2c'))

        (self.feed('res4b7_relu', 
                   'bn4b8_branch2c')
             .add(name='res4b8')
             .relu(name='res4b8_relu') # batch_size x 14 x 14 x 1024
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b9_branch2a')
             .batch_normalization(relu=True, name='bn4b9_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b9_branch2b')
             .batch_normalization(relu=True, name='bn4b9_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b9_branch2c')
             .batch_normalization(name='bn4b9_branch2c'))

        (self.feed('res4b8_relu', 
                   'bn4b9_branch2c')
             .add(name='res4b9')
             .relu(name='res4b9_relu') # batch_size x 14 x 14 x 1024
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b10_branch2a')
             .batch_normalization(relu=True, name='bn4b10_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b10_branch2b')
             .batch_normalization(relu=True, name='bn4b10_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b10_branch2c')
             .batch_normalization(name='bn4b10_branch2c'))

        (self.feed('res4b9_relu', 
                   'bn4b10_branch2c')
             .add(name='res4b10')
             .relu(name='res4b10_relu') # batch_size x 14 x 14 x 1024
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b11_branch2a')
             .batch_normalization(relu=True, name='bn4b11_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b11_branch2b')
             .batch_normalization(relu=True, name='bn4b11_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b11_branch2c')
             .batch_normalization(name='bn4b11_branch2c'))

        (self.feed('res4b10_relu', 
                   'bn4b11_branch2c')
             .add(name='res4b11')
             .relu(name='res4b11_relu') # batch_size x 14 x 14 x 1024
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b12_branch2a')
             .batch_normalization(relu=True, name='bn4b12_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b12_branch2b')
             .batch_normalization(relu=True, name='bn4b12_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b12_branch2c')
             .batch_normalization(name='bn4b12_branch2c'))

        (self.feed('res4b11_relu', 
                   'bn4b12_branch2c')
             .add(name='res4b12')
             .relu(name='res4b12_relu') # batch_size x 14 x 14 x 1024
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b13_branch2a')
             .batch_normalization(relu=True, name='bn4b13_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b13_branch2b')
             .batch_normalization(relu=True, name='bn4b13_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b13_branch2c')
             .batch_normalization(name='bn4b13_branch2c'))

        (self.feed('res4b12_relu', 
                   'bn4b13_branch2c')
             .add(name='res4b13')
             .relu(name='res4b13_relu') # batch_size x 14 x 14 x 1024
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b14_branch2a')
             .batch_normalization(relu=True, name='bn4b14_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b14_branch2b')
             .batch_normalization(relu=True, name='bn4b14_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b14_branch2c')
             .batch_normalization(name='bn4b14_branch2c'))

        (self.feed('res4b13_relu', 
                   'bn4b14_branch2c')
             .add(name='res4b14')
             .relu(name='res4b14_relu') # batch_size x 14 x 14 x 1024
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b15_branch2a')
             .batch_normalization(relu=True, name='bn4b15_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b15_branch2b')
             .batch_normalization(relu=True, name='bn4b15_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b15_branch2c')
             .batch_normalization(name='bn4b15_branch2c'))

        (self.feed('res4b14_relu', 
                   'bn4b15_branch2c')
             .add(name='res4b15')
             .relu(name='res4b15_relu') # batch_size x 14 x 14 x 1024
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b16_branch2a')
             .batch_normalization(relu=True, name='bn4b16_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b16_branch2b')
             .batch_normalization(relu=True, name='bn4b16_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b16_branch2c')
             .batch_normalization(name='bn4b16_branch2c'))

        (self.feed('res4b15_relu', 
                   'bn4b16_branch2c')
             .add(name='res4b16')
             .relu(name='res4b16_relu') # batch_size x 14 x 14 x 1024
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b17_branch2a')
             .batch_normalization(relu=True, name='bn4b17_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b17_branch2b')
             .batch_normalization(relu=True, name='bn4b17_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b17_branch2c')
             .batch_normalization(name='bn4b17_branch2c'))

        (self.feed('res4b16_relu', 
                   'bn4b17_branch2c')
             .add(name='res4b17')
             .relu(name='res4b17_relu') # batch_size x 14 x 14 x 1024
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b18_branch2a')
             .batch_normalization(relu=True, name='bn4b18_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b18_branch2b')
             .batch_normalization(relu=True, name='bn4b18_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b18_branch2c')
             .batch_normalization(name='bn4b18_branch2c'))

        (self.feed('res4b17_relu', 
                   'bn4b18_branch2c')
             .add(name='res4b18')
             .relu(name='res4b18_relu') # batch_size x 14 x 14 x 1024
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b19_branch2a')
             .batch_normalization(relu=True, name='bn4b19_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b19_branch2b')
             .batch_normalization(relu=True, name='bn4b19_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b19_branch2c')
             .batch_normalization(name='bn4b19_branch2c'))

        (self.feed('res4b18_relu', 
                   'bn4b19_branch2c')
             .add(name='res4b19')
             .relu(name='res4b19_relu') # batch_size x 14 x 14 x 1024
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b20_branch2a')
             .batch_normalization(relu=True, name='bn4b20_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b20_branch2b')
             .batch_normalization(relu=True, name='bn4b20_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b20_branch2c')
             .batch_normalization(name='bn4b20_branch2c'))

        (self.feed('res4b19_relu', 
                   'bn4b20_branch2c')
             .add(name='res4b20')
             .relu(name='res4b20_relu') # batch_size x 14 x 14 x 1024
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b21_branch2a')
             .batch_normalization(relu=True, name='bn4b21_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b21_branch2b')
             .batch_normalization(relu=True, name='bn4b21_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b21_branch2c')
             .batch_normalization(name='bn4b21_branch2c'))

        (self.feed('res4b20_relu', 
                   'bn4b21_branch2c')
             .add(name='res4b21')
             .relu(name='res4b21_relu') # batch_size x 14 x 14 x 1024
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b22_branch2a')
             .batch_normalization(relu=True, name='bn4b22_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b22_branch2b')
             .batch_normalization(relu=True, name='bn4b22_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b22_branch2c')
             .batch_normalization(name='bn4b22_branch2c'))

        (self.feed('res4b21_relu', 
                   'bn4b22_branch2c')
             .add(name='res4b22')
             .relu(name='res4b22_relu') # batch_size x 14 x 14 x 1024
             .conv(1, 1, 2048, 2, 2, biased=False, relu=False, name='res5a_branch1')
             .batch_normalization(name='bn5a_branch1'))

        (self.feed('res4b22_relu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res5a_branch2a')
             .batch_normalization(relu=True, name='bn5a_branch2a')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5a_branch2b')
             .batch_normalization(relu=True, name='bn5a_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c')
             .batch_normalization(name='bn5a_branch2c'))

        (self.feed('bn5a_branch1', 
                   'bn5a_branch2c')
             .add(name='res5a')
             .relu(name='res5a_relu') # batch_size x 7 x 7 x 2048
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
             .batch_normalization(relu=True, name='bn5b_branch2a')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5b_branch2b')
             .batch_normalization(relu=True, name='bn5b_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c')
             .batch_normalization(name='bn5b_branch2c'))

        (self.feed('res5a_relu', 
                   'bn5b_branch2c')
             .add(name='res5b')
             .relu(name='res5b_relu') # batch_size x 7 x 7 x 2048
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a')
             .batch_normalization(relu=True, name='bn5c_branch2a')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5c_branch2b')
             .batch_normalization(relu=True, name='bn5c_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c')
             .batch_normalization(name='bn5c_branch2c'))

        (self.feed('res5b_relu', 
                   'bn5c_branch2c')
             .add(name='res5c')
             .relu(name='res5c_relu') # batch_size x 7 x 7 x 2048
             .avg_pool(7, 7, 1, 1, padding='VALID', name='pool5'))
             #.fc(198, relu=False, name='fc_ftnew'))
