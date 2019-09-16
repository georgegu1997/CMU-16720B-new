import numpy as np
import torchvision
import util
import matplotlib.pyplot as plt
import visual_words
import visual_recog
import deep_recog
import skimage

def getVisualWordAndSave(path_img, save_path):
    image = skimage.io.imread(path_img)
    image = image.astype('float')/255
    dictionary = np.load('dictionary.npy')
    img = visual_words.get_visual_words(image,dictionary)
    plt.imsave(save_path + "image.png", image)
    plt.imsave(save_path + "_wordmap.png", img, cmap=plt.get_cmap('gist_rainbow'))

if __name__ == '__main__':
    num_cores = util.get_num_CPU()

    '''Load an example image'''
    path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
    image = skimage.io.imread(path_img)
    image = image.astype('float')/255

    '''q1.1.2 get filter responses and display them'''
    # filter_responses = visual_words.extract_filter_responses(image)
    # util.display_filter_responses(filter_responses)

    '''q1.2 Compute the dictionary'''
    # visual_words.compute_dictionary(num_workers=num_cores)

    '''q1.3 Get visual codes using the dictionary'''
    # path_img = "../data/desert/sun_bvlihuzwolttdrnn.jpg"
    # getVisualWordAndSave(path_img, "../results/q1_3_first")

    '''q2.1 test get_feature_from_wordmap'''
    # dictionary = np.load('dictionary.npy')
    # img = visual_words.get_visual_words(image,dictionary)
    # hist = visual_recog.get_feature_from_wordmap(img, dictionary.shape[0])
    # print(hist)
    # print(hist.shape)
    # print(hist.sum())

    '''q2.2 test get_feature_from_wordmap_SPM'''
    # dictionary = np.load('dictionary.npy')
    # img = visual_words.get_visual_words(image,dictionary)
    # hist = visual_recog.get_feature_from_wordmap_SPM(img, 3, dictionary.shape[0])

    '''q2.4 build a recognition system'''
    # visual_recog.build_recognition_system(num_workers=num_cores)

    '''q2.5 evaluate the recognition system'''
    # conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
    # print(conf)
    # print(accuracy)
    '''
    ============q2.5 Output===========
    [[14.  0.  0.  0.  0.  0.  0.  0.]
     [ 0. 14.  0.  1.  0.  1.  2.  0.]
     [ 0.  0. 15.  4.  3.  0.  0.  3.]
     [ 1.  2.  1. 17.  0.  0.  1.  4.]
     [ 1.  1.  0.  0. 10.  1.  0.  0.]
     [ 0.  2.  0.  1.  8. 12.  1.  0.]
     [ 0.  5.  0.  0.  2.  3. 11.  0.]
     [ 0.  2.  3.  2.  1.  1.  0. 10.]]
    0.64375
    '''

    '''q3.1 Test the network_layers'''
    import network_layers
    # import torch.nn
    # x = image
    # print("x.shape:", x.shape)

    '''test the multichannel_conv2d'''
    # weights = util.get_VGG16_weights()
    # # print(weights[0]) # Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # conv_w = weights[0][1]
    # conv_b = weights[0][2]
    # y = network_layers.multichannel_conv2d(x, conv_w, conv_b)
    # vgg16 = torchvision.models.vgg16(pretrained=True).double()
    # conv2d = vgg16.features[0]
    # x_torch = torch.from_numpy(np.expand_dims(x.transpose(2,0,1), axis=0))
    # y_torch = conv2d(x_torch)
    # y_torch = y_torch.detach().numpy()
    # y_torch = y_torch[0].transpose((1, 2, 0))

    '''test the max_pool2d'''
    # y = network_layers.max_pool2d(x, 2)
    # pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    # y_torch = pool(torch.from_numpy(x.transpose((2, 0, 1))))
    # y_torch = y_torch.detach().numpy()
    # y_torch = y_torch.transpose((1, 2, 0))

    '''The metrics for comparison'''
    # print("y.shape:", y.shape)
    # print("y_torch.shape:", y_torch.shape)
    # print(np.linalg.norm(y-y_torch))
    # print(np.absolute(y-y_torch).max())

    '''test network_layers.extract_deep_feature'''
    x = image
    weights = util.get_VGG16_weights()
    y = network_layers.extract_deep_feature(x, weights)
    print(y.shape)
    

    # vgg16 = torchvision.models.vgg16(pretrained=True).double()
    # print(vgg16)

    # vgg16 = torchvision.models.vgg16(pretrained=True).double()
    # vgg16.eval()
    # deep_recog.build_recognition_system(vgg16,num_workers=num_cores//2)

    #conf = deep_recog.evaluate_recognition_system(vgg16,num_workers=num_cores//2)
    #print(conf)
    #print(np.diag(conf).sum()/conf.sum())
