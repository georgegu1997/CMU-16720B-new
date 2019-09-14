import numpy as np
import torchvision
import util
import matplotlib.pyplot as plt
import visual_words
import visual_recog
import deep_recog
import skimage

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
    path_img = "../data/desert/sun_bvlihuzwolttdrnn.jpg"
    image = skimage.io.imread(path_img)
    image = image.astype('float')/255

    dictionary = np.load('dictionary.npy')
    img = visual_words.get_visual_words(image,dictionary)
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(img, cmap=plt.get_cmap('gist_rainbow'))
    plt.show()

    # util.save_wordmap(wordmap, filename)
    # visual_recog.build_recognition_system(num_workers=num_cores)

    #conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
    #print(conf)
    #print(np.diag(conf).sum()/conf.sum())

    #vgg16 = torchvision.models.vgg16(pretrained=True).double()
    #vgg16.eval()
    #deep_recog.build_recognition_system(vgg16,num_workers=num_cores//2)
    #conf = deep_recog.evaluate_recognition_system(vgg16,num_workers=num_cores//2)
    #print(conf)
    #print(np.diag(conf).sum()/conf.sum())
