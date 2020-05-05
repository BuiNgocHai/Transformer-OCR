import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa


def get_augmenter(imgs, probs=0.5):
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    sometimes = lambda aug: iaa.Sometimes(probs, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    augment_sequences = iaa.Sequential(
        [
            # # apply the following augmenters to most images
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.12, 0.12), "y": (-0.12, 0.12)}, # translate by -20 to +20 percent (per axis)
                rotate=(-5, 5), # rotate by -45 to +45 degrees
                shear=(-13, 13), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode="constant" # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute one of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 4)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.03, 0.06), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    # iaa.Invert(0.05, per_channel=True), # invert color channels
                    # iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    # iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-4, 0),
                            first=iaa.Multiply((0.5, 1.5), per_channel=True),
                            second=iaa.LinearContrast((0.5, 2.0))
                        )
                    ]),
                    # iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    # iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=True
    )

    images_aug = augment_sequences(images=imgs)
    return images_aug