# BoxNet

BoxNet is a deep residual convolutional network for particle picking and dirt masking in cryo-EM data. It's pre-trained on a lot of data and performs rather well.

The easiest way to use it is inside [Warp](https://github.com/cramerlab/warp). We are constantly re-training the model on the latest version of the training dataset. To benefit from that, grab the latest model from ftp://multiparticle.com/boxnet/models/. Models containing a "Mask" suffix are pre-trained to also mask out dirt in micrographs. In case this leads to bad results, try a model without masking.

The training dataset is too large to fit in this repository. It is hosted at ftp://multiparticle.com/boxnet/trainingdata/. The .zip file contains the entire corpus. To use it to re-train the model in [Warp](https://github.com/cramerlab/warp), put as many of the TIFF files as you want into the "boxnet2training" folder of your Warp installation.

To submit more data to the training repository, please use [this page](https://www.multiparticle.com/warp/?page_id=72).
