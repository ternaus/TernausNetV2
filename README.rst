===================================================================
TernausNetV2: Fully Convolutional Network for Instance Segmentation
===================================================================


|teaser|

We present network definition and weights for our second place solution in `CVPR 2018 DeepGlobe Building Extraction Challenge`_.

.. contents::

Team members
------------
`Vladimir Iglovikov`_, `Selim Seferbekov`_, `Alexandr Buslaev`_, `Alexey Shvets`_

Citation
----------

If you find this work useful for your publications, please consider citing::

      @InProceedings{Iglovikov_2018_CVPR_Workshops,
           author = {Iglovikov, Vladimir and Seferbekov, Selim and Buslaev, Alexander and Shvets, Alexey},
            title = {TernausNetV2: Fully Convolutional Network for Instance Segmentation},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
            month = {June},
             year = {2018}
            }


Overview
--------
Automatic building detection in urban areas is an important task that creates new opportunities for large scale urban planning and population monitoring. In a `CVPR 2018 Deepglobe Building Extraction Challenge`_ participants were asked to create algorithms that would be able to perform binary instance segmentation of the building footprints from satellite imagery. Our team finished second and in this work we share the description of our approach, network `weights`_ and code that is sufficient for inference. 

Data
----
The training data for the building detection subchallenge originate from the `SpaceNet dataset`_. The dataset uses satellite imagery with 30 cm resolution collected
from DigitalGlobe’s WorldView-3 satellite. Each image has 650x650 pixels size and covers 195x195 m2
of the earth surface. Moreover, each region consists of high-resolution RGB, panchromatic, and 8-channel low-resolution
multi-spectral images. The satellite data comes from 4 different cities: Vegas, Paris, Shanghai, and Khartoum with different coverage, of (3831, 1148, 4582, 1012)
images in the train and (1282, 381, 1528, 336) images in the test sets correspondingly.

Method
------
The originial `TernausNet`_ was extened in a few ways:
 1. The encoder was replaced with `WideResnet 38 that has In-Place Activated BatchNorm`_.
 2. The input to the network was extended to work with 11 input channels. Three for RGB and eight for multispectral data.

      In order to make our network to perform instance segmentation, we utilized the idea that was proposed
      and successfully executed by `Alexandr Buslaev`_, `Selim Seferbekov`_ and Victor Durnov in their
      winning solutions of the `Urban 3d`_ and `Data Science Bowl 2018`_ challenges.

 3. Output of the network was modified to predict both the binary mask in which we predict building / non building classes on the pixel level and binary mask in which we predict areas of an image where different objects touch or very close to each other. These predicted masks are combined and used as an input to the watershed transform.

|network|

Results
-------
Result on the public and private leaderboard with respect to the metric that was used by the organizers of the `CVPR 2018 DeepGlobe Building Extraction Challenge`_.

.. table:: Results per city

    ============= =================== ===================
    City:         Public Leaderboard  Private Leaderboard
    ============= =================== ===================
    Vegas         0.891               0.892
    Paris         0.781               0.756
    Shanghai      0.680               0.687
    Khartoum      0.603               0.608
    ------------- ------------------- -------------------
    Average       0.739               0.736
    ============= =================== ===================


Dependencies
------------

* Python 3.6
* PyTorch 0.4
* numpy 1.14.0
* opencv-python 3.3.0.10


Demo Example
~~~~~~~~~~~~~~~~~~~~~~
Network `weights`_


You can easily start using our network and weights, following the demonstration example
  `demo.ipynb`_

..  _`demo.ipynb`: https://github.com/ternaus/TernausNetV2/blob/master/Demo.ipynb
.. _`Selim Seferbekov`: https://www.linkedin.com/in/selim-seferbekov-474a4497/
.. _`Alexey Shvets`: https://www.linkedin.com/in/shvetsiya/
.. _`Vladimir Iglovikov`: https://www.linkedin.com/in/iglovikov/
.. _`Alexandr Buslaev`: https://www.linkedin.com/in/al-buslaev/
.. _`CVPR 2018 DeepGlobe Building Extraction Challenge`: https://competitions.codalab.org/competitions/18544
.. _`TernausNet`: https://arxiv.org/abs/1801.05746
.. _`U-Net`: https://arxiv.org/abs/1505.04597
.. _`Urban 3d`: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10645/0000/Urban-3D-challenge--building-footprint-detection-using-orthorectified-imagery/10.1117/12.2304682.short?SSO=1
.. _`Data Science Bowl 2018`: https://www.kaggle.com/c/data-science-bowl-2018/
.. _`WideResnet 38 that has In-Place Activated BatchNorm`: https://arxiv.org/abs/1712.02616
.. _`SpaceNet dataset`: https://spacenetchallenge.github.io/
.. _`weights`: https://drive.google.com/open?id=1k95VGNZG74Vvu-X-MSpbaHjMDvNEepIi


.. |network| image:: https://habrastorage.org/webt/jx/ni/ki/jxnikimnmkmkrrqlvcl6memouso.png
.. |teaser| image:: https://habrastorage.org/webt/ko/b2/tw/kob2twhjzjfnauix7ljted07ga8.png
