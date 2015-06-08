# feature-feedback
video feedback from feature learning

## to run caffe models

install [caffe][1], making sure to build pycaffe. I recommend using [Anaconda][2]

clone feature-feedback into same directory that caffe root is in

look at caffe/examples/cifar-10 and follow the instructions to download and convert the cifar-10 dataset

cd feature-feedback/notebooks and start IPython

open experiments.ipynb

to use GPU, you will need to change a line in *-solver.prototxt

## to run openFramworks app

install [openFrameworks][3]

copy feature-feedback/openFrameworks/apps/feedbackh into openFrameworks apps directory

Windows: follow the instructions to set up codeblocks and use the included codeblocks workspace

Linux: build with make. Makefile.config may need to be updated with cnpy.h. Watch out for old versions of OpenGL, I know the open source AMD drivers won't work (need OpenGl 3.1/GLSL 1.4 +)

OSX: you're on your own, probably need to figure out how to convert to xcode project

To do much, you'll want to edit some constants in src/ofApp.cpp and/or hack on bin/data/shader/feedbackh.frag


[1]: http://caffe.berkeleyvision.org/ "Caffe Deep learning framework by the BVLC"

[2]: https://store.continuum.io/cshop/anaconda/ "Anaconda scientific computing Python distribution"

[3]: http://openframeworks.cc "openFrameworks c++ toolkit for creative coding"