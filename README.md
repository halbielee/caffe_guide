# Caffe Installation guide & tips
 This is the installation guide for [*Caffe*](https://caffe.berkeleyvision.org/) framework under
 **CUDA 10** and ubuntu16.04. 
 
This can be compiled in the other CUDA version as well, but with slightly different setting.

Please use CUDA <= 10.0. (Higher version has not been checked, and some error happens.)

I install *Caffe* for using [DeepLab-V2](https://github.com/jay-mahadeokar/deeplab-public-ver2/blob/master/docs/installation.md).
If you go to the link, there is an instruction guide for DeepLab-V2 in *Caffe* framework.

If you have any question, please leave issue!

## Environment

```
# Environment
NVIDIA RTX 2080ti * 2 (GTX 1080ti, TitanX, or other GPUs)
Driver Version: 418.56
CUDA Version: 10.0 (< 10.1)
```

## Installation

Here, we install *caffe* under *docker* environment. But you can install with out *docker*.  

#### 1. Creating nvidia-docker

For docker environment, pull an image and make a container of it. In here, pulls an official image of nvidia. 
Please see [here](https://hub.docker.com/r/nvidia/cuda) for nvidia docker image details.

> nvidia-docker run --runtime=nvidia -it  -v [USER_LOCAL_PATH]:[INTERNAL_PATH] nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

command examples.
> nvidia-docker run --runtime=nvidia -it --shm-size=32g -e LC_ALL=C.UTF-8 -v /home/project:/workspace nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04



#### 2. Setting the docker environment for Caffe or DeepLab-v2.

Install the essential libraries or modules for *Caffe* build. 
Some libraries are for [DSRG](https://github.com/speedinghzl/DSRG).
```
apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        libmatio-dev \
        libeigen3-dev \
        python-scipy \
        libhdf5-dev \
```

#### 3. Downloading Deeplab-v2 and setting (Caffe environment example)

Downloads and sets for DeepLab v2. See the details in [here](https://bitbucket.org/aquariusjay/deeplab-public-ver2/src/master/).
You can use any other *Caffe framework* including *matcaffe*.
> git clone -b master --depth 1 https://bitbucket.org/aquariusjay/deeplab-public-ver2.git
>
> cd deeplab-public-ver2
>
> pip install -r python/requirements.txt

#### 4. Make

See [the official installation guide](https://caffe.berkeleyvision.org/installation.html) for caffe.
```bash
# go to the directory at which 'Makefile.config.example' exists

# copy the configuration file
cp Makefile.config.example Makefile.config

# build caffe framework
make all -j8

# check whether this succeed in building caffe framework 
make test

# (optional) build pycaffe
make pycaffe -j8

# (optional) build matcaffe
make matcaffe -j8
```


#### 5. Add to Environment Variable

For using eeplab-v2 caffe, Caffe which is compiled should be included in Environment Variable
The command path is
> deeplab-public-ver2/build/tools # caffe
> deeplab-public-v2/python # Python

For temporary, write down followings in command line. **Be sure the path is correct and the caffe executable file is in that directory.**
> export PATH = [DEEPLAB_PATH]/deeplab-public-ver2/build/tools
> export PYTHONPATH = [DEEPLAB_PATH]/deeplab-public-ver2/python

For permanently, go to the /etc/bash.bashrc and add the following lines. **Be sure the path is correct and the caffe executable file is in that directory. Following two lines can make your ENV PATH wrong.**

> vi /etc/bash.bashrc

> PATH=$PATH:[DEEPLAB_PATH]/deeplab-public-ver2/build/tools
>
> PYTHONPATH=$PYTHONPATH:[DEEPLAB_PATH]/deeplab-public-ver2/python


#### 5. Error case in Make

##### 1) Eigen.h

###### Error Message:
```
/usr/include/nanogui/common.h:28: error: Eigen/Core: No such file or directory
 #include <Eigen/Core>
or
fatal error: Eigen/Core: No such file or directory compilation terminated.
 ```

###### Solution:
```
apt-get update
apt-get install libeigen3-dev
# making symbolic link for Eigen library
ln -s /usr/include/eigen3/Eigen /usr/local/include/Eigen
```

There are other solutions like changing the header file, but there can be several header files to modify.

##### 2) nvcc error

###### Error Message:
```
nvcc fatal : Unsupported gpu architecture 'compute_20'
```

###### Solution:

This is the problem in CUDA version. Go to the Makefile.config and delete or comment
arch=compute_20 part.

See [here](https://github.com/kaldi-asr/kaldi/issues/1918)

```
# CUDA architecture setting: going with all of them.
# For CUDA < 6.0, comment the *_50 lines for compatibility.
CUDA_ARCH :=
# Comment the following line.
#   -gencode arch=compute_20,code=sm_20 \
		-gencode arch=compute_20,code=sm_21 \
		-gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_50,code=compute_50
```

##### 3) common.cuh

###### Error Message:
```
./include/caffe/common.cuh(9): error: function "atomicAdd(double *, double)" has already been defined
```

###### Solution:

Go to the deeplab-public-ver2/include/common.cuh and modify as follows.

See [here](https://stackoverflow.com/questions/39274472/error-function-atomicadddouble-double-has-already-been-defined)

```
#ifndef CAFFE_COMMON_CUH_
#define CAFFE_COMMON_CUH_

#include <cuda.h>

  #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

  #else
  static __inline__ __device__ double atomicAdd(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val==0.0)
      return __longlong_as_double(old);
    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
  }


  #endif
#endif
```

##### 4) opencv Version

###### Error Message:
```
AR -o .build_release/lib/libcaffe.a
LD -o .build_release/lib/libcaffe.so.1.0.0-rc3
/usr/bin/ld: cannot find -lcudnn
collect2: error: ld returned 1 exit status
Makefile:554: recipe for target '.build_release/lib/libcaffe.so.1.0.0-rc3' failed

or

.build_release/lib/libcaffe.so: undefined reference to `cv::imread(cv::String const&, int)'
.build_release/lib/libcaffe.so: undefined reference to `cv::imencode(cv::String const&, cv::_InputArray const&, std::vector<unsigned char, std::allocator<unsigned char> >&, std::vector<int, std::allocator<int> > const&)'
.build_release/lib/libcaffe.so: undefined reference to `cv::imdecode(cv::_InputArray const&, int)'
collect2: error: ld returned 1 exit status
```

###### Solution:

Uncomment some line in Makefile.config. See [here](https://github.com/BVLC/caffe/issues/4621) for details

```
# Uncomment if you're using OpenCV 3
OPENCV_VERSION := 3
```

###### Error Message:
```
/usr/bin/ld: cannot find -lopencv_imgcodecs
```
###### Solution

This case, you are using Opencv under 3. Comment some line in Makefile.config.
```
# Uncomment if you're using OpenCV 3
# OPENCV_VERSION := 3
```


##### 5) UNIX time

##### Error Message

```
*** Aborted at 1557975957 (unix time) try "date -d @1557975957" if you are using GNU date ***
```

##### Solution
```
date -d @1557975957
```

##### 6) hdf5 error

##### Error Message

```
./include/caffe/util/io.hpp:8:18: fatal error: hdf5.h: no such file or directory
 #include "hdf5.h"
 ```

##### Solution
Modify following lines in Makefile.config
```
# Whatever else you find you need goes here.
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial
```

##### 7) numpy error

##### Error Message

```
python/caffe/_caffe.cpp:10:31: fatal error: numpy/arrayobject.h: No such file or
 ```

##### Solution
Modify following lines in Makefile.config
```
# NOTE: this is required only if you will compile the python interface.
# We need to be able to find Python.h and numpy/arrayobject.h.
PYTHON_INCLUDE := /usr/include/python2.7 \
		/usr/lib/python2.7/dist-packages/numpy/core/include \
		/usr/local/lib/python2.7/dist-packages/numpy/core/include
```

##### 8) (Matlab & Matcaffe) libGL problem

When *matlab* exists as soon as it turns on  if libGL problem
```bash
libGL error: unable to load driver: r600_dri.so
libGL error: driver pointer missing
libGL error: failed to load driver: r600
libGL error: unable to load driver: swrast_dri.so
libGL error: failed to load driver: swrast
```

rename the file at "$MATLAB_ROOT$/sys/os/glnxa64/libstdc++.so.6" to any other name.

Please see the detail in [link](https://kr.mathworks.com/matlabcentral/answers/296999-libgl-error-unable-to-load-driver-in-ubuntu-16-04-while-running-matlab-r2013b).
```bash
mv $MATLAB_ROOT$/sys/os/glnxa64/libstdc++.so.6 $MATLAB_ROOT$/sys/os/glnxa64/libstdc++.so.6.old
```

##### 9) (Matlab & Matcaffe) gcc / g++ version problem

If you have gcc problem, you need to check the version of gcc

```
caffe build -> gcc >= 5.0 (if less, error happens)

matlab 2015a -> mex uses gcc 4.7 (or under)
matlab 2017b -> mex uses gcc 4.9 (or under)
```

Here, I use gcc=5.4 and g++=5.4

Please see the detail in [link](https://blog.koriel.kr/gcc-g-dareun-beojeon-cugahago-paekiji-gwanrihagi/).

##### 10) cuda dependencies

If you have CUDA dependency problem, you need to delete all the packages which version is inappropriate.



### Installing DSRG

See [here](https://github.com/speedinghzl/DSRG). If DeepLab-v2 is installed successfully, there would be no complicated problem in DSRG.
