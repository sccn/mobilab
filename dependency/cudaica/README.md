# How to build and install CUDAICA

The following tutorial shows how to build and install CUDAICA binary and its helper programs. CUDAICA is an implementation of the Infomax ICA algorithm that is critical for speeding up orders of magnitude the decomposition of data into independent components.

Although several tutorials already exist (see [cudaica on github](https://github.com/fraimondo/cudaica) and [cudaica on AAIL/UBA](https://liaa.dc.uba.ar/node/13)) we have found that the lack of details on those often leads to a cumbersome install experience. 

Each command below should be typed in a Debian-based GNU/Linux Bash terminal. We can install CUDAICA following these steps:

1- Clone the repo and *cd* into it:
```bash
git clone https://github.com/fraimondo/cudaica.git
cd cudaica
```

2- Install dependencies:
```bash
sudo apt-get update
sudo apt-get install build-essential gfortran libblas-dev liblapack-dev libatlas3gf-base autoconf
```
You may need to install other dependencies depending on the state of your system. Keep an eye on the output of the *configure* command in step 5 and install whatever you could be missing.

3- Locate the directory containing the cuda libraries, which for many systems will be */usr/local/cuda*, and add it to the */etc/ld.so.conf.d/cuda.conf* file as follows:
```bash
echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf
ldconfig	# configure dynamic linker run-time bindings
```

4- Run the automake script that comes with cudaica:
```bash
./reconf.sh
```

5- Run the *configure* script (**not with the default options**). This is probably the most critical step since we will be tempted to use the defaults (nacked *./configure* command), we strongly recommend to run the script specifying the compute capability of your GPU card (see [here](https://developer.nvidia.com/cuda-gpus)). In this example we used a Quadro K6000 with compute capability of 3.5, then we run the following command:
```bash
./configure -with-cuda-arch=35
```
note that we entered 35 instead of 3.5.

6- Build and install:
```bash
make
sudo make install
```

7- Make sure that CUDAICA was installed showing its help text:
```bash
cudaica --help
```
The ultimate proof that the installation was successful is running CUDAICA on some data within the MATLAB/EEGLAB enviroment (for test purposes you can juts use random data), for which several wrappers exist (see [cudaica plugin](https://liaa.dc.uba.ar/node/20) [MoBILAB wrapper](https://github.com/aojeda/mobilab/tree/master/dependency/cudaica)  )

That's it folks, enjoy!