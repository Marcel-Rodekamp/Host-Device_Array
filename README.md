# Host-Device Array

This class provides a basic STL Array like interface which is accessible on host and Device. 

# How To Compile

The compilation is not fully worked out and is only tested on Ubuntu 18.04 LTS.

* Clone the repository
```sh
git clone https://github.com/Marcel-Rodekamp/Host-Device_Array.git
```
* Go to the folder and call cmake
```sh
cd Host-Device_Array
cmake CMakeLists.txt
```
* Compile the class
```sh
make
```

# What you need
You will need the following modules previously installed
* cmake version 3.17 or higher
* CUDA 10.2 or higher 
    * With that nvcc
* GNU 7.5.0
