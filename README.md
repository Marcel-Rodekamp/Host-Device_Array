# Host-Device Array

This class provides a basic STL Array like interface which is accessible on host and Device. 
It is still in development, sofar only some basic access members are avaialble.

# Features

## Initializing the class 
The class has no default constructor, but can be initialized with a parameterised constructor
```c++
#include "src/array.h"

int main(void){
    // this will create an HDA::Array containing doubles and having 100 elements
    HDA::Array<double> array(100);
 
    return EXIT_SUCCESS;
}
```
Creating the array in this way does not provide a direct accessibility on the device (it is created on the host). However, it still contains a device pointer which can be accessed for example with the `HDA::Array<double> data()` member. In order to have a propper host device access the `new` operator can be used:
``` c++
#include "src/array.h"

int main(void){
    // this will create a pointer to one HDA::Array containing doubles and having 100 elements 
    // this pointer uses the CUDA Managed space and can thus be accessed on host and device
    HDA::Array<double> * array = new HDA::Array<double> (100);
 
    // At the end the array needs to be deleted
    delete array;

    return EXIT_SUCCESS;
}

```

## Accessing Data

## Synchronize

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
