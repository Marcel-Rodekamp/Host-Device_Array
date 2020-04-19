# Host-Device Array

This class provides a basic STL Array like interface which is accessible on host and Device. 
It is still in development, sofar only some basic access members are avaialble.

# Features

## Initializing the class 
### Parametized Constructor
The class has no default constructor, but can be initialized with a parameterised constructor for any type T (`template<class T>`) and size S (`size_t`)
```c++
#include "src/array.h"

int main(void){
    // this will create an HDA::Array
    HDA::Array<T> array(S);
 
    return EXIT_SUCCESS;
}
```
Creating the array in this way does not provide a direct accessibility on the device (it is created on the host). However, it still contains a device pointer which can be accessed for example with the `HDA::Array<double> data()` member. 

### `new` Operator
In order to have a propper host device access the `new` operator can be used:
``` c++
#include "src/array.h"

int main(void){
    // this will create a pointer to one HDA::Array  
    // this pointer uses the CUDA Managed space and can thus be accessed on host and device
    HDA::Array<T> * array = new HDA::Array<T> (S);
 
    // At the end the array needs to be deleted
    delete array;

    return EXIT_SUCCESS;
}

```

## Accessing Data
There are a bunch of different accessing methods implemented.
Once initialized with the `new` operator the class can be passed by reference to functions and kernels:
```c++
// definition example of CUDA kernel
__global__ void kernel_name(HDA::Array<T> & );
// definition example of host function 
void function_name(HDA::Array<T> & );
```
Note that the new operator returns a pointer thus calling these requires dereferencing:
```c++
// Create array of type with the size elements 
auto array = new HDA::Array<T> (S);

// call kernel
kernel_name<<<Blogs,Threads>>> (*array);

// call host function
function_name(*array);

```

### `operator []`

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
