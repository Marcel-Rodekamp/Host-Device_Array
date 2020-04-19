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

The following Access methods are implemented and are callable on device and host.
* `__host__ __device__ T & HDA::Array<T>::operator[](size_t)`
    * Write access to the data at given index.
* `__host__ __device__ const T & HDA::Array<T>::operator[](size_t) const `
    * Read access to the data at given index.
* `__host__ __device__ T & HDA::Array<T>::at(size_t)`
    * Write access to the data at given index.
* `__host__ __device__ const T & HDA::Array<T>::at(size_t) const`
    * Read access to the data at given index.
* `__host__ __device__ T & HDA::Array<T>::front()`
    * Write access to the data at index = 0.
* `__host__ __device__ const T & HDA::Array<T>::front() const`
    * Read access to the data at index = 0.
* `__host__ __device__ T & HDA::Array<T>::back()`
    * Write access to the data at max index.
* `__host__ __device__ const T & HDA::Array<T>::back() const`
    * Read access to the data at max index.
* `__host__ __device__ T * HDA::Array<T>::data()`
    * Returns the pointer to the data either on host or device, where ever it is called. Write access is granted.
* `__host__ __device__ const T * HDA::Array<T>::back() const`
    * Returns the pointer to the data either on host or device, where ever it is called. Read access only.
* `__host__ __device__ const size_t HDA::Array<T>::size() const`
    * Returns the number of elements of the pointer (S).
* `__host__ __device__ const size_t HDA::Array<T>::count() const`
    * Returns the data size of the array (`S * sizeof(T)`)

## Synchronize

To take care of synchronization of host and device data two simple functions are implemented
* `__host__ void HDA::Array<T>::hostToDeviceSync()`
    * Copies the host data to the device
* `__host__ void HDA::Array<T>::deviceToHostSync()`
    * Copies the device data to the host
  

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
