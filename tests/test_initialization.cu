//
// initialization.cu
// Created by Marcel Rodekamp on 29. March 2020 
//
// This test program initializes the HDA::Array<int> class using the new operator
// then prints the created class address from device and host to the terminal.
// If the addresses are equal, the test has been successfull. 
// 

#include<iostream>
#include "../src/array.h"

__global__ void printOnDevice(HDA::Array<int> * t_array){
    if(threadIdx.x == 0){
        printf(" ## array_Device = %p\n", (void *) t_array);
    }
}

int main(int argc, char **argv){
    // array is a cuda managed pointer accessible on host and device
    auto array = new HDA::Array<int>(10); 
    
    // print host address
    std::cout << " ## array_Host   = " << array << "\n";

    // print device address
    printOnDevice<<<1,1>>>(array);

    // delete cuda managed pointer
    delete array;
    
    std::cout << " ## HDA Message: Initialization Works!!!" << std::endl;

    return EXIT_SUCCESS;
}
