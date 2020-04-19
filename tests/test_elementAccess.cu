//
// elementAccess.cu
// Created by Marcel Rodekamp on 18. April 2020
// This test creates an HDA::Array<int>, writes some value to the data and then reads it.
// The output of the read is then compared to the original data. 
// This will be done on host and on device.
//

#include<iostream>
#include "../src/array.h"

// just for better output: return string of the bool being true or false.
#define BOOL_CONVERSION(t_x) ((t_x) ? "True" : "False")

__global__ void testSize(HDA::Array<int> & t_array, bool * t_flag){
    if(t_array.size() != 10){
        *t_flag = false;
    }
}

__global__ void testOp(HDA::Array<int> & t_array, bool * t_flag){
    auto index = threadIdx.x;

    if(t_array[index] != index){
        *t_flag = false;
    }
}

__global__ void testAt(HDA::Array<int> & t_array, bool * t_flag){
    auto index = threadIdx.x;

    if(t_array.at(index) != index){
        *t_flag = false;
    }
}

__global__ void testFront(HDA::Array<int> & t_array, bool * t_flag){
    if(t_array.front() != 0){
        *t_flag = false;
    }
}

__global__ void testBack(HDA::Array<int> & t_array, bool * t_flag){
    if(t_array.back() != 9){
        *t_flag = false;
    }
}

__global__ void testData1(HDA::Array<int> & t_array, bool * t_flag){
    auto index = threadIdx.x;

    if(*(t_array.data() + index) != index){
        *t_flag = false;
    }
}

__global__ void testData2(int * t_data, bool * t_flag){
    auto index = threadIdx.x;

    if(*(t_data + index) != index){
        *t_flag = false;
    }
}

int main(void){

    std::cout << " ## HDA Message: Creating Array \n";
   
    // create array (HDA::Arrat<int> * array)
    auto array = new HDA::Array<int>(10);

    // flag used to determine correctness on device
    bool * deviceFlag;
    cudaMallocManaged(&deviceFlag, 1);
    deviceFlag[0] = true;

    std::cout << " ## HDA Message: Testing Size of Array \n";

    // test array size on host
    if((*array).size() != 10){
        std::cout << " ## HDA Error: Host Array size is not 10: line " << __LINE__ << std::endl;
        return EXIT_FAILURE; 
    }

    // test array size on device 
    testSize<<<1,1>>> (*array, deviceFlag);
    cudaDeviceSynchronize();
    if (!*deviceFlag){
        std::cout << " ## HDA Error: Device Array size is not 10: line " << __LINE__ << std::endl;
        return EXIT_FAILURE; 
    }

    std::cout << " ## HDA Message: Writing Array\n";

    // write data to array
    for(size_t index=0; index < (*array).size(); ++index){
        (*array)[index] = index;
    }

        
    std::cout << " ## HDA Message: Testing Host Access\n";
    // loop through array and test every accessor
    for(size_t index=0; index < (*array).size(); ++index){
        // test []
        if((*array)[index] != index){
            std::cout << " ## HDA Error: Host Array Access [] failed at index: " << index 
                      << " line " << __LINE__ << std::endl;
            return EXIT_FAILURE;
        }
        // test at()
        if((*array).at(index) != index){
            std::cout << " ## HDA Error: Host Array Access at() failed at index: " << index 
                      << " line " << __LINE__ << std::endl;
            return EXIT_FAILURE;
        }        
    }
    // test front
    if((*array).front() != 0){
        std::cout << " ## HDA Error: Host Array Access front() failed line " 
                  << __LINE__ << std::endl;
        return EXIT_FAILURE;
    }  
    // test back
    if((*array).back() != 9){
        std::cout << " ## HDA Error: Host Array Access back() failed line " 
                  << __LINE__ << std::endl;
        return EXIT_FAILURE;
    }  
    // test data()
    for(size_t index=0; index < (*array).size(); ++index){
        if( *((*array).data() + index) != index){
            std::cout << " ## HDA Error: Host Array Access data() failed at index: " << index 
                      << " line " << __LINE__ << std::endl;
            return EXIT_FAILURE;

        }
    }

    std::cout << " ## HDA Message: Synchronizing Host To Device\n";
    // synch host data to device
    (*array).hostToDeviceSync();

    std::cout << " ## HDA Message: Testing Device Access\n";
    // test [] 
    testOp<<<1,10>>> (*array, deviceFlag);
    cudaDeviceSynchronize();
    if (!*deviceFlag){
        std::cout << " ## HDA Error: Device Array Access [] failed line " << __LINE__ << std::endl;
        return EXIT_FAILURE; 
    }
    // test at()
    testAt<<<1,10>>> (*array, deviceFlag);
    cudaDeviceSynchronize();
    if (!*deviceFlag){
        std::cout << " ## HDA Error: Device Array Access at() failedsize line " << __LINE__ << std::endl;
        return EXIT_FAILURE; 
    }
    // test front() 
    testFront<<<1,1>>> (*array, deviceFlag);
    cudaDeviceSynchronize();
    if (!*deviceFlag){
        std::cout << " ## HDA Error: Device Array Access front() failed line " << __LINE__ << std::endl;
        return EXIT_FAILURE; 
    }
    // test back() 
    testBack<<<1,1>>> (*array, deviceFlag);
    cudaDeviceSynchronize();
    if (!*deviceFlag){
        std::cout << " ## HDA Error: Device Array Access back() failed line " << __LINE__ << std::endl;
        return EXIT_FAILURE; 
    }
    // test data() called on device  
    testData1<<<1,10>>> (*array, deviceFlag);
    cudaDeviceSynchronize();
    if (!*deviceFlag){
        std::cout << " ## HDA Error: Device Array Access data() called on device failed line " << __LINE__ << std::endl;
        return EXIT_FAILURE; 
    }
    // test data() called on host
    testData2<<<1,10>>> ((*array).data(), deviceFlag);
    cudaDeviceSynchronize();
    if (!*deviceFlag){
        std::cout << " ## HDA Error: Device Array Access data() called on host failed line " << __LINE__ << std::endl;
        return EXIT_FAILURE; 
    }

    cudaFree(deviceFlag);

    return EXIT_SUCCESS;

}

