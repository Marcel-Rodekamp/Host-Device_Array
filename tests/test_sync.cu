#include "../src/array.h"

__global__ void checkArray1(HDA::Array<int> & t_arr, bool * flag){
    // device code
    int index = threadIdx.x;
    if(t_arr[index] != index ){
        printf("Sync to host did not work at t_arr[%d]\n", index);
        *flag = false;
    }
}

void checkArray2(HDA::Array<int> & t_arr, bool * flag){
    for(int index=0;index<t_arr.size();++index){
        if(t_arr[index] != 2*index ){
            std::cerr << "Sync to host did not work at t_arr[" << index <<"]\n";
            *flag = false;
        }        
    }
}

__global__ void deviceFill(HDA::Array<int> & t_arr){
    int index = threadIdx.x;
    t_arr[index] = 2 * index;
}

int main(){
    auto arr = new HDA::Array<int> (10);
    bool * flag;
    cudaMallocManaged(&flag, 1);
    cudaDeviceSynchronize();
    *flag = true;

    // fill array on host 
    for(int i=0; i<arr->size();++i){
        (*arr)[i] = i;
    }
    // copy to device 
    (*arr).hostToDeviceSync();

    // check array
    checkArray1<<<1,10>>>(*arr,flag);
    cudaDeviceSynchronize();


    // refill on device
    deviceFill<<<1,10>>>(*arr);
    cudaDeviceSynchronize();

    // copy to host
    (*arr).deviceToHostSync();

    // check array
    checkArray2(*arr,flag);

    cudaFree(flag);

    std::cout << " ## HDA Message: Synchronize Works!!!" << std::endl; 

    return 0;
}
