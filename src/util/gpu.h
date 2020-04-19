#ifndef GPU_H
#define GPU_H
#include<iostream>
#include<string>

// Handle GPU Errors
#define deviceAssert(t_ans) { gpuAssert((t_ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t t_code, const char * t_file, int t_line, bool t_abort=true){
    if (t_code != cudaSuccess){
        std::string file(t_file);
        std::cerr << " ## HDA Device Error: " 
                  << cudaGetErrorString(t_code) << " "
                  << file << " " 
                  << t_line << std::endl;
        if (t_abort) exit(t_code);
    }
}

#endif // GPU_H
