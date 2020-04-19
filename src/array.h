#ifndef ARRAY_H
#define ARRAY_H

#include "util/gpu.h"

namespace HDA
{
template<typename T> class Array;

template<typename T>
class Array{
    private:
        // data on the host 
        T * m_ptr_host = nullptr;
        // data on the device 
        T * m_ptr_device;
        // length of data, must be the same for host and device 
        const size_t m_size;
        // size of elements (i.e. m_size*sizeof(T))
        const size_t m_count;

        // ==============================================================================
        // deleted functions 
        // ==============================================================================       
        Array() = delete;
        Array(const Array<T> &) = delete;
        Array(const Array<T> &&) = delete;
        Array<T> & operator=(const Array<T> &) = delete;

    public:
        // ==============================================================================
        // Constructors
        // ==============================================================================

        Array(const size_t t_size) : 
            m_size(t_size), 
            m_count(t_size * sizeof(T))
        {
            // initialize host pointer 
            m_ptr_host = new T [t_size]; 

            // initialize device pointer
            deviceAssert(cudaMalloc(&m_ptr_device, m_count));
            cudaDeviceSynchronize();
        }

        void * operator new(size_t t_len) {
            // create new managed pointer 
            void * ptr;
            
            // allocate memory for the classes in the managed address space 
            cudaMallocManaged(&ptr, t_len);

            // sync cuda devices 
            cudaDeviceSynchronize();

            return ptr;
        }

        // ==============================================================================
        // Destructor 
        // ==============================================================================

        ~Array() {
            // destruct host pointer
            if(m_ptr_host!=nullptr){
                delete[] m_ptr_host;
            } else {
                std::cerr << " ## Can't destruct pointer for array. Abort..." 
                          << std::endl;
                exit (EXIT_FAILURE);
            }

            // destruct device pointer
            cudaDeviceSynchronize();
            deviceAssert(cudaFree(m_ptr_device));
        }

        void operator delete(void * t_ptr){
            // ensure that no device is working on the class
            cudaDeviceSynchronize();
            
            // destruct the pointer created by new
            if(t_ptr != nullptr){
                cudaFree(t_ptr);
            }
        }

        // ==============================================================================
        // Element access
        // ==============================================================================
 
        __host__ __device__ T & operator[](const size_t pos) {
            #ifndef __CUDA_ARCH__
                // return host data, __CUDA_ARCH__ is not defined on host
                return m_ptr_host[pos];
            #else 
                // return device data, __CUDA_ARCH__ is defined 
                return m_ptr_device[pos];
            #endif
        }

        __host__ __device__ const T & operator[](const size_t pos) const {
           #ifndef __CUDA_ARCH__
                // return host data, __CUDA_ARCH__ is not defined on host
                return m_ptr_host[pos];
            #else 
                // return device data, __CUDA_ARCH__ is defined 
                return m_ptr_device[pos];
            #endif
        }

        __host__ __device__ T & at(const size_t pos) {
           #ifndef __CUDA_ARCH__
                // return host data, __CUDA_ARCH__ is not defined on host
                return m_ptr_host[pos];
            #else 
                // return device data, __CUDA_ARCH__ is defined 
                return m_ptr_device[pos];
            #endif
        }
        
        __host__ __device__ const T & at(const size_t pos) const {
           #ifndef __CUDA_ARCH__
                // return host data, __CUDA_ARCH__ is not defined on host
                return m_ptr_host[pos];
            #else 
                // return device data, __CUDA_ARCH__ is defined 
                return m_ptr_device[pos];
            #endif
        }

        __host__ __device__ T & front() {
           #ifndef __CUDA_ARCH__
                // return host data, __CUDA_ARCH__ is not defined on host
                return m_ptr_host[0];
            #else 
                // return device data, __CUDA_ARCH__ is defined 
                return m_ptr_device[0];
            #endif
        }
        
        __host__ __device__ const T & front() const {
           #ifndef __CUDA_ARCH__
                // return host data, __CUDA_ARCH__ is not defined on host
                return m_ptr_host[0];
            #else 
                // return device data, __CUDA_ARCH__ is defined 
                return m_ptr_device[0];
            #endif
        }

        __host__ __device__ T & back() {
           #ifndef __CUDA_ARCH__
                // return host data, __CUDA_ARCH__ is not defined on host
                return m_ptr_host[m_size-1];
            #else 
                // return device data, __CUDA_ARCH__ is defined 
                return m_ptr_device[m_size-1];
            #endif
        }
        
        __host__ __device__ const T & back() const {
           #ifndef __CUDA_ARCH__
                // return host data, __CUDA_ARCH__ is not defined on host
                return m_ptr_host[m_size-1];
            #else 
                // return device data, __CUDA_ARCH__ is defined 
                return m_ptr_device[m_size-1];
            #endif
        }
        
        __host__ __device__ T * data() {
           #ifndef __CUDA_ARCH__
                // return host data, __CUDA_ARCH__ is not defined on host
                return m_ptr_host;
            #else 
                // return device data, __CUDA_ARCH__ is defined 
                return m_ptr_device;
            #endif
        }

        __host__ __device__ const T * data() const {
            #ifndef __CUDA_ARCH__
                // return host data, __CUDA_ARCH__ is not defined on host
                return m_ptr_host;
            #else 
                // return device data, __CUDA_ARCH__ is defined 
                return m_ptr_device;
            #endif
        }

         __host__ __device__ size_t size() const {
            return m_size;
        }

        __host__ __device__ size_t count() const {
            return m_count;
        }

        // ==============================================================================
        // Host and Device Synchronization
        // ==============================================================================
        void hostToDeviceSync() {
            cudaDeviceSynchronize();
            deviceAssert(cudaMemcpy(m_ptr_device, m_ptr_host, m_count, cudaMemcpyHostToDevice));
        }
 
        void deviceToHostSync() {
            cudaDeviceSynchronize();
            deviceAssert(cudaMemcpy(m_ptr_host, m_ptr_device, m_count, cudaMemcpyDeviceToHost));
        }

}; // class Array

} // namespace HDA
#endif // ARRAY_H
