#include <util_mugiq.h>
#include <sys/types.h>
#include <sys/sysinfo.h>
#include <cuda_runtime_api.h>

void printCPUMemInfo(){
  struct sysinfo memInfo;
  sysinfo (&memInfo);
  long long totalPhysMem = memInfo.totalram;
  long long freePhysMem  = memInfo.freeram;
  long long usedPhysMem  = memInfo.totalram - memInfo.freeram;
  totalPhysMem *= memInfo.mem_unit;
  freePhysMem  *= memInfo.mem_unit;
  usedPhysMem  *= memInfo.mem_unit;
  printf("  CPUMemInfo: Total CPU Memory: %lld MBytes.\n", totalPhysMem/(1<<20));
  printf("  CPUMemInfo: Free  CPU Memory: %lld MBytes.\n", freePhysMem /(1<<20));
  printf("  CPUMemInfo: Used  CPU Memory: %lld MBytes.\n", usedPhysMem /(1<<20));
}

void printGPUMemInfo(){
  size_t freeGPUMem, totalGPUMem;
  freeGPUMem = 5;
  totalGPUMem = freeGPUMem+2;
  if(cudaMemGetInfo(&freeGPUMem, &totalGPUMem) != cudaSuccess)
    errorQuda("  GPUMemInfo: Memory-related error occured!\n");
  else{
    printf("  GPUMemInfo: Total GPU Memory/GPU: %zd MBytes.\n", totalGPUMem/(1<<20));
    printf("  GPUMemInfo: Free  GPU Memory/GPU: %zd MBytes.\n", freeGPUMem/(1<<20));
    printf("  GPUMemInfo: Used  GPU Memory/GPU: %zd MBytes.\n", (totalGPUMem-freeGPUMem)/(1<<20));
  }
}

extern "C"
void printMemoryInfo(){
  printf("\n----------------Memory Information----------------\n");
  printCPUMemInfo();
  printf("\n");
  printGPUMemInfo();
  printf("--------------------------------------------------\n\n");
}
