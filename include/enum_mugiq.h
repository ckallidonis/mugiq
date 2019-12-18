#ifndef _ENUM_MUGIQ_H
#define _ENUM_MUGIQ_H

#include <limits.h>

#define MUGIQ_INVALID_ENUM INT_MIN

#ifdef __cplusplus
extern "C" {
#endif

  typedef enum MuGiqEigTask_s
    {
     MUGIQ_COMPUTE_EVECS_QUDA,
     MUGIQ_COMPUTE_EVECS_MUGIQ,
     MUGIQ_COMPUTE_EVECS_INVALID = MUGIQ_INVALID_ENUM
    } MuGiqEigTask;
  

#ifdef __cplusplus
}
#endif

#endif // _ENUM_MUGIQ_H  
