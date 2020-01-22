#ifndef _ENUM_MUGIQ_H
#define _ENUM_MUGIQ_H

#include <limits.h>

#define MUGIQ_INVALID_ENUM INT_MIN

#ifdef __cplusplus
extern "C" {
#endif

  typedef enum MuGiqTask_s
    {
     MUGIQ_COMPUTE_EVECS_QUDA,
     MUGIQ_COMPUTE_EVECS_MUGIQ,
     MUGIQ_COMPUTE_LOOP_ULOCAL,
     MUGIQ_TASK_INVALID = MUGIQ_INVALID_ENUM
    } MuGiqTask;
  
  typedef enum MuGiqEigOperator_s
    {
     MUGIQ_EIG_OPERATOR_MG,
     MUGIQ_EIG_OPERATOR_NO_MG,
     MUGIQ_EIG_OPERATOR_INVALID = MUGIQ_INVALID_ENUM
    } MuGiqEigOperator;
  

#ifdef __cplusplus
}
#endif

#endif // _ENUM_MUGIQ_H  
