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

  typedef enum LoopFTSign_s
    {
     LOOP_FT_SIGN_MINUS = -1,
     LOOP_FT_SIGN_PLUS  =  1,
     LOOP_FT_SIGN_INVALID = MUGIQ_INVALID_ENUM
    } LoopFTSign;

  typedef enum LoopCalcType_s
    {
     LOOP_CALC_TYPE_BLAS,          //- Calculate loop using BLAS
     LOOP_CALC_TYPE_OPT_KERNEL,    //- Calculate loop using tunable/optimized CUDA kernel
     LOOP_CALC_TYPE_BASIC_KERNEL,  //- Calculate loop using a basic CUDA kernel
     LOOP_CALC_TYPE_INVALID = MUGIQ_INVALID_ENUM
    } LoopCalcType;

  typedef enum MuGiqBool_s
    { MUGIQ_BOOL_FALSE   = 0,
      MUGIQ_BOOL_TRUE    = 1,     
      MUGIQ_BOOL_INVALID = MUGIQ_INVALID_ENUM
    } MuGiqBool;
  

#ifdef __cplusplus
}
#endif

#endif // _ENUM_MUGIQ_H  
