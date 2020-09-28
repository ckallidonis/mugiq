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
     MUGIQ_COMPUTE_LOOP,
     MUGIQ_TASK_INVALID = MUGIQ_INVALID_ENUM
    } MuGiqTask;
  
  typedef enum MuGiqEigOperator_s
    {
     MUGIQ_EIG_OPERATOR_M,
     MUGIQ_EIG_OPERATOR_Mdag,
     MUGIQ_EIG_OPERATOR_MdagM,
     MUGIQ_EIG_OPERATOR_MMdag,
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

  typedef enum DisplaceType_s
    {
     DISPLACE_TYPE_COVARIANT = 0,      //- Perform a Covariant displacement
     DISPLACE_TYPE_INVALID = MUGIQ_INVALID_ENUM
    } DisplaceType;
 
  typedef enum MuGiqBool_s
    { MUGIQ_BOOL_FALSE   = 0,
      MUGIQ_BOOL_TRUE    = 1,     
      MUGIQ_BOOL_INVALID = MUGIQ_INVALID_ENUM
    } MuGiqBool;


  //- Some enums about the Displacements
  
  typedef enum DisplaceFlag_s {
    DispFlagNone = MUGIQ_INVALID_ENUM,
    DispFlag_X = 0,  // +x
    DispFlag_x = 1,  // -x
    DispFlag_Y = 2,  // +y
    DispFlag_y = 3,  // -y
    DispFlag_Z = 4,  // +z
    DispFlag_z = 5,  // -z
    DispFlag_T = 6,  // +t
    DispFlag_t = 7,  // -t
  } DisplaceFlag;
  
  
  typedef enum DisplaceDir_s {
    DispDirNone = MUGIQ_INVALID_ENUM,
    DispDir_x = 0,
    DispDir_y = 1,
    DispDir_z = 2,
    DispDir_t = 3
  } DisplaceDir;
  
  
  typedef enum DisplaceSign_s {
    DispSignNone  = MUGIQ_INVALID_ENUM,
    DispSignMinus =  0,
    DispSignPlus  =  1
  } DisplaceSign;  
  
  
#ifdef __cplusplus
}
#endif

#endif // _ENUM_MUGIQ_H  
