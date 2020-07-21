#ifndef _LOOP_MUGIQ_H
#define _LOOP_MUGIQ_H

//#include <quda.h>
//#include <multigrid.h>           //- The QUDA MG header file
//#include <color_spinor_field.h>  //- From QUDA
#include <mugiq.h>
//#include <enum_mugiq.h>


using namespace quda;


template <typename Float>
class Loop_Mugiq {

private:
  MugiqLoopParam *loopParams;

  
public:

  Loop_Mugiq(MugiqLoopParam *loopParams);
  ~Loop_Mugiq();

}; // class Loop_Mugiq

#endif // _LOOP_MUGIQ_H
