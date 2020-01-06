#ifndef _MG_MUGIQ_H
#define _MG_MUGIQ_H

#include <multigrid.h> //- The QUDA MG header file


class MG_Mugiq {

private:
  quda::multigrid_solver *mg_solver;
  QudaMultigridParam *param;
  
public:
  MG_Mugiq(QudaMultigridParam *param_, quda::TimeProfile &profile_);
  
  virtual ~MG_Mugiq();

}; //- Class MG_Mugiq


#endif // _MG_MUGIQ_H
