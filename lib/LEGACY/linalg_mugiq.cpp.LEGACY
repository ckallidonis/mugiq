
#include <linalg_mugiq.h>
#include <blas_quda.h>

using namespace quda;


void computeEvalsMuGiq(const DiracMatrix &mat, std::vector<ColorSpinorField *> &v,
		       std::vector<Complex> &lambda, QudaEigParam *eigParams){

  if (getVerbosity() >= QUDA_SUMMARIZE)
    printfQuda("Eigenvalues from %s\n", __func__);

  int nConv = eigParams->nConv;
  
  ColorSpinorParam csParam(*v[0]);
  ColorSpinorField *w;
  w = ColorSpinorField::Create(csParam);
  
  for(int i=0; i<nConv; i++){
    mat(*w,*v[i]); //- w = M*v_i
    //-C.K. TODO: if(mass-norm == QUDA_MASS_NORMALIZATION) blas::ax(1.0/(4.0*kappa*kappa), *w);
    lambda[i] = blas::cDotProduct(*v[i], *w) / sqrt(blas::norm2(*v[i])); // lambda_i = (v_i^dag M v_i) / ||v_i||
    Complex Cm1(-1.0, 0.0);
    blas::caxpby(lambda[i], *v[i], Cm1, *w); // w = lambda_i*v_i - A*v_i 
    double r = sqrt(blas::norm2(*w)); // r = ||w||
    
    if (getVerbosity() >= QUDA_SUMMARIZE)
      printfQuda("Eval[%04d] = (%+.16e,%+.16e) residual = %+.16e\n", i, lambda[i].real(), lambda[i].imag(), r);
  }

  delete w;
}
