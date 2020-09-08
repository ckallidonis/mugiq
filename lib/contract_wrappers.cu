#include <contract_util.h>

template void copyGammaCoeffStructToSymbol<float>();
template void copyGammaCoeffStructToSymbol<double>();

template <typename Float>
void copyGammaCoeffStructToSymbol(){

  GammaCoeff<Float> gamma_h;

  for(int m=0;m<N_GAMMA_;m++){
    for(int n=0;n<N_SPIN_;n++){
      gamma_h.column_index[m][n] = GammaColumnIndex[m][n];
      gamma_h.row_value[m][n] = {static_cast<Float>(GammaRowValue[m][n][0]), static_cast<Float>(GammaRowValue[m][n][1])};
    }
  }
  
  cudaMemcpyToSymbol(cGamma, &gamma_h, sizeof(GammaCoeff<Float>));
}
