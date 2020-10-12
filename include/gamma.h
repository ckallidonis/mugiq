#ifndef _GAMMA_H
#define _GAMMA_H

#include <util_mugiq.h>
#include <mugiq.h>

using namespace quda;

/**
 * The following two functions are used to map the gamma matrix of the loop being calculated
 * to the one of the output loop that is saved. This mapping is needed because the loop required
 * contains g5*Gamma, whereas the one calculated contains Gamma.
 * The mapping is as follows:
 * Output               Input(calculated)
 * L(0)  = L(1)     <-   T(15)
 * L(1)  = L(g1)    <-  -T(14)
 * L(2)  = L(g2)    <-   T(13)
 * L(3)  = L(g1g2)  <-  -T(12)
 * L(4)  = L(g3)    <-  -T(11)
 * L(5)  = L(g1g3)  <-   T(10)
 * L(6)  = L(g2g3)  <-  -T(9)
 * L(7)  = L(g5g4)  <-   T(8)
 * L(8)  = L(g4)    <-   T(7)
 * L(9)  = L(g1g4)  <-  -T(6)
 * L(10) = L(g2g4)  <-   T(5)
 * L(11) = L(-g5g3) <-  -T(4) => L(g5g3) <- T(4)
 * L(12) = L(g3g4)  <-  -T(3)
 * L(13) = L(g5g2)  <-   T(2)
 * L(14) = L(-g5g1) <-  -T(1) => L(g5g1) <- T(1)
 * L(15) = L(g5)    <-   T(0)
 */

//- This takes care of the sign
//- T(1) and T(4) are not included because the output has a minus sign as well
std::vector<int> minusGamma(){
  std::vector<int> minusG{3, 6, 9, 11, 12, 14};
  return minusG;
}

//- This takes care of the index
std::vector<int> indexMapGamma(){
  std::vector<int> idxG{N_GAMMA_, 0};
  for(int i=0;i<N_GAMMA_;i++) idxG.at(i) = N_GAMMA_ -i -1;
  return idxG;
}


#endif
