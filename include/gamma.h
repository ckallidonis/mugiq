#ifndef _GAMMA_H
#define _GAMMA_H

#include <util_mugiq.h>
#include <mugiq.h>

using namespace quda;


//- The names/tags of the Gamma matrices
inline std::string GammaName(int m){

  std::vector<std::string> gNames {
    "1"   , "g1"  , "g2"  , "g1g2",
    "g3"  , "g1g3", "g2g3", "g5g4",
    "g4"  , "g1g4", "g2g4", "g5g3",
    "g3g4", "g5g2", "g5g1", "g5"  };

  return gNames.at(m);
}

/**
 * Hard-coded gamma coefficients for the DeGrand-Rossi basis
 * Gamma-index notation is: G{x,y,z,t} = G{1,2,3,4}
 * Gamma matrices are defined as: G(n) = g1^n0 . g2^n1 . g3^n2 . g4^n3, where n = n0*2^0 + n1*2^1 + n2*2^2 + n3*2^3
 * This parametrization helps in efficient unrolling and usage when performing trace contractions on the GPU,
 * taking into account only non-zero elements when performing the relevant summations of the Traces.
 * Any Gamma-matrix element can be obtained as: G(n)_{ij} = RowValue[n][i] * (ColumnIndex[n][i]==j ? 1 : 0)
 */

//- The value in rows 0,1,2,3, respectively, of each gamma matrix
inline int GammaRowValue(int m, int n, int r){
  constexpr int rowValue[N_GAMMA_][N_SPIN_][2] = {{ {1,0}, {1,0}, {1,0}, {1,0} },   // G0 = 1
                                                  { {0,1}, {0,1},{0,-1},{0,-1} },   // G1 = g1
                                                  {{-1,0}, {1,0}, {1,0},{-1,0} },   // G2 = g2
                                                  {{0,-1}, {0,1},{0,-1}, {0,1} },   // G3 = g1g2
                                                  { {0,1},{0,-1},{0,-1}, {0,1} },   // G4 = g3
                                                  {{-1,0}, {1,0},{-1,0}, {1,0} },   // G5 = g1g3
                                                  {{0,-1},{0,-1},{0,-1},{0,-1} },   // G6 = g2g3
                                                  { {1,0}, {1,0},{-1,0},{-1,0} },   // G7 = g1g2g3   =  g5g4
                                                  { {1,0}, {1,0}, {1,0}, {1,0} },   // G8 = g4
                                                  { {0,1}, {0,1},{0,-1},{0,-1} },   // G9 = g1g4
                                                  {{-1,0}, {1,0}, {1,0},{-1,0} },   // G10= g2g4
                                                  {{0,-1}, {0,1},{0,-1}, {0,1} },   // G11= g1g2g4   = -g5g3
                                                  { {0,1},{0,-1},{0,-1}, {0,1} },   // G12= g3g4
                                                  {{-1,0}, {1,0},{-1,0}, {1,0} },   // G13= g1g3g4   =  g5g2
                                                  {{0,-1},{0,-1},{0,-1},{0,-1} },   // G14= g2g3g4   = -g5g1
                                                  { {1,0}, {1,0},{-1,0},{-1,0} }};  // G15= g1g2g3g4 =  g5
  return rowValue[m][n][r];
}

//- The column in which RowValue exists for each gamma matrix
inline int GammaColumnIndex(int m, int n){
  constexpr int columnIdx[N_GAMMA_][N_SPIN_] = {{ 0, 1, 2, 3 },   // G0 = 1
                                                { 3, 2, 1, 0 },   // G1 = g1
                                                { 3, 2, 1, 0 },   // G2 = g2
                                                { 0, 1, 2, 3 },   // G3 = g1g2
                                                { 2, 3, 0, 1 },   // G4 = g3
                                                { 1, 0, 3, 2 },   // G5 = g1g3
                                                { 1, 0, 3, 2 },   // G6 = g2g3
                                                { 2, 3, 0, 1 },   // G7 = g1g2g3   =  g5g4
                                                { 2, 3, 0, 1 },   // G8 = g4
                                                { 1, 0, 3, 2 },   // G9 = g1g4
                                                { 1, 0, 3, 2 },   // G10= g2g4
                                                { 2, 3, 0, 1 },   // G11= g1g2g4   = -g5g3
                                                { 0, 1, 2, 3 },   // G12= g3g4
                                                { 3, 2, 1, 0 },   // G13= g1g3g4   =  g5g2
                                                { 3, 2, 1, 0 },   // G14= g2g3g4   = -g5g1
                                                { 0, 1, 2, 3 }};  // G15= g1g2g3g4 =  g5
  return columnIdx[m][n];
}

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
inline std::vector<int> minusGamma(){
  std::vector<int> minusG{3, 6, 9, 11, 12, 14};
  return minusG;
}

//- This takes care of the index
inline std::vector<int> indexMapGamma(){
  std::vector<int> idxG(N_GAMMA_, 0);
  for(int i=0;i<N_GAMMA_;i++) idxG.at(i) = N_GAMMA_ -i -1;
  return idxG;
}


#endif
