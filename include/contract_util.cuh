#ifndef _CONTRACT_UTIL_H
#define _CONTRACT_UTIL_H

#include <util_mugiq.h>

using namespace quda;

constexpr int cSize = 4096; // Size of constant memory symbols

__constant__ char cGamma[cSize]; // constant buffer for gamma matrices on GPU


/**
 * Hard-coded gamma coefficients for the DeGrand-Rossi basis
 * Gamma-index notation is: G{x,y,z,t} = G{1,2,3,4}
 * Gamma matrices are defined as: G(n) = g1^n0 . g2^n1 . g3^n2 . g4^n3, where n = n0*2^0 + n1*2^1 + n2*2^2 + n3*2^3
 * This parametrization helps in efficient unrolling and usage when performing contractions on the GPU
 * Any Gamma-matrix element can be obtained as: G(n)_{ij} = RowValue[n][i] * (ColumnIndex[n][i]==j ? 1 : 0)
 */

//- The value in rows 0,1,2,3, respectively, of each gamma matrix
constexpr int GammaRowValue[N_GAMMA_][N_SPIN_][2] = {{ {1,0}, {1,0}, {1,0}, {1,0} },   /* G0 = 1 */
						     { {0,1}, {0,1},{0,-1},{0,-1} },   /* G1 = g1 */
						     {{-1,0}, {1,0}, {1,0},{-1,0} },   /* G2 = g2 */
						     {{0,-1}, {0,1},{0,-1}, {0,1} },   /* G3 = g1 g2 */
						     { {0,1},{0,-1},{0,-1}, {0,1} },   /* G4 = g3 */
						     {{-1,0}, {1,0},{-1,0}, {1,0} },   /* G5 = g1 g3 */
						     {{0,-1},{0,-1},{0,-1},{0,-1} },   /* G6 = g2 g3 */
						     { {1,0}, {1,0},{-1,0},{-1,0} },   /* G7 = g1 g2 g3 */
						     { {1,0}, {1,0}, {1,0}, {1,0} },   /* G8 = g4 */
						     { {0,1}, {0,1},{0,-1},{0,-1} },   /* G9 = g1 g4 */
						     {{-1,0}, {1,0}, {1,0},{-1,0} },   /* G10= g2 g4 */
						     {{0,-1}, {0,1},{0,-1}, {0,1} },   /* G11= g1 g2 g4 */
						     { {0,1},{0,-1},{0,-1}, {0,1} },   /* G12= g3 g4 */
						     {{-1,0}, {1,0},{-1,0}, {1,0} },   /* G13= g1 g3 g4 */
						     {{0,-1},{0,-1},{0,-1},{0,-1} },   /* G14= g2 g3 g4 */
						     { {1,0}, {1,0},{-1,0},{-1,0} }};  /* G15= g1 g2 g3 g4 */


//- The column in which RowValue exists for each gamma matrix
constexpr int GammaColumnIndex[N_GAMMA_][N_SPIN_] = {{ 0, 1, 2, 3 },   /* G0 = 1 */
						     { 3, 2, 1, 0 },   /* G1 = g1 */
						     { 3, 2, 1, 0 },   /* G2 = g2 */
						     { 0, 1, 2, 3 },   /* G3 = g1 g2 */
						     { 2, 3, 0, 1 },   /* G4 = g3 */
						     { 1, 0, 3, 2 },   /* G5 = g1 g3 */
						     { 1, 0, 3, 2 },   /* G6 = g2 g3 */
						     { 2, 3, 0, 1 },   /* G7 = g1 g2 g3 */
						     { 2, 3, 0, 1 },   /* G8 = g4 */
						     { 1, 0, 3, 2 },   /* G9 = g1 g4 */
						     { 1, 0, 3, 2 },   /* G10= g2 g4 */
						     { 2, 3, 0, 1 },   /* G11= g1 g2 g4 */
						     { 0, 1, 2, 3 },   /* G12= g3 g4 */
						     { 3, 2, 1, 0 },   /* G13= g1 g3 g4 */
						     { 3, 2, 1, 0 },   /* G14= g2 g3 g4 */
						     { 0, 1, 2, 3 }};  /* G15= g1 g2 g3 g4 */


//- Structure that will eventually be copied to GPU __constant__ memory
template <typename Float>
struct GammaCoeff{
  complex<Float> row_value[N_GAMMA_][N_SPIN_];
  int column_index[N_GAMMA_][N_SPIN_];
}; // GammaCoeff
  


#endif // _CONTRACT_UTIL_H
