#ifndef _LOOP_MUGIQ_H
#define _LOOP_MUGIQ_H

#include <mugiq.h>
#include <eigsolve_mugiq.h>
#include <util_mugiq.h>
#include <displace.h>
#include <mpi.h>

using namespace quda;

template <typename Float, QudaFieldOrder fieldOrder>
class Loop_Mugiq {

private:

  struct LoopComputeParam;
  
  LoopComputeParam *cPrm; // Loop computation Parameter structure

  Displace<Float> *displace;  // structure holding the displacements
  
  Eigsolve_Mugiq *eigsolve; // The eigsolve object (This class is a friend of Eigsolve_Mugiq)

  ColorSpinorField *refVec; // Reference vector, whose parameters will be used throughout the class

  //- MPI/Communication-related parameters for "space" processes
  //- These are the processes that have the same time-coordinate
  MPI_Comm COMM_SPACE; //- The Communicator for "space" processes
  int space_rank;      //- Rank numbering
  int space_size;      //- Size of the COMM_SPACE space
  int cRank;           //- parameter determining the rank numbering
  int tCoord;          //- The time coordinate of each process, will be used as the "color" of the COMM_SPACE space

  
  //- MPI/Communication-related parameters for "time" processes
  //- These are the processes that have all but time coordinates equal to 0
  MPI_Comm COMM_TIME;         //- The Communicator
  int time_rank;              //- Rank numbering
  int time_size;              //- Size of the COMM_TIME space
  const int time_tag = 1000;  //- Tag that will be used to determine the "color" of the COMM_TIME space
  int time_color;             //- The "color" of the "time" processes
  MuGiqBool IamTimeProcess;   //- Whether a process belongs to the COMM_TIME space

  
  MuGiqBool commsAreSet;
  
  //- Data buffers
  complex<Float> *dataPos_d = nullptr;      // Device Position space correlator (local)
  complex<Float> *dataPosMP_d = nullptr;    // Device Position space correlator (local), with changed index order for Mom. projection 
  complex<Float> *dataMom_d = nullptr;      // Device output buffer of cuBlas (local)
  complex<Float> *dataPos   = nullptr;      // Host Position space correlator (local)
  complex<Float> *dataMom_h = nullptr;      // Host output of cuBlas momentum projection (local)
  complex<Float> *dataMom   = nullptr;      // Host Globally summed momentum projection buffer (local)
  complex<Float> *dataMom_bcast = nullptr;  // Host Final result (global summed, gathered, broadcasted) of momentum projection

  complex<Float> *phaseMatrix_d;  // Device buffer of the phase matrix
  
  const size_t SizeCplxFloat = sizeof(complex<Float>);

  long long nElemMomTotPerLoop; // Number of elements in global momentum-space data buffers, per loop
  long long nElemMomLocPerLoop; // Number of elements in local  momentum-space data buffers, per loop
  long long nElemPosLocPerLoop; // Number of elements in local  position-space data buffers, per loop
  
  long long nElemMomTot; // Total Number of elements in global momentum-space data buffers
  long long nElemMomLoc; // Total Number of elements in local  momentum-space data buffers
  long long nElemPosLoc; // Total Number of elements in local  position-space data buffers
  long long nElemPhMat;  // Number of elements in phase matrix

  MuGiqBool MomProjDone; // Whether momentum projection has been completed

  MuGiqBool writeDataPos; // Whether to write the position-space loop data
  MuGiqBool writeDataMom; // Whether to write the momentum-space loop data
  
  std::string momSpaceFilename; //- HDF5 filename for momentum-space filename
  std::string posSpaceFilename; //- HDF5 filename for position-space filename

  
  /** @brief Set up communicators required for the momentum projection and the HDF5 writing
   */
  void setupComms();
  
  /** @brief Prolongate the coarse eigenvectors to fine fields
   */
  void prolongateEvec(ColorSpinorField *fineEvec, ColorSpinorField *coarseEvec);

  
  /** @brief Print the Parameters of the Loop computation
   */
  void printLoopComputeParams();

  /** @brief Allocate host and device data buffers
   */
  void allocateDataMemory();

  /** @brief Free host and device data buffers
   */
  void freeDataMemory();

  /** @brief Wrapper to copy the gamma matrix coefficients to GPU __constant__ memory
   */
  void copyGammaToConstMem();

  /** @brief Create the Phase matrix, needed for Momentum Projection (Fourier Transform)
   */
  void createPhaseMatrix();

  /** @@brief Perform Fourier Transform (Momentum Projection) on the loop trace
   */
  void performMomentumProjection();

  /** @brief Write the momentum-space loop data in HDF5 format
   */
  void writeLoopsHDF5_Mom();

  /** @brief Write the position-space loop data in HDF5 format
   */
  void writeLoopsHDF5_Pos();

  
public:

  Loop_Mugiq(MugiqLoopParam *loopParams_, Eigsolve_Mugiq *eigsolve_);
  ~Loop_Mugiq();

  
  /** @brief Write the Loop data in an HDF5 file (called from the interface)
   */
  void writeLoopsHDF5();

  
  /** @brief Wrapper to create the coarse part of the loop
   */
  void computeCoarseLoop();
  
}; // class Loop_Mugiq




template <typename Float, QudaFieldOrder fieldOrder>
struct Loop_Mugiq<Float, fieldOrder>::LoopComputeParam {

  const int nG = N_GAMMA_;   // Number of Gamma matrices (currents, =16)
  const int momDim = MOM_DIM_;  // Momenta dimensions (=3)

  int Nmom;                                 // Number of Momenta
  LoopFTSign FTSign;                        // Sign of the Fourier Transform
  int *momMatrix;                           // Momenta Matrix, follows lexicographic order momDim-inside-Nmom
  
  int max_depth;                // maximum depth of transverse shift length (for later)

  MuGiqBool doMomProj;          // whether to do Momentum projection, if false then the position-space trace will be saved
  MuGiqBool doNonLocal;         // whether to compute loop for non-local currents

  int localL[N_DIM_];           // local dimensions
  int totalL[N_DIM_];           // global dimensions

  int nParity;  // Number of parities in ColorSpinorFields (even/odd or single-parity spinors)
  int volumeCB; // Checkerboard volume of the vectors, if even/odd then volumeCV is half the total volume
  
  int locT;                     // local  time dimension
  int totT;                     // global time dimension
  long long locV4 = 1;          // local spatial volume
  long long locV3 = 1;          // local 3d volume (no time)
  long long totV3 = 1;          // global 3d volume (no time)

  LoopCalcType calcType; // Type of computation that will take place


  int nDispEntries;                     // Number of displacement entries
  std::vector<std::string> dispEntry;   // The displacement entry, e.g. +z:1,8
  std::vector<std::string> dispString;  // The displacement string, e.g. +z,-x, etc
  std::vector<int> dispStart;           // Displacement start
  std::vector<int> dispStop;            // Displacement stop
  std::vector<int> nLoopPerEntry;       // Number of loop traces per displacement entry = dispStop - dispStart +1
  std::vector<int> nLoopOffset;         // Number of loop traces up to given entry

  int nLoop; // Total number of loop traces
  int nData; // Total number of loop data (nLoop*Ngamma)
  
  MuGiqBool init; // Whether the structure has been initialized

  
  LoopComputeParam(MugiqLoopParam *loopParams, ColorSpinorField *x) :
    Nmom(loopParams->Nmom),
    FTSign(loopParams->FTSign),
    max_depth(0),
    doMomProj(loopParams->doMomProj),
    doNonLocal(loopParams->doNonLocal),
    localL{0,0,0,0},
    totalL{0,0,0,0},
    nParity(x->SiteSubset()),
    volumeCB(x->VolumeCB()),
    locT(0), totT(0),
    locV4(1), locV3(1), totV3(1),
    calcType(loopParams->calcType),
    nDispEntries(0),
    nLoop(0), nData(0),
    init(MUGIQ_BOOL_FALSE)
  {
    for(int i=0;i<N_DIM_;i++){
      localL[i] = x->X(i);
      totalL[i] = localL[i] * comm_dim(i);
      locV4 *= localL[i];
      if(i<N_DIM_-1){
	locV3 *= localL[i];
	totV3 *= totalL[i];
      }
    }
    locT = localL[N_DIM_-1];
    totT = totalL[N_DIM_-1];

    if(doMomProj){
      momMatrix = static_cast<int*>(calloc(Nmom*momDim, sizeof(int)));
      for(int im=0;im<Nmom;im++)
	for(int id=0;id<momDim;id++)
	  momMatrix[MOM_MATRIX_IDX(id,im)] = loopParams->momMatrix[im][id];
    }
    
    if(doNonLocal){
      nDispEntries = loopParams->disp_str.size();
      if(nDispEntries != static_cast<int>(loopParams->disp_start.size()) ||
	 nDispEntries != static_cast<int>(loopParams->disp_stop.size()))
	errorQuda("Displacement string length not compatible with displacement limits length\n");

      for(int id=0;id<nDispEntries;id++){
	dispEntry.push_back(loopParams->disp_entry.at(id));
	dispString.push_back(loopParams->disp_str.at(id));
	dispStart.push_back(loopParams->disp_start.at(id));
	dispStop.push_back(loopParams->disp_stop.at(id));

	//-Some sanity checks
	if(dispStart.at(id) > dispStop.at(id)){
	  warningQuda("Stop length is smaller than Start length for displacement %d. Will switch lengths!\n", id);
	  int s = dispStart.at(id);
	  dispStart.at(id) = dispStop.at(id);
	  dispStop.at(id) = s;
	}

	nLoopPerEntry.push_back(dispStop.at(id) - dispStart.at(id) + 1);	
	nLoop += nLoopPerEntry.at(id);

	int osum = 1; //-start with ultra-local
	for(int is=0;is<id;is++)
	  osum += nLoopPerEntry.at(is);
	nLoopOffset.push_back(osum);
	
      } //- for disp entries
      nLoop += 1; // Don't forget ultra-local case!!
    }
    else{
      nDispEntries = 0; //- only ultra-local
      nLoop = 1;
    }
    nData = nLoop*nG;
    
    printfQuda("%s: Loop compute parameters are set\n", __func__);

    init = MUGIQ_BOOL_TRUE;
  } // constructor

  ~LoopComputeParam(){
    init = MUGIQ_BOOL_FALSE;
    if(doMomProj){
      free(momMatrix);
      momMatrix = nullptr;
    }
  }

};


/************************************************************/
//- Forward declarations of functions called within Loop_Mugiq


/** @brief Define the Gamma matrix coefficient structure and copy it to GPU __constant__ memory
 */
template <typename Float>
void copyGammaCoeffStructToSymbol();


/** @brief Define the Gamma matrix mapping structure and copy it to GPU __constant__ memory
 */
template <typename Float>
void copyGammaMapStructToSymbol();


/** @brief Create the phase matrix on GPU
 */
template <typename Float>
void createPhaseMatrixGPU(complex<Float> *phaseMatrix_d, const int* momMatrix_h,
                          long long locV3, int Nmom, int FTSign,
			  const int localL[], const int totalL[]);



/** @brief Perform the loop contractions
 */
template <typename Float, QudaFieldOrder fieldOrder>
void performLoopContraction(complex<Float> *loopData_d, ColorSpinorField *eVecL, ColorSpinorField *eVecR, Float sigma);



/** @brief Convert buffer index order from QUDA-Even/Odd (xyzt-inside-Gamma-inside-nLoop) to full lexicographic as
 * v3 + locV3*g + locV3*Ngamma*l+locV3+Ngamma*nLoop*tt
 */
template <typename Float>
void convertIdxOrder_mapGamma(complex<Float> *dataPosMP_d, const complex<Float> *dataPos_d,
			      int nData, int nLoop, int nParity, int volumeCB, const int localL[]);




#endif // _LOOP_MUGIQ_H
