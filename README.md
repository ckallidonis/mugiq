# MuGiq
An interface to perform MultiGrid deflation in GPUs using the QUDA library, for the efficient computation of disconnected diagrams and other measurements related to Lattice QCD applications.


## External dependencies

This interface depends on:
* The QUDA ([https://github.com/lattice/quda](https://github.com/lattice/quda)) library
* The PRIMME ([https://github.com/primme/primme](https://github.com/primme/primme)) library, which in turn depends on the [MAGMA](https://icl.cs.utk.edu/magma/) library
* MPI or QMP/QIO
* The CUDA toolkit


## Installation

This package uses CMake for the installation process. It is recommended to install MuGiq in a separate directory from the source directory.
Below are the steps for installing MuGiq using CMake:
* Create an `install` directory separate from the MuGiq source directory.
* `cd <path-to-install-dir>`
* Perhaps the most convenient way to configure installation options is to run `ccmake <path-to-MuGiq-src>` in order to set the options in an interactive manner.
* Alternatively one can run `cmake <path-to-MuGiq-src> -D<option1>=value -D<option2>=value ...` instead.
* run `make -j<N>` to install the package using N parallel jobs.


## Author and Contact

* **Christos Kallidonis** - College of William & Mary
* Web: [http://christoskallidonis.com](http://christoskallidonis.com).
