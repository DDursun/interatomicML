# Declare the class with cdef
cdef extern from "scalapack.h":
    void blacs_get_(int*, int*, int*)
    void blacs_pinfo_(int*, int*)
    void blacs_gridinit_(int*, char*, int*, int*)
    void blacs_gridinfo_(int*, int*, int*, int*, int*)
    void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*, int*)
    void pdpotrf_(char*, int*, double*, int*, int*, int*, int*)
    void psgels_(char*, int*, int*, int*, double*, int*, int*, int*, double*, int*, int*, int*, double*, int*, int*)
    void pdgels_(char*, int*, int*, int*, double*, int*, int*, int*, double*, int*, int*, int*, double*, int*, int*)
    void pdgesvd_(char*, char*, int*, int*, double*, int*, int*, int*, double*, double*, int*, int*, int*, double*, int*, int*, int*, double*, int*, int*)
    void blacs_gridexit_(int*)
    int numroc_(int*, int*, int*, int*, int*)
