#ifndef SCALAR_H
#define SCALAR_H

// --------------------------------------------------------------------------- 

#if defined(ND_SP)

// Define scalar as float 

namespace NilDa
{
    typedef float Scalar;
}

#elif defined(ND_DP)

// Define scalar as double 

namespace NilDa
{
    typedef double Scalar;
}

#elif defined(ND_LP)

// Define Scalar as long double

namespace NilDa
{
    typedef long  double Scalar;
}

#else 

    #error "Precision for the scalar not set. Specify either NP_SP ND_DP or ND_LP"

#endif

#endif