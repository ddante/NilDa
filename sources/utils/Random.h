#ifndef RANDOM_H
#define RANDOM_H

#include "primitives/Scalar.h"
#include "primitives/Vector.h"
#include "primitives/Matrix.h"

#include <random>

// ---------------------------------------------------------------------------
namespace NilDa
{

// Scalar
Scalar Random();

Scalar Random(const int seed);

Scalar Random(const Scalar low, const Scalar high);

Scalar Random(const int seed, const Scalar low, const Scalar high);

// Vector

void Random(Vector& v);

void Random(const int seed, Vector& v);

void Random(const Scalar low, const Scalar high, Vector& v);

void Random(const int seed, const Scalar low, const Scalar high, Vector& v);

// Matrix

void Random(Matrix& M);

void Random(const int seed, Matrix& M);

void Random(const Scalar low, const Scalar high, Matrix& M);

void Random(const int seed, const Scalar low, const Scalar high, Matrix& M);

} // namespace

#endif
