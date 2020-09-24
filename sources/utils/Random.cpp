#include "Random.h"

#include "assert.h"

// ---------------------------------------------------------------------------
namespace NilDa
{

// Scalar

Scalar Random()
{
#if defined(ND_RAND)

  std::random_device rand;

  std::mt19937 genRand(rand());

#else

  std::mt19937 genRand(123);

#endif

  std::uniform_real_distribution<Scalar> dis(0, 1);

  return dis(genRand);
}

Scalar Random(const int seed)
{
  std::mt19937 genRand(seed);

  std::uniform_real_distribution<Scalar> dis(0, 1);

  return dis(genRand);
}

Scalar Random(const Scalar low, const Scalar high)
{
#if defined(ND_RAND)

  std::random_device rand;

  std::mt19937 genRand(rand());

#else

  std::mt19937 genRand(123);

#endif

  assert(low != high);

  std::uniform_real_distribution<Scalar> dis(low, high);

  return dis(genRand);
}

Scalar Random(const int seed, const Scalar low, const Scalar high)
{
  std::mt19937 genRand(seed);

  assert(low != high);

  std::uniform_real_distribution<Scalar> dis(low, high);

  return dis(genRand);
}

// Vector

void Random(Vector& v)
{
#if defined(ND_RAND)

  std::random_device rand;

  std::mt19937 genRand(rand());

#else

  std::mt19937 genRand(123);

#endif

  std::uniform_real_distribution<Scalar> dis(0, 1);

  v = v.unaryExpr([&](Scalar dummy){return dis(genRand);});
}

void Random(const int seed, Vector& v)
{
  std::mt19937 genRand(seed);

  std::uniform_real_distribution<Scalar> dis(0, 1);

  v = v.unaryExpr([&](Scalar dummy){return dis(genRand);});
}

void Random(const Scalar low, const Scalar high, Vector& v)
{
#if defined(ND_RAND)

  std::random_device rand;

  std::mt19937 genRand(rand());

#else

  std::mt19937 genRand(123);

#endif

  assert(low != high);

  std::uniform_real_distribution<Scalar> dis(low, high);

  v = v.unaryExpr([&](Scalar dummy){return dis(genRand);});
}

void Random(const int seed, const Scalar low, const Scalar high, Vector& v)
{
  std::random_device rand;

  std::mt19937 genRand(seed);

  assert(low != high);

  std::uniform_real_distribution<Scalar> dis(low, high);

  v = v.unaryExpr([&](Scalar dummy){return dis(genRand);});
}

// Matrix

void Random(Matrix& M)
{
#if defined(ND_RAND)

  std::random_device rand;

  std::mt19937 genRand(rand());

#else

  std::mt19937 genRand(123);

#endif

  std::uniform_real_distribution<Scalar> dis(0, 1);

  M = M.unaryExpr([&](Scalar dummy){return dis(genRand);});
}

void Random(const int seed, Matrix& M)
{
  std::mt19937 genRand(seed);

  std::uniform_real_distribution<Scalar> dis(0, 1);

  M = M.unaryExpr([&](Scalar dummy){return dis(genRand);});
}

void Random(const Scalar low, const Scalar high, Matrix& M)
{
#if defined(ND_RAND)

  std::random_device rand;

  std::mt19937 genRand(rand());

#else

  std::mt19937 genRand(123);

#endif

  assert(low != high);

  std::uniform_real_distribution<Scalar> dis(low, high);

  M = M.unaryExpr([&](Scalar dummy){return dis(genRand);});
}

void Random(const int seed, const Scalar low, const Scalar high, Matrix& M)
{
  std::random_device rand;

  std::mt19937 genRand(seed);

  assert(low != high);

  std::uniform_real_distribution<Scalar> dis(low, high);

  M = M.unaryExpr([&](Scalar dummy){return dis(genRand);});
}


} // namespace
