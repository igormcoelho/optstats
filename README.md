## optstats

A C++ library for useful tools on optimization, statistics and curve fitting.

These currently include:

- linear regressions (by means of Eigen project)
- non-linear regressions (by means of GNU GSL project)
- T test
   * Student's t test and Welch's test for unequal variances (by means of statslib/GCEM project)

## Examples for Least Squares on C++

### least squares linear regression

```{.cpp}
  #include "optstats.hpp"
  // ...
  std::vector<double> vx = {0, 1, 2};
  std::vector<double> vy = {6, 0, 0};

  // testing with mode 'default'
  auto Mb = optstats::leastSquaresLinearRegression(vx, vy);
  
  // least squares solution is: y = -3x + 5
  assert(Mb.first == -3.0);
  assert(Mb.second == 5.0);
```

### least squares linear regression (for parabolic curve)

```{.cpp}
  #include "lsqlinear.hpp"
  // ...
  std::vector<double> vx = {-1, 1, 2, 3};
  std::vector<double> vy = {1 / 2.0, -1, -1 / 2.0, 2};

  // y = Bx²+Cx+D
  std::vector<std::function<double(double)>> vf = {
      [](double x) { return x * x; }, [](double x) { return x; }};

  // testing with mode 'SVD'
  auto vA = optstats::leastSquaresRegression(vx, vy, vf, LinearSolveMode::AccurateSVD);
  //
  // least squares solution is: y = 53/88 x² -379/440 x - 41/44
  //
  assert(53 / 88.0 == vA[0]);
  assert(-379 / 440.0 == vA[1]);
  assert(-41 / 44.0 == vA[2]);
```

### least squares nonlinear regression (with log transform)

See log transform strategies:

- https://math.stackexchange.com/questions/1488747/least-square-approximation-for-exponential-functions
- https://math.stackexchange.com/questions/2591061/exponential-least-squares-equation

There's also a test on `tests/` that works on that using Eigen.
Note that error is greater than a real nonlinear approach (such as with Levenberg-Marquardt).

### least squares nonlinear regression (using Levenberg-Marquardt on GSL)

```
  #include "lsqnonlinear.hpp"
  // ...

  // y = a*e^{-bx}
  double model_exp_mi1(double x, double a, double b) {
    return a * std::exp(-b * x);
  }

  // ...

  // y = a*e^{-bx}
  std::vector<double> ys = {8558, 5411, 2830, 2267, 760, 549, 249, 67, 47, 43};
  std::vector<double> xs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  auto res = optstats::leastSquaresNonLinearRegression(xs, ys, {0.0, 0.0}, model_exp_mi1);

  // y = a*e^{-bx}
  //
  double realA = res[0];
  double realB = res[1];
  //
  assert(8666.36934 == realA);
  assert(0.52034 == realB);
  //
  double R2 = optstats::calcR2(
      xs, ys, [realA, realB](double x) { return realA * ::exp(-realB * x); });

  // expects very good fit (by means of GNU GSL)
  assert(R2 >= 0.99);
```


### How to use

In order to use linear regressions of `lsqlinear.hpp`, just `#include "lsqlinear.hpp"`.
You will need to include Eigen support, just include `-Ipath/to/eigen`.

For nonlinear regressions, one needs to `#include "lsqnonlinear.hpp"`.
In this case, GNU GSL will be required, so as flag `-lgsl`. 
On Ubuntu 20.04, just `apt install libgsl-dev`.

## Examples for Student's T test on C++

Learn more about T vs Normal:

- https://www.statology.org/normal-distribution-vs-t-distribution/
- https://en.wikipedia.org/wiki/Student%27s_t-test

Two sided independent t-test (from Wikipedia).

```
  #include "ttest.hpp"
  //...
  std::vector<double> a1 = {30.02, 29.99, 30.11, 29.97, 30.01, 29.99};
  std::vector<double> a2 = {29.89, 29.93, 29.72, 29.98, 30.02, 29.98};

  // Null Hypothesis: means of a1 and a2 are the same
  double x1 = optstats::mean(a1);
  double x2 = optstats::mean(a2);
  assert(0.095 == (x1 - x2));

  // test with unequal variances
  auto [ttest, dof] = optstats::getIndependentTwoSampleTTest(a1, a2);

  assert(1.959 == ttest);  // check t-value
  assert(7.031 == dof);    // check degrees of freedom

  // p-value for two-sided test (note the 'TestSides::Two')
  double p = optstats::pIndependentTwoSampleTTest(a1, a2, TestSides::Both);
  assert(0.09077 == p);
```

### How to use

In order to use linear regressions of `ttest.hpp`, just `#include "ttest.hpp"`.
You will need to include `GCEM` and `stat` library support, just include `-Ipath/to/statlib` (both are header-only).


## Tests

There unit tests on `tests/` folder, feel free to use them as examples.

## TODO

- Levenberg-Marquardt algorithm used from GSL wrapper (provided by Eleobert) can also be found on "Eigen unsupported" (TODO: investigate this usage)
   * more about L-M: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
- TODO consider other statistic techniques, such as: https://en.wikipedia.org/wiki/Akaike_information_criterion
    * https://en.wikipedia.org/wiki/Foundations_of_statistics
- TODO consider Armadillo for basic operations: http://arma.sourceforge.net/docs.html

## License

Free Software - Feel free to use it and redistribute it

All novel code written here is MIT License, EXCEPT for dependencies (each .hpp has one).

Note that:

- Eigen: Mozilla Public License 2.0 (MPL2) - https://eigen.tuxfamily.org
- GNU GSL: GNU Public License 3.0 (GPL3) - https://www.gnu.org/software/gsl/
- stats library (statslib): Apache Version 2 (Apache 2.0) - https://github.com/kthohr/stats/
   * gcem is a dependency from stats (library) - Apache 2.0 - https://github.com/kthohr/gcem/

Depending on the mix, it can be GPL-like or MIT-like (better explanations may come in the future...).

On short:

- `lsqlinear.hpp` : MIT + Eigen => License MPL2
- `lsqnonlinear.hpp` : MIT + GNU GSL => License GPL3
- `ttest.hpp` : MIT + (stats library + GCEM) => MIT/Apache 2.0

Copyleft 2021
