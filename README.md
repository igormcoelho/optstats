## optstats

A C++ library for useful optimization, statistics and fitting tools in C++.

These currently include:

- linear regressions (by means of Eigen project)
- non-linear regressions (by means of GSL project)

## Examples

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
  #include "optstats.hpp"
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

### least squares nonlinear regression (using GSL)

```
  #include "nonlineargsl.hpp"
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


## How to use

In order to use linear regressions of `optstats.hpp`, just `#include "optstats.hpp"`.
You will need to include Eigen support, just include `-Ipath/to/eigen`.

For nonlinear regressions, one needs to `#include "nonlineargsl.hpp"`.
In this case, GNU GSL will be required, so as flag `-lgsl`. 
On Ubuntu 20.04, just `apt install libgsl-dev`.


## Tests

There unit tests on `tests/` folder, feel free to use them as examples.


## License

Free Software - Feel free to use it and redistribute it

Note that:

Eigen has its own free license

GSL has its own free license

Depending on the mix, it can be GPL-like or MIT-like (better explanations may come in the future)

Copyleft 2021