#define CATCH_CONFIG_MAIN // This tells Catch to provide a main()
#include <catch2/catch.hpp>

#include <optstats/lsqlinear.hpp>
#include <optstats/ttest.hpp>

// sudo apt install libgsl-dev
#include <eleobert/curve_fit.hpp> // This is GPL wrapper over gsl: https://github.com/Eleobert/gsl-curve-fit
//
#include <optstats/lsqnonlinear.hpp> // derived work from eleobert gsl wrapper

//
#define STATS_ENABLE_STDVEC_WRAPPERS
#include <statslib/include/stats.hpp>

using namespace std;
using namespace optstats;

TEST_CASE("optstats leastSquaresLinearRegression y=ax+b") {
  // http://textbooks.math.gatech.edu/ila/least-squares.html
  // https://github.com/QBobWatson/ila
  //
  // https://math.stackexchange.com/questions/2591061/exponential-least-squares-equation
  //
  // https://eigen.tuxfamily.org/dox/GettingStarted.html
  // https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html
  //
  // points: (0,6), (1,0) and (2,0)
  //
  // 6 = M . 0 + B
  // 0 = M . 1 + B
  // 0 = M . 2 + B
  //
  //     0  1                6
  // A = 1  1   e = M    b = 0    where  ê = -3
  //     2  1       B        0                5
  //
  // Solution is: M=-3 B=5,  for y=Mx + B
  //
  std::vector<double> vx = {0, 1, 2};
  std::vector<double> vy = {6, 0, 0};

  // testing with mode 'default'
  auto Mb = optstats::leastSquaresLinearRegression(vx, vy);
  //
  // least squares solution is: y = -3x + 5
  //
  REQUIRE(-3 == Approx(Mb.first).epsilon(1e-5));
  REQUIRE(5 == Approx(Mb.second).epsilon(1e-5));
}

TEST_CASE("optstats leastSquaresRegression vf parabola") {
  // http://textbooks.math.gatech.edu/ila/least-squares.html
  // https://github.com/QBobWatson/ila
  //
  // https://math.stackexchange.com/questions/2591061/exponential-least-squares-equation
  //
  // https://eigen.tuxfamily.org/dox/GettingStarted.html
  // https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html
  //
  // ( − 1,1 / 2 ) , ( 1, − 1 ) , ( 2, − 1 / 2 ) , ( 3,2 )
  // y = Bx²+Cx+D

  // 1/2  = B(-1)² + C(-1) + D
  // -1   = B(1)²  + C(1)  + D
  // -1/2 = B(2)²  + C(2)  + D
  // 2    = B(3)²  + C(3)  + D
  //
  //      1 -1  1      B        1/2
  // A =  1  1  1   x= C   b =  -1
  //      4  2  1      D       -1/2
  //      9  3  1                2
  //
  // best-fit parabola: B=53/88 C=-379/440 D=-41/44

  std::vector<double> vx = {-1, 1, 2, 3};
  std::vector<double> vy = {1 / 2.0, -1, -1 / 2.0, 2};

  // y = Bx²+Cx+D
  std::vector<std::function<double(double)>> vf = {
      [](double x) { return x * x; }, [](double x) { return x; }};

  // testing with mode 'default'
  auto vA = optstats::leastSquaresRegression(vx, vy, vf,
                                             LinearSolveMode::AccurateSVD);
  //
  // least squares solution is: y = 53/88 x² -379/440 x - 41/44
  //
  REQUIRE(53 / 88.0 == Approx(vA[0]).epsilon(1e-5));
  REQUIRE(-379 / 440.0 == Approx(vA[1]).epsilon(1e-5));
  REQUIRE(-41 / 44.0 == Approx(vA[2]).epsilon(1e-5));
}

TEST_CASE("optstats leastSquaresLinearRegression basic_exponential MI") {
  // BASIC EXPONENTIAL: y = a*e^{-bx}
  //
  // https://math.stackexchange.com/questions/2591061/exponential-least-squares-equation
  //
  // by applying 'ln' to both sides, we have:
  // ln(y) = ln(a) - bx
  //
  // we will solve linear: lny = ZB x + lna
  // where:
  // lny = ln(y)
  // lna = ln(a)
  // ZB = -b
  //

  std::vector<double> vy_original = {8558, 5411, 2830, 2267, 760,
                                     549,  249,  67,   47,   43};
  std::vector<double> vx = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  //
  int N = vy_original.size();
  //
  REQUIRE(vx.size() == N);
  //
  // get vy in log natural
  //
  std::vector<double> vy(N);
  for (unsigned i = 0; i < N; i++)
    vy[i] = ::log(vy_original[i]); // log natural

  // get linear result
  auto res = optstats::leastSquaresLinearRegression(vx, vy);
  double ZB = res.first;
  double lna = res.second;
  //
  double realA = ::exp(lna);
  double realB = -ZB;

  // y = a*e^{-bx}
  REQUIRE(10565.97 == Approx(realA).epsilon(1e-5));
  REQUIRE(0.645643 == Approx(realB).epsilon(1e-5));
  //
  double R2 = optstats::calcR2(vx, vy_original, [realA, realB](double x) {
    return realA * ::exp(-realB * x);
  });
  // why fit is not near 0.99?
  REQUIRE(R2 >= 0.93);
}

/*
double gaussian(double x, double a, double b, double c) {
  const double z = (x - b) / c;
  return a * std::exp(-0.5 * z * z);
}

template <typename Container>
Container generate_linspace(typename Container::value_type a,
                            typename Container::value_type b, size_t n) {
  assert(b > a);
  assert(n > 1);

  Container res(n);
  const auto step = (b - a) / (n - 1);
  auto val = a;
  for (auto &e : res) {
    e = val;
    val += step;
  }
  return res;
}

TEST_CASE("optstats eleobert non-linear curve fit") {

  auto device = std::random_device();
  auto gen = std::mt19937(device());

  auto xs = generate_linspace<std::vector<double>>(0.0, 1.0, 300);
  auto ys = std::vector<double>(xs.size());

  double a = 5.0, b = 0.4, c = 0.15;

  for (size_t i = 0; i < xs.size(); i++) {
    auto y = gaussian(xs[i], a, b, c);
    auto dist = std::normal_distribution(0.0, 0.1 * y);
    ys[i] = y + dist(gen);
  }

  auto r = curve_fit(gaussian, {1.0, 0.0, 1.0}, xs, ys);

  std::cout << "result: " << r[0] << ' ' << r[1] << ' ' << r[2] << '\n';
  std::cout << "error : " << r[0] - a << ' ' << r[1] - b << ' ' << r[2] - c
            << '\n';

  REQUIRE((r[0] - a) == Approx(0.01).epsilon(1e-1));
}
*/

// ====================

// y = a*e^{-bx}
double model_exp_mi1(double x, double a, double b) {
  return a * std::exp(-b * x);
}

TEST_CASE("optstats nonlineargsl fit model_exp_mi1") {
  // y = a*e^{-bx}
  std::vector<double> ys = {8558, 5411, 2830, 2267, 760, 549, 249, 67, 47, 43};
  std::vector<double> xs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  auto res = optstats::leastSquaresNonLinearRegression(xs, ys, {0.0, 0.0},
                                                       model_exp_mi1);

  // y = a*e^{-bx}
  //
  double realA = res[0];
  double realB = res[1];
  //
  REQUIRE(8666.36934 == Approx(realA).epsilon(1e-5));
  REQUIRE(0.52034 == Approx(realB).epsilon(1e-5));
  //
  double R2 = optstats::calcR2(
      xs, ys, [realA, realB](double x) { return realA * ::exp(-realB * x); });

  // std::cout << "R2=" << R2 << std::endl;
  // why fit is not near 0.99?
  REQUIRE(R2 >= 0.99);
}

// =====================

void printv(std::vector<double> v) {
  for (unsigned i = 0; i < v.size(); i++)
    std::cout << v[i] << " ";
  std::cout << std::endl;
}

TEST_CASE("optstats statslib dt t test") {
  // https://statslib.readthedocs.io/en/latest/api/t.html
  // density function for T-student
  double x1 = stats::dt(0.37, 11, false);
  // distribution function for T-student (CDF)
  std::vector<double> vt = {0.0, 1.0, 2.0};
  auto vp = stats::pt(vt, 4, false);
  // vp: 0.5 0.81305 0.941942
  //
  // Degree of Freedom DoF = 4 (T-Table)
  // t.50 = 0, t.75=0.741, t.80 = 0.941, t.85 = 1.190, t.90=1.533, t.95=2.132
  // t=0 = 0.5
  REQUIRE(0.5 == Approx(vp[0]).epsilon(1e-2));
  // t=1 >= 0.8 and <= 0.85
  REQUIRE(vp[1] >= 0.8);
  REQUIRE(vp[1] <= 0.85);
  // t=2 >= 1.5 and <= 2.1
  REQUIRE(vp[2] >= 0.9);
  REQUIRE(vp[2] <= 0.95);
}

TEST_CASE("optstats students t test") {
  // https://en.wikipedia.org/wiki/Student%27s_t-test#Independent_(unpaired)_samples
  // Section "Worked Samples"
  //
  std::vector<double> a1 = {30.02, 29.99, 30.11, 29.97, 30.01, 29.99};
  std::vector<double> a2 = {29.89, 29.93, 29.72, 29.98, 30.02, 29.98};

  // Null Hypothesis: means of a1 and a2 are the same
  double x1 = optstats::mean(a1);
  double x2 = optstats::mean(a2);
  REQUIRE(0.095 == Approx(x1 - x2).epsilon(1e-4));

  {
    // test with unequal variances (two-sided)
    auto [ttest, dof] = optstats::getIndependentTwoSampleTTest(a1, a2, false);

    REQUIRE(1.959 == Approx(ttest).epsilon(1e-4));
    REQUIRE(7.031 == Approx(dof).epsilon(1e-4));

    double p =
        optstats::pIndependentTwoSampleTTest(a1, a2, TestSides::Both, false);
    // std::cout << p << std::endl;
    REQUIRE(0.09077 == Approx(p).epsilon(1e-4));
  }

  {
    // test with equal variances (two-sided)
    auto [ttest, dof] = optstats::getIndependentTwoSampleTTest(a1, a2, true);

    REQUIRE(1.959 == Approx(ttest).epsilon(1e-4));
    REQUIRE(10 == Approx(dof).epsilon(1e-4));

    double p =
        optstats::pIndependentTwoSampleTTest(a1, a2, TestSides::Both, true);
    // std::cout << p << std::endl;
    REQUIRE(0.07857 == Approx(p).epsilon(1e-4));
  }

  {
    // test with unequal variances (one-tailed) - greater
    REQUIRE(x1 >= x2);

    double p =
        optstats::pIndependentTwoSampleTTest(a1, a2, TestSides::Greater, false);
    REQUIRE(0.04539 == Approx(p).epsilon(1e-4));
  }

  {
    // test with unequal variances (one-tailed) - less
    REQUIRE(!(x1 <= x2));
    // note that this will yield a high p-value, as x1 is NOT less than x2

    double p =
        optstats::pIndependentTwoSampleTTest(a1, a2, TestSides::Less, false);
    REQUIRE(0.9546 == Approx(p).epsilon(1e-4));
  }
}
