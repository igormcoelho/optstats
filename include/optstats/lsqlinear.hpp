#ifndef OPTSTATS_LINREG_HPP
#define OPTSTATS_LINREG_HPP

#include <Eigen/Dense>
#include <functional>
#include <iostream>
#include <vector>

using namespace std;
using namespace Eigen;

namespace optstats {

// From Eigen documentation
// https://eigen.tuxfamily.org/dox/GettingStarted.html
// https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html
// The three methods discussed on this page are the SVD decomposition,
// the QR decomposition and normal equations.
// Of these, the SVD decomposition is generally the most accurate but the
// slowest, normal equations is the fastest but least accurate,
// and the QR decomposition is in between.
//
enum class LinearSolveMode {
  LRDefault = 0, // adopting FastestNormal
  AccurateSVD = 1,
  BalancedQR = 2,
  FastestNormal = 3
};

// output pair (M,B), for equation: y=Mx + B
std::pair<double, double> leastSquaresLinearRegression(
    const std::vector<double> &vx, const std::vector<double> &vy,
    LinearSolveMode mode = LinearSolveMode::LRDefault) {
  assert(vx.size() == vy.size());
  int N = vx.size();
  assert(N > 0);
  //
  // for every point i in (vx,vy)
  // vy[i] = M . vx[i] + B
  //
  // Create system Ae = b, looking for solution ê
  //

  MatrixXf A = MatrixXf::Zero(N, 2);
  for (unsigned i = 0; i < N; i++) {
    A(i, 0) = vx[i];
    A(i, 1) = 1.0;
  }

  VectorXf b = VectorXf::Zero(N);
  for (unsigned i = 0; i < N; i++)
    b[i] = vy[i];

  // execute according to 'mode'
  if (mode == LinearSolveMode::LRDefault)
    mode = LinearSolveMode::FastestNormal;

  if (mode == LinearSolveMode::AccurateSVD) {
    VectorXf res = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
    return std::make_pair(res[0], res[1]);
  }

  if (mode == LinearSolveMode::BalancedQR) {
    VectorXf res = A.colPivHouseholderQr().solve(b);
    return std::make_pair(res[0], res[1]);
  }

  // LinearSolveMode::FastestNormal
  VectorXf res = (A.transpose() * A).ldlt().solve(A.transpose() * b);
  return std::make_pair(res[0], res[1]);
}

// outputs, for A_1, A_2, ... on equation: y=A_1 f_1(x) + A_2 f_2(x) + ... + B
// considers single variable 'x' and multiple functions f(x)
std::vector<double>
leastSquaresRegression(const std::vector<double> &vx,
                       const std::vector<double> &vy,
                       std::vector<std::function<double(double)>> &vf,
                       LinearSolveMode mode = LinearSolveMode::LRDefault) {
  assert(vx.size() == vy.size());
  int N = vx.size();
  assert(N > 0);
  assert(vf.size() > 0);
  //
  // for every point i in (vx,vy), and j \in f
  // vy[i] = (forall j \in f) A_j . vf_j(vx[i]) + B
  //
  // Create system Ae = b, looking for solution ê
  //

  MatrixXf A = MatrixXf::Zero(N, vf.size() + 1);
  for (unsigned i = 0; i < N; i++) {
    for (unsigned j = 0; j < vf.size(); j++)
      A(i, j) = vf[j](vx[i]);
    A(i, vf.size()) = 1.0;
  }

  VectorXf b = VectorXf::Zero(N);
  for (unsigned i = 0; i < N; i++)
    b[i] = vy[i];

  // std::cout << "A=" << A << std::endl;
  // std::cout << "b=" << b << std::endl;

  // execute according to 'mode'
  if (mode == LinearSolveMode::LRDefault)
    mode = LinearSolveMode::FastestNormal;

  if (mode == LinearSolveMode::AccurateSVD) {
    VectorXf res = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
    std::vector<double> vres(res.size());
    for (unsigned i = 0; i < vres.size(); i++)
      vres[i] = res[i];
    return vres; // std::make_pair(res[0], res[1]);
  }

  if (mode == LinearSolveMode::BalancedQR) {
    VectorXf res = A.colPivHouseholderQr().solve(b);
    std::vector<double> vres(res.size());
    for (unsigned i = 0; i < vres.size(); i++)
      vres[i] = res[i];
    return vres; // std::make_pair(res[0], res[1]);
  }

  // LinearSolveMode::FastestNormal
  VectorXf res = (A.transpose() * A).ldlt().solve(A.transpose() * b);
  std::cout << "res=" << res << std::endl;
  std::vector<double> vres(res.size());
  for (unsigned i = 0; i < vres.size(); i++)
    vres[i] = res[i];
  return vres; // std::make_pair(res[0], res[1]);
}

double calcR2(const std::vector<double> &vx, const std::vector<double> &vy,
              std::function<double(double)> f) {
  // https://en.wikipedia.org/wiki/Coefficient_of_determination
  double ss_res = 0;
  double ss_tot = 0;
  double mean = 0;
  int N = vx.size();
  //
  assert(N == vy.size());
  //
  for (unsigned i = 0; i < N; i++)
    mean += vy[i];
  mean = mean / N;

  for (unsigned i = 0; i < N; i++) {
    double yest = f(vx[i]);
    double ydiff = (vy[i] - mean) * (vy[i] - mean);
    double ydiff2 = (yest - vy[i]) * (yest - vy[i]);
    ss_res += ydiff2;
    ss_tot += ydiff;

    /*
  std::cout << "y[" << i << "] ~~ " << yest << " y=" << vy[i]
            << " (mean=" << mean << ") diff2=" << ydiff2
            << " / diff=" << ydiff << "(R2 = " << (1 - (ydiff2 / ydiff))
            << ")" << std::endl;
  std::cout << "SS_res = " << ss_res << std::endl;
  std::cout << "SS_tot = " << ss_tot << std::endl;

  double r2_partial = 1 - (ss_res / ss_tot);
  */
    // std::cout << "r2 = " << r2 << std::endl;
  }
  return 1 - (ss_res / ss_tot);
}

/*
int main() {
  MatrixXf A = MatrixXf::Random(3, 2);
  cout << "Here is the matrix A:\n" << A << endl;
  VectorXf b = VectorXf::Random(3);
  cout << "Here is the right hand side b:\n" << b << endl;
  cout << "The least-squares solution is:\n"
       << A.bdcSvd(ComputeThinU | ComputeThinV).solve(b) << endl;

  // http://textbooks.math.gatech.edu/ila/least-squares.html
  // https://github.com/QBobWatson/ila
  //
  //
https://math.stackexchange.com/questions/2591061/exponential-least-squares-equation
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
  MatrixXf M2{{0, 1}, {1, 1}, {2, 1}};
  VectorXf b2{{6}, {0}, {0}};

  // The three methods discussed on this page are the SVD decomposition,
  // the QR decomposition and normal equations.
  // Of these, the SVD decomposition is generally the most accurate but the
  // slowest, normal equations is the fastest but least accurate,
  // and the QR decomposition is in between.

  cout << "The SVD linear least-squares solution is:\n"
       << M2.bdcSvd(ComputeThinU | ComputeThinV).solve(b2) << endl;
  // The SVD linear least-squares solution is: \n  -3 \n  5
  //
  cout << "The solution using the QR decomposition is:\n"
       << M2.colPivHouseholderQr().solve(b2) << endl;
  cout << "The solution using normal equations is:\n"
       << (M2.transpose() * M2).ldlt().solve(M2.transpose() * b2) << endl;
  // least squares solution is: y = -3x + 5

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

  MatrixXf M3{{1, -1, 1}, {1, 1, 1}, {4, 2, 1}, {9, 3, 1}};
  VectorXf b3{{1 / 2.0}, {-1}, {-1 / 2.0}, {2}};
  cout << "The solution using normal equations is:\n"
       << (M3.transpose() * M3).ldlt().solve(M3.transpose() * b3) << endl;
  cout << "B=53/88=" << 53 / 88.0 << " C=-379/440=" << -379 / 440.0
       << " D=-82=" << -82 << std::endl;

  // BASIC EXPONENTIAL: y = a*e^{-bx}


int N = 10;

// auto ones = MatrixXd::Ones(N, 1);

//
YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY
// MatrixXf = ... MatrixXd ...

MatrixXf M4 = MatrixXf::Zero(N, 2);
for (unsigned i = 0; i < N; i++) {
  // M4.topRows(i) = {i, 1};
  // M4.col(1) = ones.col(0);
  M4(i, 0) = i;
  M4(i, 1) = 1.0;
}

//{{0, 1}, {1, 1}, {2, 1}};
// VectorXf b2{{6}, {0}, {0}};

// https://gist.github.com/gocarlos/c91237b02c120c6319612e42fa196d77
//
https://math.stackexchange.com/questions/2591061/exponential-least-squares-equation

// example MI: https://mycurvefit.com

VectorXf v = VectorXf::Zero(N);
std::vector<float> vf = {8558, 5411, 2830, 2267, 760, 549, 249, 67, 47, 43};

// BASIC EXPONENTIAL: y = a*e^{-bx}
// ln(y) = ln(a) - bx
// ZY = ln(y)
// ZA = ln(a)
// ZB = -b
// ZY = ZB x + ZA

for (unsigned i = 0; i < N; i++)
  v[i] = ::log(vf[i]); // log natural

// VectorXf res = M4.bdcSvd(ComputeThinU | ComputeThinV).solve(v);
VectorXf res = (M4.transpose() * M4).ldlt().solve(M4.transpose() * v);

cout << "The solution using normal equations is:\n" << res << endl;
double ZB = res[0];
double ZA = res[1];
double realA = ::exp(ZA);
double realB = -ZB;

cout << "y = " << realA << " * e^{" << -realB << " * x }" << endl;
double ss_res = 0;
double ss_tot = 0;
double mean = 0;
// https://en.wikipedia.org/wiki/Coefficient_of_determination
for (unsigned i = 0; i < N; i++)
  mean += vf[i];
mean = mean / N;

for (unsigned i = 0; i < N; i++) {
  double yest = realA * ::exp(-realB * i);
  double ydiff = (vf[i] - mean) * (vf[i] - mean);
  double ydiff2 = (yest - vf[i]) * (yest - vf[i]);
  ss_res += ydiff2;
  ss_tot += ydiff;
  std::cout << "y[" << i << "] ~~ " << yest << " y=" << vf[i]
            << " (mean=" << mean << ") diff2=" << ydiff2 << " / diff=" << ydiff
            << "(R2 = " << (1 - (ydiff2 / ydiff)) << ")" << std::endl;
  std::cout << "SS_res = " << ss_res << std::endl;
  std::cout << "SS_tot = " << ss_tot << std::endl;
  double r2 = 1 - (ss_res / ss_tot);
  std::cout << "r2 = " << r2 << std::endl;
}

return 0;
}
*/

} // namespace optstats

#endif // OPTSTATS_LINREG_HPP