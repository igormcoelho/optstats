
// http://eigen.tuxfamily.org/dox-devel/group__LeastSquares.html

#include <Eigen/Dense>
#include <iostream>

#include <optstats/optstats.hpp>

using namespace std;
using namespace Eigen;

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
  /*
1 8558
2 5411
3 2830
4 2267
5 760
6 549
7 249
8 67
9 47
10 43
*/

  int N = 10;

  // auto ones = MatrixXd::Ones(N, 1);

  // YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY
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
  // https://math.stackexchange.com/questions/2591061/exponential-least-squares-equation

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
              << " (mean=" << mean << ") diff2=" << ydiff2
              << " / diff=" << ydiff << "(R2 = " << (1 - (ydiff2 / ydiff))
              << ")" << std::endl;
    std::cout << "SS_res = " << ss_res << std::endl;
    std::cout << "SS_tot = " << ss_tot << std::endl;
    double r2 = 1 - (ss_res / ss_tot);
    std::cout << "r2 = " << r2 << std::endl;
  }

  return 0;
}
