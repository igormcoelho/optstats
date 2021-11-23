#ifndef OPTSTATS_TTEST_HPP
#define OPTSTATS_TTEST_HPP

#include <bits/stdc++.h>

//
#define STATS_ENABLE_STDVEC_WRAPPERS
#include <statslib/include/stats.hpp>

namespace optstats {

// Function to find mean.
double mean(const std::vector<double> &v) {
  double sum = 0;
  for (int i = 0; i < v.size(); i++)
    sum = sum + v[i];
  return sum / v.size();
}

double stdev(const std::vector<double> &v) {
  double sum = 0;
  int n = v.size();
  for (int i = 0; i < n; i++)
    sum = sum + (v[i] - mean(v)) * (v[i] - mean(v));

  // Bessel's Correction: use n-1 instead of n on sampling, for unbiased
  // estimator: https://en.wikipedia.org/wiki/Bessel%27s_correction
  return ::sqrt(sum / (n - 1));
}

// Independent two-sample t-test:
// https://en.wikipedia.org/wiki/Student%27s_t-test#Independent_(unpaired)_samples
// returns pair (tvalue, dof) with t test value and degrees of freedom
auto getIndependentTwoSampleTTest(std::vector<double> v1,
                                  std::vector<double> v2,
                                  bool equalVariances = false) {
  double n1 = v1.size();
  double n2 = v2.size();
  assert(n1 > 0);
  assert(n2 > 0);
  auto mean1 = mean(v1);
  auto mean2 = mean(v2);
  auto sd1 = stdev(v1);
  auto sd2 = stdev(v2);

  double t_test = 0.0;
  double dof = 0.0; // degrees of freedom

  // three cases
  if ((v1.size() == v2.size()) && equalVariances) {
    // (Student's t test)
    // first case: Equal sample sizes and variance
    //
    // compute pooled standard deviation
    double s_p = ::sqrt(((sd1 * sd1) + (sd2 * sd2)) / 2);
    t_test = (mean1 - mean2) / (s_p * ::sqrt(2 / n1));
    dof = 2 * n1 - 2;
  } else if (equalVariances) {
    // (Student's t test)
    // second case: Equal or unequal sample sizes, similar variances
    //
    // compute pooled standard deviation
    double sp_over = (n1 - 1) * (sd1 * sd1) + (n2 - 1) * (sd2 * sd2);
    double s_p = ::sqrt(sp_over / (n1 + n2 - 2));
    t_test = (mean1 - mean2) / (s_p * ::sqrt((1 / n1) + (1 / n2)));
    dof = n1 + n2 - 2;
  } else {
    assert(!equalVariances);
    // (Welch's t-test)
    // third case: Equal or unequal sample sizes, unequal variances
    double s_delta = ::sqrt((sd1 * sd1) / n1 + (sd2 * sd2) / n2);
    t_test = (mean1 - mean2) / s_delta;
    //
    // Welch–Satterthwaite equation (for degrees of freedom)
    double s1_2_div_n1 = (sd1 * sd1) / n1;
    double s2_2_div_n2 = (sd2 * sd2) / n2;
    dof = (s1_2_div_n1 + s2_2_div_n2) * (s1_2_div_n1 + s2_2_div_n2) /
          (s1_2_div_n1 * s1_2_div_n1 / (n1 - 1) +
           s2_2_div_n2 * s2_2_div_n2 / (n2 - 1));
  }
  return std::make_tuple(t_test, dof);
}

double pIndependentTwoSampleTTest(std::vector<double> a1,
                                  std::vector<double> a2, bool twoSided = true,
                                  bool equalVariances = false) {

  auto [ttest, dof] =
      optstats::getIndependentTwoSampleTTest(a1, a2, equalVariances);
  double prob = stats::pt(ttest, dof, false);
  double p = 1 - prob;
  if (twoSided)
    p = 2 * p;
  return p;
}

} // namespace optstats

#endif // OPTSTATS_TTEST_HPP