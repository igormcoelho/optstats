#ifndef OPTSTATS_NONLINEAR_GSL_HPP
#define OPTSTATS_NONLINEAR_GSL_HPP
//
// Copyleft: optstats - GPL license
// derived work from github.com/eleobert (GPL license)
//
// IMPORTANT: this file re-organizes the files from
// https://github.com/Eleobert/gsl-curve-fit
//
// this process was necessary for:
//
//(1) organize into a namespace (name was given as 'optstats', internally as
//'eleobert' to honor the original creator)
//
//(2) remove dependency from .cpp file (that breaks single .hpp format)
//

// =========================================
// ORIGINAL eleobert/gsl-curve-git' dependencies (including GNU GSL)
//
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_vector.h>

#include <cassert>
#include <functional>
#include <vector>
// =========================================

namespace optstats {

namespace eleobert {

// For information about non-linear least-squares fit with gsl
// see https://www.gnu.org/software/gsl/doc/html/nls.html

namespace detail {

auto internal_solve_system(gsl_vector *initial_params,
                           gsl_multifit_nlinear_fdf *fdf,
                           gsl_multifit_nlinear_parameters *params)
    -> std::vector<double> {
  // This specifies a trust region method
  const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;
  const size_t max_iter = 200;
  const double xtol = 1.0e-8;
  const double gtol = 1.0e-8;
  const double ftol = 1.0e-8;

  auto *work = gsl_multifit_nlinear_alloc(T, params, fdf->n, fdf->p);
  int info;

  // initialize solver
  gsl_multifit_nlinear_init(initial_params, fdf, work);
  // iterate until convergence
  gsl_multifit_nlinear_driver(max_iter, xtol, gtol, ftol, nullptr, nullptr,
                              &info, work);

  // result will be stored here
  gsl_vector *y = gsl_multifit_nlinear_position(work);
  auto result = std::vector<double>(initial_params->size);

  for (int i = 0; i < result.size(); i++) {
    result[i] = gsl_vector_get(y, i);
  }

  auto niter = gsl_multifit_nlinear_niter(work);
  auto nfev = fdf->nevalf;
  auto njev = fdf->nevaldf;
  auto naev = fdf->nevalfvv;

  // nfev - number of function evaluations
  // njev - number of Jacobian evaluations
  // naev - number of f_vv evaluations
  // logger::debug("curve fitted after ", niter, " iterations {nfev = ", nfev,
  // "} {njev = ", njev, "} {naev = ", naev, "}");

  gsl_multifit_nlinear_free(work);
  gsl_vector_free(initial_params);
  return result;
}

auto internal_make_gsl_vector_ptr(const std::vector<double> &vec)
    -> gsl_vector * {
  auto *result = gsl_vector_alloc(vec.size());
  int i = 0;
  for (const auto e : vec) {
    gsl_vector_set(result, i, e);
    i++;
  }
  return result;
}

} // namespace detail

template <class R, class... ARGS> struct function_ripper {
  static constexpr size_t n_args = sizeof...(ARGS);
};

/**
 * This function returns the number of parameters of a given function.This
 * overload is to be used specialy with lambdas.
 */
template <class R, class... ARGS>
auto constexpr n_params(std::function<R(ARGS...)>) {
  return function_ripper<R, ARGS...>();
}

/**
 * This function returns the number of parameters of a given function.
 */
template <class R, class... ARGS> auto constexpr n_params(R(ARGS...)) {
  return function_ripper<R, ARGS...>();
}

template <typename F, size_t... Is>
auto gen_tuple_impl(F func, std::index_sequence<Is...>) {
  return std::make_tuple(func(Is)...);
}

template <size_t N, typename F> auto gen_tuple(F func) {
  return gen_tuple_impl(func, std::make_index_sequence<N>{});
}

template <typename C1> struct fit_data {
  const std::vector<double> &t;
  const std::vector<double> &y;
  // the actual function to be fitted
  C1 f;
};

template <typename FitData, int n_params>
int internal_f(const gsl_vector *x, void *params, gsl_vector *f) {
  auto *d = static_cast<FitData *>(params);
  // Convert the parameter values from gsl_vector (in x) into std::tuple
  auto init_args = [x](int index) { return gsl_vector_get(x, index); };
  auto parameters = gen_tuple<n_params>(init_args);

  // Calculate the error for each...
  for (size_t i = 0; i < d->t.size(); ++i) {
    double ti = d->t[i];
    double yi = d->y[i];
    auto func = [ti, &d](auto... xs) {
      // call the actual function to be fitted
      return d->f(ti, xs...);
    };
    auto y = std::apply(func, parameters);
    gsl_vector_set(f, i, yi - y);
  }
  return GSL_SUCCESS;
}

using func_f_type = int (*)(const gsl_vector *, void *, gsl_vector *);
using func_df_type = int (*)(const gsl_vector *, void *, gsl_matrix *);
using func_fvv_type = int (*)(const gsl_vector *, const gsl_vector *, void *,
                              gsl_vector *);

auto internal_make_gsl_vector_ptr(const std::vector<double> &vec)
    -> gsl_vector *;

auto internal_solve_system(gsl_vector *initial_params,
                           gsl_multifit_nlinear_fdf *fdf,
                           gsl_multifit_nlinear_parameters *params)
    -> std::vector<double>;

template <typename C1>
auto curve_fit_impl(func_f_type f, func_df_type df, func_fvv_type fvv,
                    gsl_vector *initial_params, fit_data<C1> &fd)
    -> std::vector<double> {
  assert(fd.t.size() == fd.y.size());

  auto fdf = gsl_multifit_nlinear_fdf();
  auto fdf_params = gsl_multifit_nlinear_default_parameters();

  fdf.f = f;
  fdf.df = df;
  fdf.fvv = fvv;
  fdf.n = fd.t.size();
  fdf.p = initial_params->size;
  fdf.params = &fd;

  // "This selects the Levenberg-Marquardt algorithm with geodesic
  // acceleration."
  fdf_params.trs = gsl_multifit_nlinear_trs_lmaccel;
  return detail::internal_solve_system(initial_params, &fdf, &fdf_params);
}

/**
 * Performs a non-linear least-squares fit.
 *
 * @param f a function of type double (double x, double c1, double c2, ...,
 * double cn) where  c1, ..., cn are the coefficients to be fitted.
 * @param initial_params intial guess for the parameters. The size of the array
 * must to be equal to the number of coefficients to be fitted.
 * @param x the idependent data.
 * @param y the dependent data, must to have the same size as x.
 * @return std::vector<double> with the computed coefficients
 */
template <typename Callable>
auto curve_fit(Callable f, const std::vector<double> &initial_params,
               const std::vector<double> &x, const std::vector<double> &y)
    -> std::vector<double> {
  // We can't pass lambdas without convert to std::function.
  constexpr auto n = decltype(n_params(std::function(f)))::n_args - 1;
  assert(initial_params.size() == n);

  auto params = detail::internal_make_gsl_vector_ptr(initial_params);
  auto fd = fit_data<Callable>{x, y, f};
  return curve_fit_impl(internal_f<decltype(fd), n>, nullptr, nullptr, params,
                        fd);
}

// ===========================================
// now stats '.cpp' file as namespace 'detail'
// ===========================================

//#include "curve_fit.hpp"

} // namespace eleobert

template <typename Callable>
std::vector<double> leastSquaresNonLinearRegression(
    const std::vector<double> &xs, const std::vector<double> &ys,
    const std::vector<double> &initialParams, Callable func) {
  std::vector<double> res = eleobert::curve_fit(func, initialParams, xs, ys);
  return res;
}

} // namespace optstats

//

#endif // OPTSTATS_NONLINEAR_GSL_HPP