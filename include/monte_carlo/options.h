#pragma once
#ifndef MONTE_CARLO_OPTIONS_H
#define MONTE_CARLO_OPTIONS_H

#include <stdexcept>
#include <string>

struct MonteCarloResult {
    double call;
    double put;

    //Monte-Carlo stats
    double call_variance;
    double put_variance;
    double call_std_error;
    double put_std_error;

    double call_ci_low;
    double call_ci_high;
    double put_ci_low;
    double put_ci_high;

    //Control Variance
    double call_cv;
    double put_cv;
    double call_cv_variance;
    double put_cv_variance;
    double call_cv_std_error;
    double put_cv_std_error;
    double variance_reduction;

    //Greeks
    double beta;
    double call_delta;
    double put_delta;
    double call_gamma;
    double put_gamma;
    double call_vega;
    double put_vega;

    //Greeks CV adjusted
    double call_delta_cv;
    double put_delta_cv;
    double call_gamma_cv;
    double put_gamma_cv;
    double call_vega_cv;
    double put_vega_cv;


};
void validate_option_parameters(int num_sims, double S, double K, double r, double sigma, double T);

MonteCarloResult monte_carlo_sequential(int num_sims, double S, double K, double r, double sigma, double T, double sqrt_T);
MonteCarloResult monte_carlo_parallel(int num_sims, double S, double K, double r, double sigma, double T, double sqrt_T);

double black_scholes_call(double S, double K, double r, double sigma, double T, double sqrt_T);
double black_scholes_put(double S, double K, double r, double sigma, double T, double sqrt_T);
double black_scholes_gamma(double S, double K, double r, double sigma, double T, double sqrt_T);

#endif // MONTE_CARLO_OPTIONS_H