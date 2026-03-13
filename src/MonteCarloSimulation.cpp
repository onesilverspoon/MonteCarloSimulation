#define _USE_MATH_DEFINES

#include "../include/monte_carlo/options.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>

int main()
{
    omp_set_num_threads(omp_get_num_procs());
    int num_sims = 10000000;
    double S = 100.0;           // Spot price
    double K = 100.0;           // Strike price
    double r = 0.05;            // Risk-free rate
    double sigma = 0.2;         // Volatility
    double T = 1.0;             // Time to maturity (years)
    double sqrt_T = std::sqrt(T);

    double start_seq = omp_get_wtime();
    MonteCarloResult res = monte_carlo_sequential(num_sims, S, K, r, sigma, T,sqrt_T );
    double end_seq = omp_get_wtime();

    double start_par = omp_get_wtime();
    MonteCarloResult res_paral = monte_carlo_parallel(num_sims, S, K, r, sigma, T, sqrt_T);
    double end_par = omp_get_wtime();

    double t_seq = end_seq - start_seq;
    double t_par = end_par - start_par;

    int threads = 0;
#pragma omp parallel
    {
#pragma omp single
        threads = omp_get_num_threads();
    }
    double speedup = t_seq / t_par;
    double efficiency = (speedup / threads) * 100;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Call price        :" << res.call << std::endl;
    std::cout << "Call Var          :" << res.call_variance << std::endl;
    std::cout << "Call SE           :" << res.call_std_error << std::endl;
    std::cout << "Call CI           :[" << res.call_ci_low << ", " << res.call_ci_high << "]" << std::endl;
    std::cout << "Puts price        :" << res.put << std::endl;
    std::cout << "Puts Var          :" << res.put_variance << std::endl;
    std::cout << "Puts SE           :" << res.put_std_error << std::endl;
    std::cout << "Puts CI           :[" << res.put_ci_low << ", " << res.put_ci_high << "]" << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "CV Call           :" << res.call_cv << std::endl;
    std::cout << "CV Call Var       :" << res.call_cv_variance << std::endl;
    std::cout << "CV Call SE        :" << res.call_cv_std_error << std::endl;
    std::cout << "CV Puts           :" << res.put_cv << std::endl;
    std::cout << "CV Puts Var       :" << res.put_cv_variance << std::endl;
    std::cout << "CV Puts SE        :" << res.put_cv_std_error << std::endl;
    std::cout << "Var Reduction     :" << res.variance_reduction << std::endl;

 
    //black_schole benchmark
    double BS_call = black_scholes_call(S, K, r, sigma, T, sqrt_T);
    double abs_error_call = std::abs(res.call - BS_call);
    double rel_error_call = abs_error_call / BS_call;

    double BS_put = black_scholes_put(S, K, r, sigma, T, sqrt_T);
    double abs_error_put = std::abs(res.put - BS_put);
    double rel_error_put = abs_error_put / BS_put;


    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "Black-Scholes call:" << BS_call << std::endl;
    std::cout << "Absolute error    :" << abs_error_call << std::endl;
    std::cout << "Relative error    :" << rel_error_call << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "Black-Scholes puts:" << BS_put << std::endl;
    std::cout << "Absolute error    :" << abs_error_put << std::endl;
    std::cout << "Relative error    :" << rel_error_put << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "Greeks" << std::endl;
    std::cout << "Beta              :" << res.beta << std::endl;
    std::cout << "Call Delta        :" << res.call_delta << std::endl;
    std::cout << "Puts Delta        :" << res.put_delta << std::endl;
    std::cout << "Call Delta Adj.   :" << res.call_delta_cv << std::endl;
    std::cout << "Puts Delta Adj.   :" << res.put_delta_cv << std::endl;
    std::cout << "Gamma call        :" << res.call_gamma << std::endl;
    std::cout << "Gamma puts        :" << res.put_gamma << std::endl;
    std::cout << "Reallity chech gamma-----------------------------------------" << std::endl;
    double bs_gamma = black_scholes_gamma(S, K, r, sigma, T, sqrt_T);
    std::cout << "Black-scholes gamma:" << bs_gamma << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "Call Gamma Adj.   :" << res.call_gamma_cv << std::endl;
    std::cout << "Puts Gamma Adj.   :" << res.put_gamma_cv << std::endl;
    std::cout << "Call Vega Adj.    :" << res.call_vega_cv << std::endl;
    std::cout << "Puts Vega Adj.    :" << res.put_vega_cv << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "Parallel execution:" << std::endl;
    std::cout << "Call Paral price        :" << res_paral.call << std::endl;
    std::cout << "Call Paral Var          :" << res_paral.call_variance << std::endl;
    std::cout << "Call Paral SE           :" << res_paral.call_std_error << std::endl;
    std::cout << "Call Paral CI           :[" << res_paral.call_ci_low << ", " << res_paral.call_ci_high << "]" << std::endl;
    std::cout << "Puts Paral price        :" << res_paral.put << std::endl;
    std::cout << "Puts Paral Var          :" << res_paral.put_variance << std::endl;
    std::cout << "Puts Paral SE           :" << res_paral.put_std_error << std::endl;
    std::cout << "Puts Paral CI           :[" << res_paral.put_ci_low << ", " << res_paral.put_ci_high << "]" << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "CV Call Paral           :" << res_paral.call_cv << std::endl;
    std::cout << "CV Call Paral Var       :" << res_paral.call_cv_variance << std::endl;
    std::cout << "CV Call Paral SE        :" << res_paral.call_cv_std_error << std::endl;
    std::cout << "CV Puts Paral           :" << res_paral.put_cv << std::endl;
    std::cout << "CV Puts Paral Var       :" << res_paral.put_cv_variance << std::endl;
    std::cout << "CV Puts Paral SE        :" << res_paral.put_cv_std_error << std::endl;
    std::cout << "Var Paral Reduction     :" << res_paral.variance_reduction << std::endl;


    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "Greeks" << std::endl;
    std::cout << "Beta              :" << res_paral.beta << std::endl;
    std::cout << "Call Delta        :" << res_paral.call_delta << std::endl;
    std::cout << "Puts Delta        :" << res_paral.put_delta << std::endl;
    std::cout << "Call Delta Adj.   :" << res_paral.call_delta_cv << std::endl;
    std::cout << "Puts Delta Adj.   :" << res_paral.put_delta_cv << std::endl;
    std::cout << "Gamma call        :" << res_paral.call_gamma << std::endl;
    std::cout << "Gamma puts        :" << res_paral.put_gamma << std::endl;
    std::cout << "Reallity chech gamma-----------------------------------------" << std::endl;
    std::cout << "Black-scholes gamma:" << bs_gamma << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "Call Gamma Adj.   :" << res_paral.call_gamma_cv << std::endl;
    std::cout << "Puts Gamma Adj.   :" << res_paral.put_gamma_cv << std::endl;
    std::cout << "Call Vega Adj.    :" << res_paral.call_vega_cv << std::endl;
    std::cout << "Puts Vega Adj.    :" << res_paral.put_vega_cv << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "Speedup           :" << speedup << std::endl;
    std::cout << "Efficiency        :" << efficiency << "%" << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;


    return 0;
}