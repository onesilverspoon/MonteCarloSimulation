
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <omp.h>

struct Monte_carlo_results {
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

    
};

double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2));
}

double black_scholes_call(double S, double K, double r, double sigma, double T) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}
double black_scholes_put(double S, double K, double r, double sigma, double T) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return  K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
}


Monte_carlo_results monte_carlo_call_put_price(int num_sims, double S, double K, double r, double sigma, double T, std::mt19937& gen) {
    double Drift = T * (r - 0.5 * sigma * sigma);
    double Vol_sqrt_T = sigma * std::sqrt(T);
    double Discount = std::exp(-r * T);
    int half_sims = num_sims / 2;

    std::normal_distribution<double> d(0.0, 1.0);

    //stats
    double call_sum = 0.0;
    double put_sum = 0.0;
    double call_sq_sum = 0.0;
    double put_sq_sum = 0.0;

    //beta,delta,gamma,vega, CV accumulators
    double Y_sum = 0.0;
    double Y_sq_sum = 0.0;
    double cross_sum = 0.0;
    double delta_call_sum = 0.0;
    double delta_put_sum = 0.0;
    double gamma_call_sum = 0.0;
    double gamma_put_sum = 0.0;
    double vega_call_sum = 0.0;
    double vega_put_sum = 0.0;
    double cross_sum_put = 0.0;


    for (int i = 0; i < half_sims; i++) {
        double gauss_bm = d(gen);
        double gauss_bm_antithetic = -gauss_bm;
        double ST1 = S * std::exp(Drift + Vol_sqrt_T * gauss_bm);
        double ST2 = S * std::exp(Drift + Vol_sqrt_T * gauss_bm_antithetic);
        double Z1 = gauss_bm;
        double Z2 = gauss_bm_antithetic;

        double call_payoff1 = std::max(ST1 - K, 0.0);
        double call_payoff2 = std::max(ST2 - K, 0.0);
        double put_payoff1 = std::max(K - ST1, 0.0);
        double put_payoff2 = std::max(K - ST2, 0.0);

        //Delta
        if (ST1 > K)delta_call_sum += ST1 / S;
        if (ST2 > K)delta_call_sum += ST2 / S;
        if (ST1 < K)delta_put_sum -= ST1 / S;
        if (ST2 < K)delta_put_sum -= ST2 / S;

        //Gamma wrong impementation dissabled for now
        /*
        double common1 = (Z1 / (S * sigma * std::sqrt(T)));
        double common2 = (Z2 / (S * sigma * std::sqrt(T)));
        gamma_call_sum += std::max(ST1 - K, 0.0) * common1;
        gamma_call_sum += std::max(ST2 - K, 0.0) * common2;
        gamma_put_sum += std::max(K - ST1, 0.0) * common1;
        gamma_put_sum += std::max(K - ST2, 0.0) * common2;
        */

        //Vega
        vega_call_sum += std::max(ST1 - K, 0.0) * (Z1 * std::sqrt(T));
        vega_call_sum += std::max(ST2 - K, 0.0) * (Z2 * std::sqrt(T));
        vega_put_sum += std::max(K - ST1, 0.0) * (Z1 * std::sqrt(T));
        vega_put_sum += std::max(K - ST2, 0.0) * (Z2 * std::sqrt(T));

        //call stats
        call_sum += call_payoff1 + call_payoff2;
        call_sq_sum += call_payoff1 * call_payoff1 + call_payoff2 * call_payoff2;

        //put stats
        put_sum += put_payoff1 + put_payoff2;
        put_sq_sum += put_payoff1 * put_payoff1 + put_payoff2 * put_payoff2;

        //beta
        double discounted_call1 = call_payoff1 * Discount;
        double discounted_call2 = call_payoff2 * Discount;
        double discounted_ST1 = ST1 * Discount;
        double discounted_ST2 = ST2 * Discount;
        Y_sum += discounted_ST1 + discounted_ST2;
        Y_sq_sum += discounted_ST1 * discounted_ST1 + discounted_ST2 * discounted_ST2;
        cross_sum += discounted_call1 * discounted_ST1 + discounted_call2 * discounted_ST2;

        double discounted_put1 = put_payoff1 * Discount;
        double discounted_put2 = put_payoff2 * Discount;
        cross_sum_put += discounted_put1 * discounted_ST1 + discounted_put2 * discounted_ST2;
    }
    //mean calculation
    double call_mean = call_sum / num_sims;
    double put_mean = put_sum / num_sims;
    //price estimation
    double call_seq = call_mean * Discount;
    double put_seq = put_mean * Discount;

    //var calculation (undiscounted)
    double call_var = (call_sq_sum / num_sims) - call_mean * call_mean;
    double put_var = (put_sq_sum / num_sims) - put_mean * put_mean;
    //var Monte Carlo discounted
    double call_var_MC = call_var * (Discount * Discount);
    double put_var_MC = put_var * (Discount * Discount);

    //standard error
    double call_se = std::sqrt(call_var_MC / num_sims);
    double put_se = std::sqrt(put_var_MC / num_sims);

    // 1.96= Z-value for 95% Confidence
    double call_ci_low = call_seq - 1.96 * call_se;
    double call_ci_high = call_seq + 1.96 * call_se;
    double put_ci_low = put_seq - 1.96 * put_se;
    double put_ci_high = put_seq + 1.96 * put_se;

    //beta calculation
    double X_mean = (call_sum / num_sims) * Discount;           // not already discounted
    double Y_mean = Y_sum / num_sims;                           // discounted ST mean
    double cov_XY = (cross_sum / num_sims) - X_mean * Y_mean;   // already discounted
    double var_Y = (Y_sq_sum / num_sims) - Y_mean * Y_mean;
    double beta = (var_Y > 1e-12) ? (cov_XY / var_Y) : 0.0;     // avoid div by 0
    double call_delta = Discount * (delta_call_sum / num_sims);
    double put_delta = Discount * (delta_put_sum / num_sims);

    double call_gamma = Discount * (gamma_call_sum / num_sims);
    double put_gamma = Discount * (gamma_put_sum / num_sims);

    double call_vega = Discount * (vega_call_sum / num_sims);
    double put_vega = Discount * (vega_put_sum / num_sims);
    
    //control variate
    double expected_Y = S ;                                     // E[e^{-rT} S_T] = S , risk-neutral measure.
    double call_cv = X_mean + beta * (expected_Y - Y_mean);     // new price

    //CV variace
    double var_cv = call_var_MC + beta * beta * var_Y - 2.0 * beta * cov_XY;
    double se_cv = std::sqrt(var_cv / num_sims);
    double reduction = 1.0 - (var_cv / call_var_MC);
    double X_put_mean = (put_sum / num_sims) * Discount;
    double cov_put_Y = (cross_sum_put / num_sims) - X_put_mean * Y_mean;
    double beta_put = (var_Y > 1e-12) ? cov_put_Y / var_Y : 0.0;
    double put_cv = X_put_mean + beta_put * (expected_Y - Y_mean);
    double var_put_CV = put_var_MC + beta_put * beta_put * var_Y - 2.0 * beta_put * cov_put_Y;
    double put_se_cv = std::sqrt(var_put_CV / num_sims);

    

    Monte_carlo_results result;
    result.call = call_seq;
    result.put = put_seq;
    result.call_variance = call_var_MC;
    result.put_variance = put_var_MC;
    result.call_std_error = call_se;
    result.put_std_error = put_se;
    result.call_ci_low = call_ci_low;
    result.call_ci_high = call_ci_high;
    result.put_ci_low = put_ci_low;
    result.put_ci_high = put_ci_high;
    result.call_cv = call_cv;
    result.call_cv_variance = var_cv;
    result.call_cv_std_error = se_cv;
    result.variance_reduction = reduction;
    result.beta = beta;
    result.call_delta = call_delta;
    result.put_delta = put_delta;
    result.call_gamma = call_gamma;
    result.put_gamma = put_gamma;
    result.call_vega = call_vega;
    result.put_vega = put_vega;
    result.put_cv = put_cv;
    result.put_cv_variance = var_put_CV;
    result.put_cv_std_error = put_se_cv;

    return result;

}

void monte_carlo_call_put_price_paral(int num_sims, double S, double K, double r, double sigma, double T, double& call_paral, double& put_paral) {
    double Drift = T * (r - 0.5 * sigma * sigma);
    double Vol_sqrt_T = sigma * std::sqrt(T);
    double Discount = std::exp(-r * T);
    double call_sum = 0.0;
    double put_sum = 0.0;
    int half_sims = num_sims / 2;

#pragma omp parallel 
    {
        //gen random number from normal distribution N(0,1) with deterministic per-thread seeding using seed_seq    
        std::seed_seq seq{ 1234, omp_get_thread_num() };
        std::mt19937  gen(seq);
        std::normal_distribution<double> d(0.0, 1.0);
        double Op_price_cur1 = 0.0;
        double Op_price_cur2 = 0.0;

#pragma omp for reduction(+:call_sum,put_sum)
        for (int i = 0; i < half_sims; i++) {
            double gauss_bm = d(gen);
            double gauss_bm_antithetic = -gauss_bm;
            Op_price_cur1 = S * std::exp(Drift + Vol_sqrt_T * gauss_bm);
            Op_price_cur2 = S * std::exp(Drift + Vol_sqrt_T * gauss_bm_antithetic);
            call_sum += std::max(Op_price_cur1 - K, 0.0);
            call_sum += std::max(Op_price_cur2 - K, 0.0);
            put_sum += std::max(K - Op_price_cur1, 0.0);
            put_sum += std::max(K - Op_price_cur2, 0.0);
        }
    }
    call_paral = (call_sum / static_cast<double>(num_sims)) * Discount;
    put_paral = (put_sum / static_cast<double>(num_sims)) * Discount;

}

int main()
{
    int num_sims = 10000000;
    omp_set_num_threads(omp_get_num_procs());
    double S = 100.0;
    double K = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    double T = 1.0; //in years
    std::mt19937 gen(42);

    //double call_seq, put_seq;
    double call_paral, put_paral;

    double start_seq = omp_get_wtime();
    Monte_carlo_results res = monte_carlo_call_put_price(num_sims, S, K, r, sigma, T, gen);
    double end_seq = omp_get_wtime();

    double start_par = omp_get_wtime();
    monte_carlo_call_put_price_paral(num_sims, S, K, r, sigma, T, call_paral, put_paral);
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
    std::cout << "Var Red           :" << res.variance_reduction << std::endl;

 
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "Call Parallel     :" << call_paral << std::endl;
    std::cout << "Puts Parallel     :" << put_paral << std::endl;
    std::cout << "Speedup           :" << speedup << std::endl;
    std::cout << "Efficiency        :" << efficiency << "%" << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;
    //black_schole benchmark
    double BS_call = black_scholes_call(S, K, r, sigma, T);
    double abs_error_call = std::abs(res.call - BS_call);
    double rel_error_call = abs_error_call / BS_call;

    double BS_put = black_scholes_put(S, K, r, sigma, T);
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

    return 0;
}