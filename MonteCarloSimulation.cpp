
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <omp.h>

struct Monte_carlo_results {
    double call;
    double put;
    double varience;
    double ci_low;
    double ci_high;
    double str_error;
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


void monte_carlo_call_put_price(int num_sims, double S, double K, double r, double sigma, double T, std::mt19937& gen, double& call_seq, double& put_seq) {
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

    //beta
    double Y_sum = 0.0;
    double Y_sq_sum = 0.0;
    double cross_sum = 0.0;


    for (int i = 0; i < half_sims; i++) {
        double gauss_bm = d(gen);
        double gauss_bm_antithetic = -gauss_bm;
        double ST1 = S * std::exp(Drift + Vol_sqrt_T * gauss_bm);
        double ST2 = S * std::exp(Drift + Vol_sqrt_T * gauss_bm_antithetic);

        double call_payoff1 = std::max(ST1 - K, 0.0);
        double call_payoff2 = std::max(ST2 - K, 0.0);
        double put_payoff1 = std::max(K - ST1, 0.0);
        double put_payoff2 = std::max(K - ST2, 0.0);
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
    }
    //mean calculation
    double call_mean = call_sum / num_sims;
    double put_mean = put_sum / num_sims;
    //price estimation
    call_seq = call_mean * Discount;
    put_seq = put_mean * Discount;

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

    // Statistics printing
    std::cout << "Call price        :" << call_seq << std::endl;
    std::cout << "Standard error    :" << call_se << std::endl;
    std::cout << "95% CI            :[" << call_ci_low << ", " << call_ci_high << " ]" << std::endl;
    std::cout << "Put price         :" << put_seq << std::endl;
    std::cout << "Standard error    :" << put_se << std::endl;
    std::cout << "95% CI            :[" << put_ci_low << ", " << put_ci_high << " ]" << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;
    //beta calculation
    double X_mean = call_seq;           // call seq already discounted
    double Y_mean = Y_sum / num_sims;   //discounted ST mean
    double cov_XY = (cross_sum / num_sims) - X_mean * Y_mean;
    double var_Y = (Y_sq_sum / num_sims) - Y_mean * Y_mean;
    double beta = cov_XY / var_Y;
    
    //control variate
    double expected_Y = S ;                                     // E[e^{-rT} S_T] = S , risk-neutral measure.
    double call_cv = X_mean + beta * (expected_Y - Y_mean);     // new price
    std::cout << "CV new price              :" << call_cv << std::endl;

    //CV variace
    double var_CV = call_var_MC - beta * beta * var_Y;
    double se_cv = std::sqrt(var_CV / num_sims);
    double reduction = 1.0 - (var_CV / call_var_MC);

    std::cout << "CV Standard error         :" << se_cv << std::endl;
    std::cout << "MC var                    :" << call_var_MC << std::endl;
    std::cout << "Varience Reduction        :" << reduction << std::endl;

    Monte_carlo_results result;
    result.call = call_seq;
    result.put = put_seq;
    result.varience = call_var_MC;
    result.str_error = call_se;


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

    double call_seq, put_seq;
    double call_paral, put_paral;

    double start_seq = omp_get_wtime();
    monte_carlo_call_put_price(num_sims, S, K, r, sigma, T, gen, call_seq, put_seq);
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

 
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "Call Parallel: " << call_paral << "\nPut Parallel: " << put_paral << std::endl;
    std::cout << "Speedup: " << speedup << "x\nEfficiency: " << efficiency << "%" << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;
    //black_schole benchmark
    double BS_call = black_scholes_call(S, K, r, sigma, T);
    double abs_error_call = std::abs(call_seq - BS_call);
    double rel_error_call = abs_error_call / BS_call;

    double BS_put = black_scholes_put(S, K, r, sigma, T);
    double abs_error_put = std::abs(put_seq - BS_put);
    double rel_error_put = abs_error_put / BS_put;

    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "Black-Scholes call:" << BS_call << std::endl;
    std::cout << "Absolute error    :" << abs_error_call << std::endl;
    std::cout << "Relative error    :" << rel_error_call << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "Black-Scholes puts:" << BS_put << std::endl;
    std::cout << "Absolute error    :" << abs_error_put << std::endl;
    std::cout << "Relative error    :" << rel_error_put << std::endl;

    return 0;
}