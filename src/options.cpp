#define _USE_MATH_DEFINES

#include "monte_carlo/options.h"
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <omp.h>

double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2));
}

double black_scholes_call(double S, double K, double r, double sigma, double T, double sqrt_T) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T);
    double d2 = d1 - sigma * sqrt_T;
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

double black_scholes_put(double S, double K, double r, double sigma, double T, double sqrt_T) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T);
    double d2 = d1 - sigma * sqrt_T;
    return  K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
}

double normal_pdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2 * M_PI);
}

double black_scholes_gamma(double S, double K, double r, double sigma, double T, double sqrt_T) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T);
    return normal_pdf(d1) / (S * sigma * sqrt_T);
}

void validate_option_parameters(int num_sims, double S, double K, double r, double sigma, double T) {
    // Validate all input parameters for Monte Carlo simulation

    if (num_sims <= 0) {
        throw std::invalid_argument("num_sims must be > 0");
    }
    if (S <= 0.0) {
        throw std::invalid_argument("Spot price S must be > 0");
    }
    if (K <= 0.0) {
        throw std::invalid_argument("Strike price K must be > 0");
    }
    if (r < -1.0 || r > 1.0 ) {
        throw std::invalid_argument("Time to maturity r must be in the range {-1,1}");
    }
    if (sigma < 0.0) {
        throw std::invalid_argument("Volatility sigma must be >= 0");
    }
    if (T < 0.0) {
        throw std::invalid_argument("Time to maturity T must be >= 0");
    }
}

MonteCarloResult monte_carlo_sequential(int num_sims, double S, double K, double r, double sigma, double T, double sqrt_T) {
    /**
     Monte Carlo option pricing - Sequential version
    
     Algorithm overview:
     1. Generate random numbers in antithetic pairs for variance reduction
     2. Simulate stock price paths using geometric Brownian motion
     3. Calculate option payoffs at maturity
     4. Apply control variate technique to reduce variance further
     5. Estimate Greeks using likelihood ratio method
     6. Compute confidence intervals and variance reduction metrics
     */

     // Input validation
    validate_option_parameters(num_sims, S, K, r, sigma, T);

    // Pre-compute constants to avoid repeated calculations
    
    double Drift = T * (r - 0.5 * sigma * sigma);      // Drift term: (r - σ²/2)T
    double Vol_sqrt_T = sigma * sqrt_T;                // Volatility term: σ√T
    double Discount = std::exp(-r * T);                // Discount factor: e^(-rT)

    double base = S * std::exp(Drift);                  // S * e^(drift)
    int half_sims = num_sims / 2;                       // Number of antithetic pairs

    // Random number generator
    std::mt19937 gen(42);
    std::normal_distribution<double> d(0.0, 1.0);

    // Accumulators for basic statistics
    double call_sum = 0.0, call_sq_sum = 0.0;
    double put_sum = 0.0, put_sq_sum = 0.0;

    // Greeks accumulators
    double delta_call_sum = 0.0, delta_put_sum = 0.0;
    double gamma_call_sum = 0.0, gamma_put_sum = 0.0;
    double vega_call_sum = 0.0, vega_put_sum = 0.0;

    // Control variate accumulators
    double X_sum = 0.0, X_put_sum = 0.0;               // Discounted payoff sums
    double Y_sum = 0.0, Y_sq_sum = 0.0;                // Discounted spot sums
    double cross_sum = 0.0, cross_sum_put = 0.0;       // Cross-product terms
    double cross_sum_gamma = 0.0, cross_sum_gamma_put = 0.0;
    double cross_sum_vega = 0.0, cross_sum_vega_put = 0.0;
    double stock_vega_sum = 0.0;

    // Main Monte Carlo loop - antithetic pairs
    for (int i = 0; i < half_sims; i++) {
        // Generate standard normal and its negative (antithetic pair)
        double Z = d(gen);
        double Z_anti = -Z;

        // Compute exp(σ√T * Z) for stock price calculation
        double exp_term = std::exp(Vol_sqrt_T * Z);

        // Stock prices at maturity using antithetic pair
        // S_T(Z) = base * exp(σ√T * Z)
        // S_T(-Z) = base * exp(-σ√T * Z) = base / exp_term
        double ST1 = base * exp_term;
        double ST2 = base / exp_term;

        // Option payoffs at maturity
        double call_payoff1 = std::max(ST1 - K, 0.0);
        double call_payoff2 = std::max(ST2 - K, 0.0);
        double put_payoff1 = std::max(K - ST1, 0.0);
        double put_payoff2 = std::max(K - ST2, 0.0);

        
        // Delta Calculation (Likelihood Ratio Method)
        
        // Delta approximation: payoff / S when in-the-money
        if (ST1 > K) delta_call_sum += ST1 / S;
        if (ST2 > K) delta_call_sum += ST2 / S;
        if (ST1 < K) delta_put_sum += (-ST1 / S);
        if (ST2 < K) delta_put_sum += (-ST2 / S);

        
        // Gamma Calculation (Likelihood Ratio Method)
        
        // Gamma = d²C/dS² using likelihood ratio method
        // Weight: (Z² - 1) / (S² * σ² * T)
        double L1 = (std::log(ST1 / S) - Drift) / Vol_sqrt_T;
        double L2 = (std::log(ST2 / S) - Drift) / Vol_sqrt_T;
        double gamma_weight1 = (Z * Z - 1.0) / (S * S * sigma * sigma * T);
        double gamma_weight2 = (Z_anti * Z_anti - 1.0) / (S * S * sigma * sigma * T);

        gamma_call_sum += call_payoff1 * gamma_weight1 + call_payoff2 * gamma_weight2;
        gamma_put_sum += put_payoff1 * gamma_weight1 + put_payoff2 * gamma_weight2;

        // Vega Calculation (Likelihood Ratio Method)
        
        // Vega = dC/dσ using likelihood ratio method
        double vega_weight1_call = call_payoff1 * ((L1 * L1 - 1.0) / sigma - L1 * sqrt_T);
        double vega_weight2_call = call_payoff2 * ((L2 * L2 - 1.0) / sigma - L2 * sqrt_T);
        double vega_weight1_put = put_payoff1 * (L1 * L1 / sigma - L1 * sqrt_T - 1.0 / sigma);
        double vega_weight2_put = put_payoff2 * (L2 * L2 / sigma - L2 * sqrt_T - 1.0 / sigma);

        vega_call_sum += vega_weight1_call + vega_weight2_call;
        vega_put_sum += vega_weight1_put + vega_weight2_put;

        
        // Basic Statistics
        
        call_sum += call_payoff1 + call_payoff2;
        call_sq_sum += call_payoff1 * call_payoff1 + call_payoff2 * call_payoff2;
        put_sum += put_payoff1 + put_payoff2;
        put_sq_sum += put_payoff1 * put_payoff1 + put_payoff2 * put_payoff2;

        
        // Control Variate: Accumulate discounted values
        
        double discounted_call1 = call_payoff1 * Discount;
        double discounted_call2 = call_payoff2 * Discount;
        double discounted_ST1 = ST1 * Discount;
        double discounted_ST2 = ST2 * Discount;

        X_sum += discounted_call1 + discounted_call2;
        Y_sum += discounted_ST1 + discounted_ST2;
        Y_sq_sum += discounted_ST1 * discounted_ST1 + discounted_ST2 * discounted_ST2;
        cross_sum += discounted_call1 * discounted_ST1 + discounted_call2 * discounted_ST2;

        // Put control variate terms
        double discounted_put1 = put_payoff1 * Discount;
        double discounted_put2 = put_payoff2 * Discount;
        cross_sum_put += discounted_put1 * discounted_ST1 + discounted_put2 * discounted_ST2;
        X_put_sum += discounted_put1 + discounted_put2;

        // Gamma and Vega control variate terms
        double discounted_gamma1 = call_payoff1 * gamma_weight1 * Discount;
        double discounted_gamma2 = call_payoff2 * gamma_weight2 * Discount;
        cross_sum_gamma += discounted_gamma1 * discounted_ST1 + discounted_gamma2 * discounted_ST2;

        double discounted_gamma_put1 = put_payoff1 * gamma_weight1 * Discount;
        double discounted_gamma_put2 = put_payoff2 * gamma_weight2 * Discount;
        cross_sum_gamma_put += discounted_gamma_put1 * discounted_ST1 + discounted_gamma_put2 * discounted_ST2;

        double discounted_vega1 = call_payoff1 * vega_weight1_call * Discount;
        double discounted_vega2 = call_payoff2 * vega_weight2_call * Discount;
        cross_sum_vega += discounted_vega1 * discounted_ST1 + discounted_vega2 * discounted_ST2;

        double discounted_vega_put1 = put_payoff1 * vega_weight1_put * Discount;
        double discounted_vega_put2 = put_payoff2 * vega_weight2_put * Discount;
        cross_sum_vega_put += discounted_vega_put1 * discounted_ST1 + discounted_vega_put2 * discounted_ST2;

        // Stock vega (for vega adjustment)
        double stock_vega1 = ST1 * (Z * sqrt_T - sigma * T);
        double stock_vega2 = ST2 * (Z_anti * sqrt_T - sigma * T);
        stock_vega_sum += stock_vega1 + stock_vega2;
    }

    
    // Compute Statistics

    // Basic price estimates
    double call_mean = call_sum / num_sims;
    double put_mean = put_sum / num_sims;
    double call_price = call_mean * Discount;
    double put_price = put_mean * Discount;

    // Variance: Var = E[X²] - E[X]² = (ΣX² - N*E[X]²) / (N-1)
    double call_var = (call_sq_sum - num_sims * call_mean * call_mean) / (num_sims - 1);
    double put_var = (put_sq_sum - num_sims * put_mean * put_mean) / (num_sims - 1);

    // Discounted variance (apply discount factor squared)
    double call_var_MC = call_var * (Discount * Discount);
    double put_var_MC = put_var * (Discount * Discount);

    // Standard error: SE = √(Var / N)
    double call_se = std::sqrt(call_var_MC / num_sims);
    double put_se = std::sqrt(put_var_MC / num_sims);

    // 95% confidence intervals: Price ± 1.96 * SE
    double call_ci_low = call_price - 1.96 * call_se;
    double call_ci_high = call_price + 1.96 * call_se;
    double put_ci_low = put_price - 1.96 * put_se;
    double put_ci_high = put_price + 1.96 * put_se;

    // Control Variate Calculations

    double X_mean = X_sum / num_sims;           // E[discounted payoff]
    double Y_mean = Y_sum / num_sims;           // E[discounted spot]
    double cov_XY = (cross_sum / num_sims) - X_mean * Y_mean;
    double var_Y = (Y_sq_sum / num_sims) - Y_mean * Y_mean;

    // Optimal control variate coefficient: β = Cov(X, Y) / Var(Y)
    double beta = (var_Y > 1e-12) ? (cov_XY / var_Y) : 0.0;

    // Under risk-neutral measure: E[e^(-rT) * S_T] = S_0
    double expected_Y = S;

    // Control variate adjusted call: X̂ = X + β(E[Y] - Ȳ)
    double call_cv_price = X_mean + beta * (expected_Y - Y_mean);

    // CV variance: Var(X̂) = Var(X) + β²*Var(Y) - 2β*Cov(X,Y)
    double call_cv_var = call_var_MC + beta * beta * var_Y - 2.0 * beta * cov_XY;
    double call_cv_se = std::sqrt(call_cv_var / num_sims);

    // Variance reduction percentage: 1 - Var(X̂) / Var(X)
    double variance_reduction = 1.0 - (call_cv_var / call_var_MC);

    // Similar for puts
    double X_put_mean = X_put_sum / num_sims;
    double cov_put_Y = (cross_sum_put / num_sims) - X_put_mean * Y_mean;
    double beta_put = (var_Y > 1e-12) ? (cov_put_Y / var_Y) : 0.0;
    double put_cv_price = X_put_mean + beta_put * (expected_Y - Y_mean);
    double put_cv_var = put_var_MC + beta_put * beta_put * var_Y - 2.0 * beta_put * cov_put_Y;
    double put_cv_se = std::sqrt(put_cv_var / num_sims);

    // Greeks Calculations

    double call_delta = Discount * (delta_call_sum / num_sims);
    double put_delta = Discount * (delta_put_sum / num_sims);

    double call_gamma = Discount * (gamma_call_sum / num_sims);
    double put_gamma = Discount * (gamma_put_sum / num_sims);

    double call_vega = Discount * (vega_call_sum / num_sims);
    double put_vega = Discount * (vega_put_sum / num_sims);

    // Greeks with control variate adjustment
    double delta_Y_MC = (Y_sum / num_sims) / S;  // MC estimate of d(e^(-rT)*S)/dS
    double delta_Y_exact = 1.0;                   // Theoretical: d(S)/dS = 1
    double call_delta_cv = call_delta + (delta_Y_exact - delta_Y_MC);
    double put_delta_cv = put_delta + (delta_Y_exact - delta_Y_MC);

    // Gamma with control variate
    double X_gamma_mean = (gamma_call_sum * Discount) / num_sims;
    double cov_gamma_Y = (cross_sum_gamma / num_sims) - X_gamma_mean * Y_mean;
    double beta_gamma = (var_Y > 1e-12) ? (cov_gamma_Y / var_Y) : 0.0;
    double call_gamma_cv = call_gamma + beta_gamma * (expected_Y - Y_mean);

    double cov_gamma_Y_put = (cross_sum_gamma_put / num_sims) - (put_gamma * Y_mean);
    double beta_gamma_put = (var_Y > 1e-12) ? (cov_gamma_Y_put / var_Y) : 0.0;
    double put_gamma_cv = put_gamma + beta_gamma_put * (expected_Y - Y_mean);

    // Vega with adjustment
    double stock_vega_MC = Discount * (stock_vega_sum / num_sims);
    double call_vega_cv = call_vega - stock_vega_MC;
    double put_vega_cv = put_vega - stock_vega_MC;


    // Return Results

    MonteCarloResult result;
    result.call = call_price;
    result.put= put_price;
    result.call_variance = call_var_MC;
    result.put_variance = put_var_MC;
    result.call_std_error = call_se;
    result.put_std_error = put_se;
    result.call_ci_low = call_ci_low;
    result.call_ci_high = call_ci_high;
    result.put_ci_low = put_ci_low;
    result.put_ci_high = put_ci_high;

    result.call_cv = call_cv_price;
    result.put_cv = put_cv_price;
    result.call_cv_variance = call_cv_var;
    result.put_cv_variance = put_cv_var;
    result.call_cv_std_error = call_cv_se;
    result.put_cv_std_error = put_cv_se;
    result.variance_reduction = variance_reduction;

    result.beta = beta;
    result.call_delta = call_delta;
    result.put_delta = put_delta;
    result.call_gamma = call_gamma;
    result.put_gamma = put_gamma;
    result.call_vega = call_vega;
    result.put_vega = put_vega;

    result.call_delta_cv = call_delta_cv;
    result.put_delta_cv = put_delta_cv;
    result.call_gamma_cv = call_gamma_cv;
    result.put_gamma_cv = put_gamma_cv;
    result.call_vega_cv = call_vega_cv;
    result.put_vega_cv = put_vega_cv;

    return result;
}

MonteCarloResult monte_carlo_parallel(int num_sims, double S, double K, double r, double sigma, double T, double sqrt_T) {
    //Parallel version of Monte Carlo using , Same algorithm as sequential but parallelized with #pragma omp parallel for
     

    validate_option_parameters(num_sims, S, K, r, sigma, T);

    double Drift = T * (r - 0.5 * sigma * sigma);
    double Vol_sqrt_T = sigma * sqrt_T;
    double Discount = std::exp(-r * T);
    double base = S * std::exp(Drift);
    int half_sims = num_sims / 2;

    double call_sum = 0.0, call_sq_sum = 0.0;
    double put_sum = 0.0, put_sq_sum = 0.0;
    double X_sum = 0.0, X_put_sum = 0.0;
    double Y_sum = 0.0, Y_sq_sum = 0.0;
    double cross_sum = 0.0, cross_sum_put = 0.0;
    double delta_call_sum = 0.0, delta_put_sum = 0.0;
    double gamma_call_sum = 0.0, gamma_put_sum = 0.0;
    double vega_call_sum = 0.0, vega_put_sum = 0.0;
    double cross_sum_gamma = 0.0, cross_sum_gamma_put = 0.0;
    double cross_sum_vega = 0.0, cross_sum_vega_put = 0.0;
    double stock_vega_sum = 0.0;

#pragma omp parallel
    {
        // Per-thread RNG with deterministic seeding
        std::random_device rd;
        std::seed_seq seq{ 1234, omp_get_thread_num(), (int)rd() };
        std::mt19937 gen(seq);
        std::normal_distribution<double> d(0.0, 1.0);

#pragma omp for reduction(+:call_sum, call_sq_sum, put_sum, put_sq_sum, \
                            X_sum, X_put_sum, Y_sum, Y_sq_sum, cross_sum, cross_sum_put, \
                            delta_call_sum, delta_put_sum, gamma_call_sum, gamma_put_sum, \
                            vega_call_sum, vega_put_sum, cross_sum_gamma, cross_sum_gamma_put, \
                            cross_sum_vega, cross_sum_vega_put, stock_vega_sum)
        for (int i = 0; i < half_sims; i++) {
            double Z = d(gen);
            double Z_anti = -Z;
            double exp_term = std::exp(Vol_sqrt_T * Z);

            double ST1 = base * exp_term;
            double ST2 = base / exp_term;

            double call_payoff1 = std::max(ST1 - K, 0.0);
            double call_payoff2 = std::max(ST2 - K, 0.0);
            double put_payoff1 = std::max(K - ST1, 0.0);
            double put_payoff2 = std::max(K - ST2, 0.0);

            if (ST1 > K) delta_call_sum += ST1 / S;
            if (ST2 > K) delta_call_sum += ST2 / S;
            if (ST1 < K) delta_put_sum += (-ST1 / S);
            if (ST2 < K) delta_put_sum += (-ST2 / S);

            double L1 = (std::log(ST1 / S) - Drift) / Vol_sqrt_T;
            double L2 = (std::log(ST2 / S) - Drift) / Vol_sqrt_T;
            double gamma_weight1 = (Z * Z - 1.0) / (S * S * sigma * sigma * T) - Z / (S * S * sigma * sqrt_T);
            double gamma_weight2 = (Z_anti * Z_anti - 1.0) / (S * S * sigma * sigma * T) - Z_anti / (S * S * sigma * sqrt_T);

            gamma_call_sum += call_payoff1 * gamma_weight1 + call_payoff2 * gamma_weight2;
            gamma_put_sum += put_payoff1 * gamma_weight1 + put_payoff2 * gamma_weight2;

            double vega_weight1_call = call_payoff1 * ((L1 * L1 - 1.0) / sigma - L1 * sqrt_T);
            double vega_weight2_call = call_payoff2 * ((L2 * L2 - 1.0) / sigma - L2 * sqrt_T);
            double vega_weight1_put = put_payoff1 * (L1 * L1 / sigma - L1 * sqrt_T - 1.0 / sigma);
            double vega_weight2_put = put_payoff2 * (L2 * L2 / sigma - L2 * sqrt_T - 1.0 / sigma);

            vega_call_sum += vega_weight1_call + vega_weight2_call;
            vega_put_sum += vega_weight1_put + vega_weight2_put;

            call_sum += call_payoff1 + call_payoff2;
            call_sq_sum += call_payoff1 * call_payoff1 + call_payoff2 * call_payoff2;
            put_sum += put_payoff1 + put_payoff2;
            put_sq_sum += put_payoff1 * put_payoff1 + put_payoff2 * put_payoff2;

            double discounted_call1 = call_payoff1 * Discount;
            double discounted_call2 = call_payoff2 * Discount;
            double discounted_ST1 = ST1 * Discount;
            double discounted_ST2 = ST2 * Discount;

            X_sum += discounted_call1 + discounted_call2;
            Y_sum += discounted_ST1 + discounted_ST2;
            Y_sq_sum += discounted_ST1 * discounted_ST1 + discounted_ST2 * discounted_ST2;
            cross_sum += discounted_call1 * discounted_ST1 + discounted_call2 * discounted_ST2;

            double discounted_put1 = put_payoff1 * Discount;
            double discounted_put2 = put_payoff2 * Discount;
            cross_sum_put += discounted_put1 * discounted_ST1 + discounted_put2 * discounted_ST2;
            X_put_sum += discounted_put1 + discounted_put2;

            double discounted_gamma1 = call_payoff1 * gamma_weight1 * Discount;
            double discounted_gamma2 = call_payoff2 * gamma_weight2 * Discount;
            cross_sum_gamma += discounted_gamma1 * discounted_ST1 + discounted_gamma2 * discounted_ST2;

            double discounted_gamma_put1 = put_payoff1 * gamma_weight1 * Discount;
            double discounted_gamma_put2 = put_payoff2 * gamma_weight2 * Discount;
            cross_sum_gamma_put += discounted_gamma_put1 * discounted_ST1 + discounted_gamma_put2 * discounted_ST2;

            double discounted_vega1 = call_payoff1 * vega_weight1_call * Discount;
            double discounted_vega2 = call_payoff2 * vega_weight2_call * Discount;
            cross_sum_vega += discounted_vega1 * discounted_ST1 + discounted_vega2 * discounted_ST2;

            double discounted_vega_put1 = put_payoff1 * vega_weight1_put * Discount;
            double discounted_vega_put2 = put_payoff2 * vega_weight2_put * Discount;
            cross_sum_vega_put += discounted_vega_put1 * discounted_ST1 + discounted_vega_put2 * discounted_ST2;

            double stock_vega1 = ST1 * (Z * sqrt_T - sigma * T);
            double stock_vega2 = ST2 * (Z_anti * sqrt_T - sigma * T);
            stock_vega_sum += stock_vega1 + stock_vega2;
        }
    }

    // Post-computation (identical to sequential version)
    double call_mean = call_sum / num_sims;
    double put_mean = put_sum / num_sims;
    double call_price = call_mean * Discount;
    double put_price = put_mean * Discount;

    double call_var = (call_sq_sum - num_sims * call_mean * call_mean) / (num_sims - 1);
    double put_var = (put_sq_sum - num_sims * put_mean * put_mean) / (num_sims - 1);

    double call_var_MC = call_var * (Discount * Discount);
    double put_var_MC = put_var * (Discount * Discount);

    double call_se = std::sqrt(call_var_MC / num_sims);
    double put_se = std::sqrt(put_var_MC / num_sims);

    double call_ci_low = call_price - 1.96 * call_se;
    double call_ci_high = call_price + 1.96 * call_se;
    double put_ci_low = put_price - 1.96 * put_se;
    double put_ci_high = put_price + 1.96 * put_se;

    double X_mean = X_sum / num_sims;
    double Y_mean = Y_sum / num_sims;
    double cov_XY = (cross_sum / num_sims) - X_mean * Y_mean;
    double var_Y = (Y_sq_sum / num_sims) - Y_mean * Y_mean;

    double beta = (var_Y > 1e-12) ? (cov_XY / var_Y) : 0.0;
    double expected_Y = S;

    double call_cv_price = X_mean + beta * (expected_Y - Y_mean);
    double call_cv_var = call_var_MC + beta * beta * var_Y - 2.0 * beta * cov_XY;
    double call_cv_se = std::sqrt(call_cv_var / num_sims);
    double variance_reduction = 1.0 - (call_cv_var / call_var_MC);

    double X_put_mean = X_put_sum / num_sims;
    double cov_put_Y = (cross_sum_put / num_sims) - X_put_mean * Y_mean;
    double beta_put = (var_Y > 1e-12) ? (cov_put_Y / var_Y) : 0.0;
    double put_cv_price = X_put_mean + beta_put * (expected_Y - Y_mean);
    double put_cv_var = put_var_MC + beta_put * beta_put * var_Y - 2.0 * beta_put * cov_put_Y;
    double put_cv_se = std::sqrt(put_cv_var / num_sims);

    double call_delta = Discount * (delta_call_sum / num_sims);
    double put_delta = Discount * (delta_put_sum / num_sims);

    double call_gamma = Discount * (gamma_call_sum / num_sims);
    double put_gamma = Discount * (gamma_put_sum / num_sims);

    double call_vega = Discount * (vega_call_sum / num_sims);
    double put_vega = Discount * (vega_put_sum / num_sims);

    double delta_Y_MC = (Y_sum / num_sims) / S;
    double delta_Y_exact = 1.0;
    double call_delta_cv = call_delta + (delta_Y_exact - delta_Y_MC);
    double put_delta_cv = put_delta + (delta_Y_exact - delta_Y_MC);

    double X_gamma_mean = (gamma_call_sum * Discount) / num_sims;
    double cov_gamma_Y = (cross_sum_gamma / num_sims) - X_gamma_mean * Y_mean;
    double beta_gamma = (var_Y > 1e-12) ? (cov_gamma_Y / var_Y) : 0.0;
    double call_gamma_cv = call_gamma + beta_gamma * (expected_Y - Y_mean);

    double cov_gamma_Y_put = (cross_sum_gamma_put / num_sims) - (put_gamma * Y_mean);
    double beta_gamma_put = (var_Y > 1e-12) ? (cov_gamma_Y_put / var_Y) : 0.0;
    double put_gamma_cv = put_gamma + beta_gamma_put * (expected_Y - Y_mean);

    double stock_vega_MC = Discount * (stock_vega_sum / num_sims);
    double call_vega_cv = call_vega - stock_vega_MC;
    double put_vega_cv = put_vega - stock_vega_MC;

    MonteCarloResult result;
    result.call = call_price;
    result.put = put_price;
    result.call_variance = call_var_MC;
    result.put_variance = put_var_MC;
    result.call_std_error = call_se;
    result.put_std_error = put_se;
    result.call_ci_low = call_ci_low;
    result.call_ci_high = call_ci_high;
    result.put_ci_low = put_ci_low;
    result.put_ci_high = put_ci_high;

    result.call_cv = call_cv_price;
    result.put_cv = put_cv_price;
    result.call_cv_variance = call_cv_var;
    result.put_cv_variance = put_cv_var;
    result.call_cv_std_error = call_cv_se;
    result.put_cv_std_error = put_cv_se;
    result.variance_reduction = variance_reduction;

    result.beta = beta;
    result.call_delta = call_delta;
    result.put_delta = put_delta;
    result.call_gamma = call_gamma;
    result.put_gamma = put_gamma;
    result.call_vega = call_vega;
    result.put_vega = put_vega;

    result.call_delta_cv = call_delta_cv;
    result.put_delta_cv = put_delta_cv;
    result.call_gamma_cv = call_gamma_cv;
    result.put_gamma_cv = put_gamma_cv;
    result.call_vega_cv = call_vega_cv;
    result.put_vega_cv = put_vega_cv;

    return result;
}