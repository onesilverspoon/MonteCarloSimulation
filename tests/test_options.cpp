#include<gtest/gtest.h>
#include<cmath>
#include"monte_carlo/options.h"

TEST(BlackScholes, CallKnownValue) {
    double price = black_scholes_call(100, 100, 0.05, 0.2, 1.0, 1.0);
    EXPECT_NEAR(price, 10.4506, 0.001);
}

TEST(BlackScholes, PutKnownValue) {
    double price = black_scholes_put(100, 100, 0.05, 0.2, 1.0, 1.0);
    EXPECT_NEAR(price, 5.5735, 0.001);
}

// Put-Call parity: C - P = S - K*e^(-rT)
TEST(BlackScholes, PutCallParity) {
    double C = black_scholes_call(100, 100, 0.05, 0.2, 1.0, 1.0);
    double P = black_scholes_put(100, 100, 0.05, 0.2, 1.0, 1.0);
    double parity = 100.0 - 100.0 * std::exp(-0.05 * 1.0);
    EXPECT_NEAR(C - P, parity, 1e-6);
}

TEST(BlackScholes, GammaKnownValue) {
    double gamma = black_scholes_gamma(100, 100, 0.05, 0.2, 1.0, 1.0);
    EXPECT_NEAR(gamma, 0.01876, 0.001);
}

// Validation: should throw on bad inputs
TEST(Validation, NegativeSpot) {
    EXPECT_THROW(validate_option_parameters(1000, -1.0, 100, 0.05, 0.2, 1.0),
        std::invalid_argument);
}

TEST(Validation, ZeroSims) {
    EXPECT_THROW(validate_option_parameters(0, 100, 100, 0.05, 0.2, 1.0),
        std::invalid_argument);
}

TEST(Validation, NegativeVolatility) {
    EXPECT_THROW(validate_option_parameters(1000, 100, 100, 0.05, -0.1, 1.0),
        std::invalid_argument);
}

TEST(Validation, NegativeMaturity) {
    EXPECT_THROW(validate_option_parameters(1000, 100, 100, 0.05, 0.2, -1.0),
        std::invalid_argument);
}

//MC convergence to Black-Scholes (1% tolerance with 1M paths)
TEST(MonteCarlo, SequentialConvergesToBS) {
    double bs_call = black_scholes_call(100, 100, 0.05, 0.2, 1.0, 1.0);
    MonteCarloResult res = monte_carlo_sequential(1000000, 100, 100, 0.05, 0.2, 1.0, 1.0);
    EXPECT_NEAR(res.call, bs_call, bs_call * 0.01);
}

TEST(MonteCarlo, SequentialPutConvergesToBS) {
    double bs_put = black_scholes_put(100, 100, 0.05, 0.2, 1.0, 1.0);
    MonteCarloResult res = monte_carlo_sequential(1000000, 100, 100, 0.05, 0.2, 1.0, 1.0);
    EXPECT_NEAR(res.put, bs_put, bs_put * 0.01);
}

//Control variate actually reduces variance
TEST(MonteCarlo, VarianceReductionIsPositive) {
    MonteCarloResult res = monte_carlo_sequential(100000, 100, 100, 0.05, 0.2, 1.0, 1.0);
    EXPECT_GT(res.variance_reduction, 0.0);
}

//Confidence interval is valid
TEST(MonteCarlo, ConfidenceIntervalIsOrdered) {
    MonteCarloResult res = monte_carlo_sequential(100000, 100, 100, 0.05, 0.2, 1.0, 1.0);
    EXPECT_LT(res.call_ci_low, res.call_ci_high);
    EXPECT_LT(res.put_ci_low, res.put_ci_high);
}

//Parallel matches sequential within 1%
TEST(MonteCarlo, ParallelMatchesSequential) {
    MonteCarloResult seq = monte_carlo_sequential(500000, 100, 100, 0.05, 0.2, 1.0, 1.0);
    MonteCarloResult par = monte_carlo_parallel(500000, 100, 100, 0.05, 0.2, 1.0, 1.0);
    EXPECT_NEAR(seq.call, par.call, seq.call * 0.01);
    EXPECT_NEAR(seq.put, par.put, seq.put * 0.01);
}