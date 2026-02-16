
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <omp.h>

void monte_carlo_call_put_price(int num_sims, double Option_price, double Strike_price, double risk_free_per, double sigma, double Time_till_exp, std::mt19937& gen, double& call_seq, double& put_seq) {
    double Drift = Time_till_exp * (risk_free_per - 0.5 * sigma * sigma);
    double Op_price_cur1 = 0.0;
    double Op_price_cur2 = 0.0;
    double Vol_sq = sigma * std::sqrt(Time_till_exp);
    double call_sum = 0.0;
    double put_sum = 0.0;
    double call_sq_sum = 0.0;
    double put_sq_sum = 0.0;
    double call_payoff1, call_payoff2;
    double put_payoff1, put_payoff2;
    double Discount = std::exp(-risk_free_per * Time_till_exp);
    int half_sims = num_sims / 2;

    std::normal_distribution<double> d(0.0, 1.0);


    for (int i = 0; i < half_sims; i++) {
        double gauss_bm = d(gen);
        double gauss_bm_antithetic = -gauss_bm;
        Op_price_cur1 = Option_price * std::exp(Drift + Vol_sq * gauss_bm);
        Op_price_cur2 = Option_price * std::exp(Drift + Vol_sq * gauss_bm_antithetic);
        call_payoff1 = std::max(Op_price_cur1 - Strike_price, 0.0);
        call_payoff2 = std::max(Op_price_cur2 - Strike_price, 0.0);

        call_sum += call_payoff1 + call_payoff2;
        call_sq_sum += call_payoff1 * call_payoff1 + call_payoff2 * call_payoff2;

        put_payoff1 = std::max(Strike_price - Op_price_cur1, 0.0);
        put_payoff2 = std::max(Strike_price - Op_price_cur2, 0.0);

        put_sum += put_payoff1 + put_payoff2;
        put_sq_sum += put_payoff1 * put_payoff1 + put_payoff2 * put_payoff2;

    }
    call_seq = (call_sum / static_cast<double>(num_sims)) * Discount;
    put_seq = (put_sum / static_cast<double>(num_sims)) * Discount;

    //mean calculation
    double call_mean = call_sum / num_sims;
    double put_mean = put_sum / num_sims;
    //var calculation
    double call_var = (call_sq_sum / num_sims - call_mean * call_mean);
    double put_var = (put_sq_sum / num_sims - put_mean * put_mean);
    //standar error
    double call_se = std::sqrt(call_var / num_sims);
    double put_se = std::sqrt(put_var / num_sims);
        
    call_seq = call_mean * Discount;
    put_seq = put_mean * Discount;

    // 1.96= Z-value for 95% Confidence
    double call_ci_low = call_seq - 1.96 * call_se * Discount;
    double call_ci_high = call_seq + 1.96 * call_se * Discount;
    double put_ci_low = put_seq - 1.96 * put_se * Discount;
    double put_ci_high = put_seq + 1.96 * put_se * Discount;

    // Statistics printing
    std::cout << "Call price    :" << call_seq << std::endl;
    std::cout << "Standar error :" << call_se << std::endl;
    std::cout << "95% CI        :[" << call_ci_low << ", " << call_ci_high << " ]" << std::endl;
    std::cout << "Put price     :" << put_seq << std::endl;
    std::cout << "Standar error :" << put_se << std::endl;
    std::cout << "95% CI        :[" << put_ci_low << ", " << put_ci_high << " ]" << std::endl;


}

void monte_carlo_call_put_price_paral(int num_sims, double Option_price, double Strike_price, double risk_free_per, double sigma, double Time_till_exp, double& call_paral, double& put_paral) {
    double Drift = Time_till_exp * (risk_free_per - 0.5 * sigma * sigma);
    double Vol_sq = sigma * std::sqrt(Time_till_exp);
    double Discount = std::exp(-risk_free_per * Time_till_exp);
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
            Op_price_cur1 = Option_price * std::exp(Drift + Vol_sq * gauss_bm);
            Op_price_cur2 = Option_price * std::exp(Drift + Vol_sq * gauss_bm_antithetic);
            call_sum += std::max(Op_price_cur1 - Strike_price, 0.0);
            call_sum += std::max(Op_price_cur2 - Strike_price, 0.0);
            put_sum += std::max(Strike_price - Op_price_cur1, 0.0);
            put_sum += std::max(Strike_price - Op_price_cur2, 0.0);
        }
    }
    call_paral = (call_sum / static_cast<double>(num_sims)) * Discount;
    put_paral = (put_sum / static_cast<double>(num_sims)) * Discount;

}

int main()
{
    int num_sims = 10000000;
    omp_set_num_threads(omp_get_num_procs());
    double Option_price = 100.0;
    double Strike_price = 100.0;
    double risk_free_per = 0.05;
    double sigma = 0.2;
    double Time_till_exp = 1.0; //in years
    std::mt19937 gen(42);

    double call_seq, put_seq;
    double call_paral, put_paral;

    double start_seq = omp_get_wtime();
    monte_carlo_call_put_price(num_sims, Option_price, Strike_price, risk_free_per, sigma, Time_till_exp, gen, call_seq, put_seq);
    double end_seq = omp_get_wtime();

    double start_par = omp_get_wtime();
    monte_carlo_call_put_price_paral(num_sims, Option_price, Strike_price, risk_free_per, sigma, Time_till_exp, call_paral, put_paral);
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

    //std::cout << "Call: " << call_seq << "\nPut: " << put_seq << std::endl;
    std::cout << "Call Parallel: " << call_paral << "\nPut Parallel: " << put_paral << std::endl;
    std::cout << "Speedup: " << speedup << "x\nEfficiency: " << efficiency << "%" << std::endl;

    return 0;
}