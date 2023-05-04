from Scripts.Data_simulations import agg_sim
n_agg = 10
p_agg = [.33,.33,.34]
noise_ratio = 1.5
data_set = agg_sim(n_agg = n_agg,noise_ratio = noise_ratio,include_ratio = p_agg)
data_set.to_csv('Test_data/simulation_test.csv',index = False)

