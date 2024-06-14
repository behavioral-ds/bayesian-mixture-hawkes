data {
    int N_videolevel_features;

    int<lower=0> V_train;   
    int<lower=0> C_train;
    int<lower=0> E_train;
    int N_cascades_per_video_train[V_train]; 
    int N_cascadesize_per_cascade_train[C_train] ;
    int N_clippedsize_per_cascade_train[C_train];
    vector<lower=0>[E_train] y_train;    
    vector[N_videolevel_features * V_train] videolevel_features_train;

    int<lower=0> V_holdout;   
    int<lower=0> C_holdout;
    int<lower=0> E_holdout;
    int N_cascades_per_video_holdout[V_holdout];
    int N_cascadesize_per_cascade_holdout[C_holdout];
    int N_clippedsize_per_cascade_holdout[C_holdout];
    vector<lower=0>[E_holdout] y_holdout;
    vector[N_videolevel_features * V_holdout] videolevel_features_holdout;

    int<lower=0> pmf_size;
    vector[pmf_size] pmf_x;
    vector[pmf_size] pmf_vals;
}

parameters { 
    // artist-level, simple model fixes beta across popular and unpopular, but have varying intercepts.
    vector[3] beta_a_raw; // intercept_pop, intercept_unpop, covariate_z. intercept z is separate
    cholesky_factor_corr[3] L;        // prior on cholesky decomp on correlation
    vector<lower=0>[3] sigma_beta_a;      // prior scale
    array[V_train] vector[3] beta_av_raw;
    
    // artist-level cascade-feature effects
    real<lower=0> gamma_z_alpha_sd;
    vector[N_videolevel_features] gamma_z_alpha_raw;

    real<lower=0> gamma_a_alpha_sd;
    vector[N_videolevel_features] gamma_a_alpha_raw;
} 

transformed parameters {
    // prior on artist alpha
    vector[3] beta_a_mean = rep_vector(0, 3);
    beta_a_mean[1] = -1.39; // bias the popular intercept
    beta_a_mean[2] = 2; // bias the unpopular intercept
    beta_a_mean[3] = -2; // bias the unpopular intercept

    vector[3] beta_a_sd = rep_vector(0.1, 3);
    beta_a_sd[1] = 0.5; // special prior for intercept
    beta_a_sd[2] = 0.5; // special prior for intercept
    beta_a_sd[3] = 0.5; // special prior for intercept

    vector[3] beta_a = beta_a_mean + beta_a_sd .* beta_a_raw;

    array[V_train] vector[3] beta_av;

    real gamma_a_alpha_mean = 0;
    vector[N_videolevel_features] gamma_a_alpha = gamma_a_alpha_mean+gamma_a_alpha_sd*gamma_a_alpha_raw;

    real gamma_z_alpha_mean = 0;
    vector[N_videolevel_features] gamma_z_alpha = gamma_z_alpha_mean+gamma_z_alpha_sd*gamma_z_alpha_raw;

    for (v in 1:V_train) {   
        beta_av[v] = beta_a + sigma_beta_a .* (L * beta_av_raw[v]);        
    }
}

model {
    // prior on video level covariance  
    sigma_beta_a ~ normal(0,1);
    L ~ lkj_corr_cholesky(2); # implies L * L' ~ lkj_corr(2.0);
    beta_a_raw ~ normal(0,1);

    gamma_z_alpha_sd ~ exponential(10); // prior for laplace scale
    gamma_z_alpha_raw ~ normal(0,1);

    gamma_a_alpha_sd ~ exponential(10); // prior for laplace scale
    gamma_a_alpha_raw ~ normal(0,1);

    int pos_E = 1;   // tracker on y
    int pos_C = 1;   // tracker on N_events_per_cascade
    
    for (v in 1:V_train) {        
        int C_of_v = N_cascades_per_video_train[v];
        vector[N_videolevel_features] y_av = videolevel_features_train[1+(v-1)*N_videolevel_features: v*N_videolevel_features];
        
        // video-level intercepts
        beta_av_raw[v] ~ normal(0,1);

        int cascade_sizes[C_of_v] = N_cascadesize_per_cascade_train[pos_C:pos_C+C_of_v-1];
        
        real alpha_avc_popular = inv_logit(beta_av[v][2]+ dot_product(gamma_a_alpha, y_av));
        real alpha_avc_unpopular = inv_logit(beta_av[v][3] + dot_product(gamma_a_alpha, y_av));

        for (k in 1:C_of_v) {
            // cascade-level features
            real zinf_avc = inv_logit(beta_av[v][1] + dot_product(gamma_z_alpha, y_av));
            
            target += log_sum_exp(log(zinf_avc) + poisson_lpmf(cascade_sizes[k]-1 | cascade_sizes[k]*alpha_avc_popular) - log(cascade_sizes[k]), log1m(zinf_avc) + poisson_lpmf(cascade_sizes[k]-1 | cascade_sizes[k]*alpha_avc_unpopular) - log(cascade_sizes[k]));
       }

        // iterate to the beginning of the next video's first cascade  
        pos_E += to_int(sum(N_cascadesize_per_cascade_train[pos_C:pos_C+C_of_v-1]));       
        pos_C += C_of_v;        
    }
}

generated quantities {
    int pos_C = 1;   // tracker on N_events_per_cascade
    int pos_E = 1;   // tracker on y
        
    array[V_holdout] vector[3] beta_av_holdout;

    matrix[3, 3] Omega_beta = L * L';
    matrix[3, 3] covariance_beta = quad_form_diag(Omega_beta, sigma_beta_a);

    vector[V_holdout] pred_holdout = rep_vector(0,V_holdout);
    vector[V_holdout] qred_holdout = rep_vector(0,V_holdout);
    vector[V_holdout] actual_holdout = rep_vector(0,V_holdout);
    
    for (v in 1:V_holdout) {      
        // draw samples
        beta_av_holdout[v] = multi_normal_rng(beta_a, covariance_beta);
        
        int C_of_v = N_cascades_per_video_holdout[v];
        vector[N_videolevel_features] y_av_holdout = videolevel_features_holdout[1+(v-1)*N_videolevel_features: v*N_videolevel_features];
        int cascade_sizes_holdout[C_of_v] = to_array_1d(N_cascadesize_per_cascade_holdout[pos_C:pos_C+C_of_v-1]);
        real alpha_avc_popular_holdout = inv_logit(beta_av_holdout[v][2] + dot_product(gamma_a_alpha, y_av_holdout));
        real alpha_avc_unpopular_holdout = inv_logit(beta_av_holdout[v][3] + dot_product(gamma_a_alpha, y_av_holdout));

        for (k in 1:pmf_size) {
            real zinf_avc_holdout = inv_logit(beta_av_holdout[v][1] + dot_product(gamma_z_alpha, y_av_holdout));

            real pred_cascadesize_popular_avc_holdout = (1 / (1-alpha_avc_popular_holdout));
            real pred_cascadesize_unpopular_avc_holdout = (1 / (1-alpha_avc_unpopular_holdout));
            real pred_cascadesize_avc_holdout = pred_cascadesize_popular_avc_holdout * zinf_avc_holdout + pred_cascadesize_unpopular_avc_holdout * (1-zinf_avc_holdout);
            real qred_cascadesize_avc_holdout = pred_cascadesize_popular_avc_holdout * round(zinf_avc_holdout) + pred_cascadesize_unpopular_avc_holdout * (1-round(zinf_avc_holdout));

            pred_holdout[v] += pred_cascadesize_avc_holdout * pmf_vals[k];
            qred_holdout[v] += qred_cascadesize_avc_holdout * pmf_vals[k];
        }
        actual_holdout[v] = sum(N_cascadesize_per_cascade_holdout[pos_C:pos_C+C_of_v-1]);
        
        pos_E += to_int(sum(N_cascadesize_per_cascade_holdout[pos_C:pos_C+C_of_v-1]));         
        pos_C += C_of_v;        
    }

}