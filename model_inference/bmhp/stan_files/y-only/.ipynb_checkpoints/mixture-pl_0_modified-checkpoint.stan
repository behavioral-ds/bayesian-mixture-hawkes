data {
    int<lower=0> V_train;   
    int<lower=0> C_train;
    int<lower=0> E_train;
    int N_cascades_per_video_train[V_train]; 
    int N_cascadesize_per_cascade_train[C_train] ;
    int N_clippedsize_per_cascade_train[C_train];
    vector<lower=0>[E_train] y_train;    
    int<lower=0> V_holdout;   
    int<lower=0> C_holdout;
    int<lower=0> E_holdout;
    int N_cascades_per_video_holdout[V_holdout];
    int N_cascadesize_per_cascade_holdout[C_holdout];
    int N_clippedsize_per_cascade_holdout[C_holdout];
    vector<lower=0>[E_holdout] y_holdout;

    int<lower=0> pmf_size;
    vector[pmf_size] pmf_x;
    vector[pmf_size] pmf_vals;
}

parameters { 
    // artist-level
    real delta_a_alpha_popular_raw;
    real<lower=0> sigma_delta_a_alpha_popular;
    real delta_a_alpha_unpopular_raw;
    real<lower=0> sigma_delta_a_alpha_unpopular;
    
    // video-level
    vector[V_train] log_delta_av_alpha_popular_raw;
    vector[V_train] log_delta_av_alpha_unpopular_raw;
    
    // zinf parameters
    real delta_a_zinf_raw;
    real<lower=0> sigma_delta_a_zinf;
    vector[V_train] log_delta_av_zinf_raw;
} 

transformed parameters {
    // prior mean and sd;
    real delta_a_alpha_popular_mean = 2;
    real delta_a_alpha_popular_sd = 0.5;
    
    real delta_a_alpha_unpopular_mean = -2;
    real delta_a_alpha_unpopular_sd = 0.5;

    real delta_a_zinf_mean = -1.39;
    real delta_a_zinf_sd = 0.5;

    // artist level
    real delta_a_alpha_popular = delta_a_alpha_popular_mean+delta_a_alpha_popular_sd*delta_a_alpha_popular_raw;
    real delta_a_alpha_unpopular = delta_a_alpha_unpopular_mean+delta_a_alpha_unpopular_sd*delta_a_alpha_unpopular_raw;
    real delta_a_zinf = delta_a_zinf_mean+delta_a_zinf_sd*delta_a_zinf_raw;
    
    vector[V_train] delta_av_alpha_popular;
    vector[V_train] delta_av_alpha_unpopular;
    vector[V_train] delta_av_zinf;

    for (v in 1:V_train) {   
        delta_av_alpha_popular[v] = delta_a_alpha_popular + sigma_delta_a_alpha_popular * log_delta_av_alpha_popular_raw[v];
        delta_av_alpha_unpopular[v] = delta_a_alpha_unpopular + sigma_delta_a_alpha_unpopular * log_delta_av_alpha_unpopular_raw[v];
        
        delta_av_zinf[v] = delta_a_zinf + sigma_delta_a_zinf * log_delta_av_zinf_raw[v];
    }
}

model {
    // set hyperpriors  
    sigma_delta_a_alpha_popular ~ normal(0,1); // video-level variation
    sigma_delta_a_alpha_unpopular ~ normal(0,1); // video-level variation
    sigma_delta_a_zinf ~ normal(0,1);
    
    delta_a_alpha_popular_raw ~ normal(0,1);
    delta_a_alpha_unpopular_raw ~ normal(0,1);
    delta_a_zinf_raw ~ normal(0,1);

    int pos_C = 1;   // tracker on N_events_per_cascade
    
    for (v in 1:V_train) {        
        int C_of_v = N_cascades_per_video_train[v];
        
        // video-level intercepts
        log_delta_av_alpha_popular_raw[v] ~ normal(0,1);
        log_delta_av_alpha_unpopular_raw[v] ~ normal(0,1);
        log_delta_av_zinf_raw[v] ~ normal(0,1);
        
        int cascade_sizes[C_of_v] = N_cascadesize_per_cascade_train[pos_C:pos_C+C_of_v-1];
        
        real alpha_avc_popular = inv_logit(delta_av_alpha_popular[v]);
        real alpha_avc_unpopular = inv_logit(delta_av_alpha_unpopular[v]);
        real zinf_avc = inv_logit(delta_av_zinf[v]);

        for (k in 1:C_of_v) {
            target += log_sum_exp(log(zinf_avc) + poisson_lpmf(cascade_sizes[k]-1 | cascade_sizes[k]*alpha_avc_popular) - log(cascade_sizes[k]), log1m(zinf_avc) + poisson_lpmf(cascade_sizes[k]-1 | cascade_sizes[k]*alpha_avc_unpopular) - log(cascade_sizes[k]));
       }

        // iterate to the beginning of the next video's first cascade  
        pos_C += C_of_v;        
    }
}

generated quantities {
    int pos_C = 1;   // tracker on N_events_per_cascade
        
    vector[V_holdout] delta_av_alpha_popular_holdout;
    vector[V_holdout] delta_av_alpha_unpopular_holdout;
    vector[V_holdout] delta_av_zinf_holdout;

    vector[V_holdout] pred_holdout = rep_vector(0,V_holdout);
    vector[V_holdout] qred_holdout = rep_vector(0,V_holdout);
    vector[V_holdout] actual_holdout = rep_vector(0,V_holdout);
    
    for (v in 1:V_holdout) {        
        delta_av_alpha_popular_holdout[v] = normal_rng(delta_a_alpha_popular, sigma_delta_a_alpha_popular);
        delta_av_alpha_unpopular_holdout[v] = normal_rng(delta_a_alpha_unpopular, sigma_delta_a_alpha_unpopular);

        delta_av_zinf_holdout[v] = normal_rng(delta_a_zinf, sigma_delta_a_zinf);
        
        int C_of_v = N_cascades_per_video_holdout[v];
        int cascade_sizes_holdout[C_of_v] = to_array_1d(N_cascadesize_per_cascade_holdout[pos_C:pos_C+C_of_v-1]);
        real alpha_avc_popular_holdout = inv_logit(delta_av_alpha_popular_holdout[v]);
        real alpha_avc_unpopular_holdout = inv_logit(delta_av_alpha_unpopular_holdout[v]);
        real zinf_avc_holdout = inv_logit(delta_av_zinf_holdout[v]);
        real pred_cascadesize_popular_avc_holdout = (1 / (1-alpha_avc_popular_holdout));
        real pred_cascadesize_unpopular_avc_holdout = (1 / (1-alpha_avc_unpopular_holdout));
        real pred_cascadesize_avc_holdout = pred_cascadesize_popular_avc_holdout * zinf_avc_holdout + pred_cascadesize_unpopular_avc_holdout * (1-zinf_avc_holdout);
        real qred_cascadesize_avc_holdout = pred_cascadesize_popular_avc_holdout * round(zinf_avc_holdout) + pred_cascadesize_unpopular_avc_holdout * (1-round(zinf_avc_holdout));

        for (k in 1:pmf_size) {
            pred_holdout[v] += pred_cascadesize_avc_holdout * pmf_vals[k];
            qred_holdout[v] += qred_cascadesize_avc_holdout * pmf_vals[k];
        }
        actual_holdout[v] = sum(N_cascadesize_per_cascade_holdout[pos_C:pos_C+C_of_v-1]);

        pos_C += C_of_v;   
    }  
}