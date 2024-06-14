functions {
    vector g_pl(vector x, real theta_pl, real cutoff) {
        return (theta_pl / cutoff) * pow(cutoff / (x+cutoff), (1+theta_pl));
  }
    vector log_g_pl(vector x, real theta, real cutoff, real log_theta, real log_cutoff) {
        return log_theta + theta * log_cutoff - (1 + theta) * log(x + cutoff);
  }
}

data {
    int<lower=0> V_train;   
    int<lower=0> C_train;
    int<lower=0> E_train;
    int N_cascades_per_video_train[V_train]; 
    int N_cascadesize_per_cascade_train[C_train] ;
    vector<lower=0>[E_train] d_train;    
    int<lower=0> V_holdout;   
    int<lower=0> C_holdout;
    int<lower=0> E_holdout;
    int N_cascades_per_video_holdout[V_holdout];
    int N_cascadesize_per_cascade_holdout[C_holdout];
    vector<lower=0>[E_holdout] d_holdout;
}

parameters { 
    cholesky_factor_corr[2] L_z; // mixture probabilities, set first class to 0
    // z refers to class 2 (slow) and class 3 (fast) probs
    cholesky_factor_corr[2] L_plfast;
    cholesky_factor_corr[2] L_plmid;
    cholesky_factor_corr[2] L_plslow;

    vector[2] beta_a_z_raw;
    vector[2] beta_a_plfast_raw;
    vector[2] beta_a_plmid_raw;
    vector[2] beta_a_plslow_raw;

    array[V_train] vector<lower=-5,upper=5>[2] beta_av_z_raw;
    array[V_train] vector[2] beta_av_plfast_raw;
    array[V_train] vector[2] beta_av_plmid_raw;
    array[V_train] vector[2] beta_av_plslow_raw;

    vector<lower=0>[2] sigma_beta_a_z;
    vector<lower=0>[2] sigma_beta_a_plfast;
    vector<lower=0>[2] sigma_beta_a_plmid;
    vector<lower=0>[2] sigma_beta_a_plslow;
} 

transformed parameters { 
    // means, sds of artist-level distributions
    vector[2] beta_a_z_mean = rep_vector(0, 2);
    beta_a_z_mean[1] = -2; // slow intercept
    beta_a_z_mean[2] = -2; // fast intercept
    vector[2] beta_a_z_sd = rep_vector(1, 2);

    vector[2] beta_a_plslow_mean = rep_vector(0, 2);
    beta_a_plslow_mean[1] = 4;
    beta_a_plslow_mean[2] = 4.805;
    vector[2] beta_a_plslow_sd = rep_vector(0, 2);
    beta_a_plslow_sd[1] = 0.5;
    beta_a_plslow_sd[2] = 0.5;

    vector[2] beta_a_plmid_mean = rep_vector(0, 2);
    beta_a_plmid_mean[1] = -0.41;
    beta_a_plmid_mean[2] = -1.37;
    vector[2] beta_a_plmid_sd = rep_vector(0, 2);
    beta_a_plmid_sd[1] = 0.5;
    beta_a_plmid_sd[2] = 1;

    vector[2] beta_a_plfast_mean = rep_vector(0, 2);
    beta_a_plfast_mean[1] = 4;
    beta_a_plfast_mean[2] = 1;
    vector[2] beta_a_plfast_sd = rep_vector(0, 2);
    beta_a_plfast_sd[1] = 0.5;
    beta_a_plfast_sd[2] = 0.5;

    vector[2] beta_a_z = beta_a_z_mean + beta_a_z_sd .* beta_a_z_raw;
    vector[2] beta_a_plfast = beta_a_plfast_mean + beta_a_plfast_sd .* beta_a_plfast_raw;
    vector[2] beta_a_plmid = beta_a_plmid_mean + beta_a_plmid_sd .* beta_a_plmid_raw;
    vector[2] beta_a_plslow = beta_a_plslow_mean + beta_a_plslow_sd .* beta_a_plslow_raw;

    array[V_train] vector[2] beta_av_z;
    array[V_train] vector[2] beta_av_plfast;
    array[V_train] vector[2] beta_av_plmid;
    array[V_train] vector[2] beta_av_plslow;
    for (v in 1:V_train) {   
        beta_av_z[v] = beta_a_z + sigma_beta_a_z .* (L_z * beta_av_z_raw[v]);        
        beta_av_plfast[v] = beta_a_plfast + sigma_beta_a_plfast .* (L_plfast * beta_av_plfast_raw[v]);        
        beta_av_plmid[v] = beta_a_plmid + sigma_beta_a_plmid .* (L_plmid * beta_av_plmid_raw[v]);        
        beta_av_plslow[v] = beta_a_plslow + sigma_beta_a_plslow .* (L_plslow * beta_av_plslow_raw[v]);        
    }
}


model {
    // sigma priors
    sigma_beta_a_z ~ normal(0,1);
    sigma_beta_a_plfast[1] ~ normal(0,0.5);
    sigma_beta_a_plfast[2] ~ normal(0,0.5);
    sigma_beta_a_plmid[1] ~ normal(0,0.5); // theta 
    sigma_beta_a_plmid[2] ~ normal(0,1); // c
    sigma_beta_a_plslow[1] ~ normal(0,0.5);
    sigma_beta_a_plslow[2] ~ normal(0,0.5);

    L_z ~ lkj_corr_cholesky(2); # implies L * L' ~ lkj_corr(2.0);
    L_plslow ~ lkj_corr_cholesky(0.5);
    L_plmid ~ lkj_corr_cholesky(0.5);
    L_plfast ~ lkj_corr_cholesky(0.5);

    beta_a_z_raw ~ normal(0,1);
    beta_a_plmid_raw ~ normal(0,1);
    beta_a_plfast_raw ~ normal(0,1);
    beta_a_plslow_raw ~ normal(0,1);

    int pos_C = 1;   // tracker on N_events_per_cascade
    int pos_E = 1;   // tracker on d

    for (v in 1:V_train) {        
        int C_of_v = N_cascades_per_video_train[v];
        beta_av_z_raw[v] ~ normal(0,1);
        beta_av_plmid_raw[v] ~ normal(0,1);
        beta_av_plfast_raw[v] ~ normal(0,1);
        beta_av_plslow_raw[v] ~ normal(0,1);
        
        // article-level intercepts

        vector[3] logz_avc = log_softmax(append_row(0, beta_av_z[v])); // vector with 3 components right now. logprob of mid, slow, fast resp.

        real log_theta_plslow_avc = beta_av_plslow[v][1];
        real log_theta_plfast_avc = beta_av_plfast[v][1];
        real log_theta_plmid_avc = beta_av_plmid[v][1];
        real log_cutoff_plslow_avc = beta_av_plslow[v][2];
        real log_cutoff_plfast_avc = beta_av_plfast[v][2];
        real log_cutoff_plmid_avc = beta_av_plmid[v][2];

        real theta_plmid_avc= exp(log_theta_plmid_avc);
        real cutoff_plmid_avc = exp(log_cutoff_plmid_avc);
        real theta_plfast_avc= exp(log_theta_plfast_avc);
        real cutoff_plfast_avc = exp(log_cutoff_plfast_avc);
        real theta_plslow_avc= exp(log_theta_plslow_avc);
        real cutoff_plslow_avc = exp(log_cutoff_plslow_avc);

        int cascade_sizes[C_of_v] = N_cascadesize_per_cascade_train[pos_C:pos_C+C_of_v-1];        

        for (k in 1:C_of_v) {
            int G = cascade_sizes[k]; // length of cascade k

            real log_plfast = 0;
            real log_plslow = 0;
            real log_plmid = 0;
            for (i in 1:G-1) {
                log_plmid += log_sum_exp(log_g_pl(d_train[pos_E : pos_E+i-1], theta_plmid_avc, cutoff_plmid_avc, log_theta_plmid_avc, log_cutoff_plmid_avc));
                log_plfast += log_sum_exp(log_g_pl(d_train[pos_E : pos_E+i-1], theta_plfast_avc, cutoff_plfast_avc, log_theta_plfast_avc, log_cutoff_plfast_avc));
                log_plslow += log_sum_exp(log_g_pl(d_train[pos_E : pos_E+i-1], theta_plslow_avc, cutoff_plslow_avc, log_theta_plslow_avc, log_cutoff_plslow_avc));
                // increment d
                pos_E += i;
            }

            vector[3] log_l;
            log_l[1] = logz_avc[1] + log_plmid;
            log_l[2] = logz_avc[2] + log_plslow;
            log_l[3] = logz_avc[3] + log_plfast;
            target += log_sum_exp(log_l);   
        }

        // iterate to the beginning of the next video's first cascade  
        pos_C += C_of_v;        
    }
    print("total E: ", pos_E);
}

generated quantities {
    int pos_C = 1;   // tracker on N_events_per_cascade
    int pos_E = 1;
        
    array[V_holdout] vector[2] beta_av_z_holdout;
    array[V_holdout] vector[2] beta_av_plslow_holdout;
    array[V_holdout] vector[2] beta_av_plfast_holdout;
    array[V_holdout] vector[2] beta_av_plmid_holdout;

    matrix[2,2] Omega_beta_z = L_z * L_z';
    matrix[2,2] Omega_beta_plfast = L_plfast * L_plfast';
    matrix[2,2] Omega_beta_plmid = L_plmid * L_plmid';
    matrix[2,2] Omega_beta_plslow = L_plslow * L_plslow';

    matrix[2,2] covariance_beta_z = quad_form_diag(Omega_beta_z, sigma_beta_a_z);
    matrix[2,2] covariance_beta_plslow = quad_form_diag(Omega_beta_plslow, sigma_beta_a_plslow);
    matrix[2,2] covariance_beta_plfast = quad_form_diag(Omega_beta_plfast, sigma_beta_a_plfast);
    matrix[2,2] covariance_beta_plmid = quad_form_diag(Omega_beta_plmid, sigma_beta_a_plmid);

    vector[V_holdout] log_lik = rep_vector(0,V_holdout);    
    vector[C_holdout] log_lik_per_cascade = rep_vector(0,C_holdout);    

    int pos_C_flat = 1; // tracker on cascade number flattened    
    for (v in 1:V_holdout) {        
        beta_av_z_holdout[v] = multi_normal_rng(beta_a_z, covariance_beta_z);
        beta_av_plslow_holdout[v] = multi_normal_rng(beta_a_plslow, covariance_beta_plslow);
        beta_av_plfast_holdout[v] = multi_normal_rng(beta_a_plfast, covariance_beta_plfast);
        beta_av_plmid_holdout[v] = multi_normal_rng(beta_a_plmid, covariance_beta_plmid);

        vector[3] logz_avc_holdout = log_softmax(append_row(0, beta_av_z_holdout[v])); 

        real log_theta_plslow_avc_holdout = beta_av_plslow_holdout[v][1];
        real log_theta_plfast_avc_holdout = beta_av_plfast_holdout[v][1];
        real log_theta_plmid_avc_holdout = beta_av_plmid_holdout[v][1];
        real log_cutoff_plslow_avc_holdout = beta_av_plslow_holdout[v][2];
        real log_cutoff_plfast_avc_holdout = beta_av_plfast_holdout[v][2];
        real log_cutoff_plmid_avc_holdout = beta_av_plmid_holdout[v][2];

        real theta_plslow_avc_holdout = exp(log_theta_plslow_avc_holdout);
        real theta_plfast_avc_holdout = exp(log_theta_plfast_avc_holdout);
        real theta_plmid_avc_holdout = exp(log_theta_plmid_avc_holdout);
        real cutoff_plslow_avc_holdout = exp(log_cutoff_plslow_avc_holdout);
        real cutoff_plfast_avc_holdout = exp(log_cutoff_plfast_avc_holdout);
        real cutoff_plmid_avc_holdout = exp(log_cutoff_plmid_avc_holdout);

        int C_of_v = N_cascades_per_video_holdout[v];
        int cascade_sizes_holdout[C_of_v] = to_array_1d(N_cascadesize_per_cascade_holdout[pos_C:pos_C+C_of_v-1]);
         
        for (k in 1:C_of_v) {
            int G = cascade_sizes_holdout[k]; // length of cascade k
            real log_plslow = 0;
            real log_plfast = 0;
            real log_plmid = 0;
            for (i in 1:G-1) {
                log_plslow += log_sum_exp(log_g_pl(d_holdout[pos_E : pos_E+i-1], theta_plslow_avc_holdout, cutoff_plslow_avc_holdout, log_theta_plslow_avc_holdout, log_cutoff_plslow_avc_holdout));
                log_plfast += log_sum_exp(log_g_pl(d_holdout[pos_E : pos_E+i-1], theta_plfast_avc_holdout, cutoff_plfast_avc_holdout, log_theta_plfast_avc_holdout, log_cutoff_plfast_avc_holdout));
                log_plmid += log_sum_exp(log_g_pl(d_holdout[pos_E : pos_E+i-1], theta_plmid_avc_holdout, cutoff_plmid_avc_holdout, log_theta_plmid_avc_holdout, cutoff_plmid_avc_holdout));
                
                // increment d
                pos_E += i;
            }
            vector[3] log_l_holdout;
            log_l_holdout[1] = logz_avc_holdout[1] + log_plmid;
            log_l_holdout[2] = logz_avc_holdout[2] + log_plslow;
            log_l_holdout[3] = logz_avc_holdout[3] + log_plfast;

            log_lik_per_cascade[pos_C_flat] = log_sum_exp(log_l_holdout);
            log_lik[v] += log_lik_per_cascade[pos_C_flat];   

            pos_C_flat += 1;
        }
        pos_C += C_of_v;        
    }

}
