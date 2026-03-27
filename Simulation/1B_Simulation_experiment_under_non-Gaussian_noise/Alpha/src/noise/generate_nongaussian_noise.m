function n = generate_nongaussian_noise(sz, sigma, params, sensor_name)
% Generate non-Gaussian noise for a selected sensor model and target scale.
% sz: [m,n]
% sigma: target RMS scale from SNR
% sensor_name: 'tdoa' | 'camera' | 'detector' | 'depth'

    type = params.noise.(sensor_name).type;

    switch lower(type)
        case 'gmm'
            n = noise_gmm(sz, sigma, params.noise.gmm.eps, params.noise.gmm.kappa);

        case 'alpha'
            alpha = params.noise.alpha.alpha;
            beta  = params.noise.alpha.beta;
            n = noise_alpha_stable(sz, sigma, alpha, beta);

        case 'middletona'
            A     = params.noise.midA.A;
            Gamma = params.noise.midA.Gamma;
            Mmax  = params.noise.midA.Mmax;
            n = noise_middletonA(sz, sigma, A, Gamma, Mmax);

        otherwise
            n = randn(sz) * sigma;
    end
end
