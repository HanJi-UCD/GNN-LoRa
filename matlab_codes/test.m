%
sigma = 6.87;
% find 95% CI for noise distribution
alpha = 0.05; % 
z_critical = norminv(1 - alpha/2); 
lower_bound = 0 + z_critical * sigma;
upper_bound = 0 - z_critical * sigma;
fprintf('95% CI is: [%.4f, %.4f]\n', lower_bound, upper_bound);
CI = [lower_bound, upper_bound];
% d < 345






