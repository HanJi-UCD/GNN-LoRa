function Lpl = calculateLpl(d, d0, Lpl_bar, gamma, sigma)
    % 生成 N(0, sigma^2) 的随机数
    limit = 13.45; % unit: dB, a very sensitive and dedicated parameter
    chi_sigma = min(max(sigma * randn(length(d), 1), -limit), limit);
    % 计算路径损耗
    Lpl = Lpl_bar + 10 * gamma * log10(d/d0) + chi_sigma;
end
