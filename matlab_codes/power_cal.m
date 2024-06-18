function energy = power_cal(SF, PL, CRC, H, BW, DE, P_tx)
    Tsym = 2.^SF / BW;
    Tx = Tsym .* (20.25 + max(ceil((8 * PL - 4 * SF + 28 + 16 * CRC - 20 * H) / (4 * (SF - 2 * DE))) * (1 + 4), 0));
    energy = Tx .* P_tx;
end
