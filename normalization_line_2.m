function nsig = normalization_line_2(sig)

c = 1;
miu = mean(sig);
tmp = bsxfun(@minus, sig, miu);
sigma = sqrt(sum(tmp.^2)./size(sig,1));
nsig = bsxfun(@rdivide, tmp, sigma+c);

end
