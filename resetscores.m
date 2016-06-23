function newscore = resetscores(mos)

ind = find(mos~=0);
mos = mos(ind);
mos(mos>=85) = 84;
mos(mos<25) = 24;
newscore = round(mos./10) - 1; % for softmax
%     MOS = MOS/100; % for logistic

end