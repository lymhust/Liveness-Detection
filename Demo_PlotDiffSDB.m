clc; clear; close all;
ACCU = zeros(3,3,4);
Snum = [2 3 4];
Dnum = [3 4 6];
T = 4;

for i = 1:3
    for j = 1:3
        str = sprintf('LOW_S%d_D%d',Snum(i),Dnum(j))
        load(str);
        for k = 2:5
            fea = IMGFEATURE{k};
            lab = LABELS{k};
            ACCU(i,j,k-1) = Fun_Classification_DiffSDB(fea,lab,T)
        end
    end
end

[x,y,z] = meshgrid(1:3,1:3,1:3);
c = ACCU(:,:,1:3);
xs = 1:3;
ys = xs;
zs = xs;
h = slice(x,y,z,c,xs,ys,zs);
set(h,'FaceColor','interp',...
    'EdgeColor','none')
camproj perspective
box on
view(-70,70)
colormap hsv
colorbar
