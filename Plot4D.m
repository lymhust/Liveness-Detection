xs = 1:1:128;
ys = 1:1:128;
zs = 1:1:64;
c = ans;
[x,y,z] = meshgrid(xs,ys,zs);
%c = x.^2+y.^2+z.^2;
%c(7:15,7:15,13:21)=NaN;
h = slice(x,y,z,c,xs,ys,zs);
set(h,'FaceColor','interp',...
    'EdgeColor','none')
camproj perspective
box on
view(-70,70)
colormap hsv
colorbar