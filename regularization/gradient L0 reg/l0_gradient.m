clc
clear

img = double(imread('cameraman.tif'))/255;

mom1 = 0;
mom2 = 0;


r1 = 0.9;
r2 = 0.999;
lr = 5e-4;

s = img;
dx = psf2otf([-1,1],size(s));
dy = psf2otf([-1;1],size(s));
dtd = abs(dx).^2 + abs(dy).^2;

o = s;
o_last = o;
lambda = 0.1;

f = @(x,a) a * exp(-a * abs(x)).* sign(x); 

for iter = 1:120
    
    gx = real(ifft2(fft2(o) .* dx));
    gy = real(ifft2(fft2(o) .* dy));
    sss = sqrt(gx.^2 + gy.^2 + 1e-5);

    gx = real(ifft2(fft2(f(gx,12)).*conj(dx)));
    gy = real(ifft2(fft2(f(gy,12)).*conj(dy)));

    grad = (o - s) + lambda * (gx + gy);

    mom1 = r1 * mom1 + (1 - r1) * grad;
    mom2 = r2 * mom2 + (1 - r2) * grad.^2;

    update = r1 * mom1 + (1 - r1) * grad;
    update = update ./ (sqrt(mom2) + 1e-5);

    o_last = o;
    o = o - lr * update;

    figure(121);
    imshow([o,mat2gray(log(sss + 1e-5))],[]);
    drawnow;



end
