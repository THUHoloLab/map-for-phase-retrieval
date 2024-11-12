% solving TV regularization using ADMM (A.K.A. Split Bregman)

clc
clear


img = double(imread('cameraman.tif'))/255;


lambda  = 0.1;
lambda0 = 1;

dx = psf2otf([-1,1],size(img));
dy = psf2otf([-1;1],size(img));
dtd = abs(dx).^2 + abs(dy).^2;


s = img;
fft_s = fft2(s);
o = s;

bx = 0; % admm multiplier
by = 0; % admm multiplier

gx = 0;
gy = 0;

for iter = 1:300

    % solving o-subproblem
    fenzi = fft_s + lambda0 * (fft2(gx - bx).*conj(dx) + ...
                               fft2(gy - by).*conj(dy));

    fenmu = 1 + lambda0 * dtd;
    o = real(ifft2(fenzi./fenmu));

    % solving G-subproblem
    gx_old = real(ifft2(fft2(o) .* dx));
    gy_old = real(ifft2(fft2(o) .* dy));
   
    temp_gx = gx_old + bx;
    temp_gy = gy_old + by;
    sss = sqrt(temp_gx.^2 + temp_gy.^2 + 1e-5);

    gx = temp_gx./sss .* max(sss - lambda/lambda0,0);
    gy = temp_gy./sss .* max(sss - lambda/lambda0,0);

    % solving multiplier subproblem
    bx = bx + gx_old - gx;
    by = by + gy_old - gy;

    figure(121);
    sss_temp = sqrt(gx_old.^2 + gy_old.^2 + 1e-5);
    imshow([o,mat2gray(log(sss_temp+1e-5))],[]);
    drawnow;

end