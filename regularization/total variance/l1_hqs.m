% solving TV regularization using half quadratic splitting (HQS)

clc
clear


img = double(imread('cameraman.tif'))/255;


lambda     = 0.1;
lambda0    = lambda/2;

lambda_max = 1e9;
kappa      = 2;

dx = psf2otf([-1,1],size(img));
dy = psf2otf([-1;1],size(img));
dtd = abs(dx).^2 + abs(dy).^2;


s = img;
fft_s = fft2(s);
o = s;
while lambda0 < lambda_max

    % solving G-subproblem
    gx = real(ifft2(fft2(o) .* dx));
    gy = real(ifft2(fft2(o) .* dy));
    sss = sqrt(gx.^2 + gy.^2 + 1e-5);

    gx = gx./sss .* max(sss - lambda/lambda0,0);
    gy = gy./sss .* max(sss - lambda/lambda0,0);

    % solving o-subproblem
    fenzi = fft_s + lambda0 * (fft2(gx).*conj(dx) + fft2(gy).*conj(dy));
    fenmu = 1 + lambda0 * dtd;
    o = real(ifft2(fenzi./fenmu));

    figure(121);
    imshow([o,mat2gray(log(sss+1e-5))],[]);
    drawnow;
    lambda0 = lambda0 * kappa;

end