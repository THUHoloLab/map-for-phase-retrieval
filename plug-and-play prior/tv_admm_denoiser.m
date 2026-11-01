function output = tv_admm_denoiser(noisy, weight, iterations)
%TV_ADMM_DENOISER Complex isotropic-TV denoising with ADMM.

if nargin < 3, iterations = 20; end

rho = 2;
[m, n] = size(noisy);
[fy, fx] = ndgrid(0:m-1, 0:n-1);
denominator = 1 + rho * ( ...
    4 * sin(pi * fx / n).^2 + 4 * sin(pi * fy / m).^2);

output = noisy;
gx = zeros(size(noisy));
gy = gx;
ux = gx;
uy = gx;

for k = 1:iterations
    rhs = noisy + rho * ( ...
        circshift(gx - ux, [0, 1]) - (gx - ux) + ...
        circshift(gy - uy, [1, 0]) - (gy - uy));
    output = ifft2(fft2(rhs) ./ denominator);

    dx = circshift(output, [0, -1]) - output;
    dy = circshift(output, [-1, 0]) - output;
    qx = dx + ux;
    qy = dy + uy;

    magnitude = sqrt(abs(qx).^2 + abs(qy).^2);
    scale = max(1 - weight ./ (rho * max(magnitude, eps)), 0);
    gx = scale .* qx;
    gy = scale .* qy;

    ux = ux + dx - gx;
    uy = uy + dy - gy;
end
end
