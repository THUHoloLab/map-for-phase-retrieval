%DEMO_PNP_INLINE PnP reconstruction of a simulated inline hologram.

clear
close all

n = 128;
pixelSize = 1.12e-6;
wavelength = 532e-9;
distance = 1e-3;

[x, y] = meshgrid(linspace(-1, 1, n));
amplitude = 1 - 0.35 * exp(-20 * (x.^2 + y.^2));
phase = 1.2 * exp(-25 * ((x - 0.2).^2 + (y + 0.1).^2));
truth = amplitude .* exp(1i * phase);

detector = propagate_angular_spectrum( ...
    truth, distance, wavelength, pixelSize);
intensity = abs(detector).^2;

[admm, errorAdmm] = pnp_admm_inline( ...
    intensity, distance, wavelength, pixelSize, 100, 0.005);
[fista, errorFista] = pnp_fista_inline( ...
    intensity, distance, wavelength, pixelSize, 100, 0.005);
[red, errorRed] = red_inline( ...
    intensity, distance, wavelength, pixelSize, 100, 0.1);

figure
tiledlayout(2, 3, "Padding", "compact")
nexttile, imagesc(abs(truth)), axis image off, title("True amplitude")
nexttile, imagesc(abs(admm)), axis image off, title("PnP-ADMM")
nexttile, imagesc(abs(fista)), axis image off, title("PnP-FISTA")
nexttile, imagesc(angle(truth)), axis image off, title("True phase")
nexttile, imagesc(angle(red)), axis image off, title("RED")
nexttile
semilogy(errorAdmm)
hold on
semilogy(errorFista)
semilogy(errorRed)
grid on
xlabel("Iteration"), ylabel("Relative amplitude error")
legend("PnP-ADMM", "PnP-FISTA", "RED")
