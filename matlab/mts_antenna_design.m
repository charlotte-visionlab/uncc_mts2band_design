close all;
clear;
clc;

Z0 = 376.73031366857;
speed_of_light= 299792458; % m / s
freq=17e9;
Z_avg = 190; % value of X parameter
M = 28;
k = 2*pi*freq/speed_of_light;
phi=0*pi/180;
stepsize = 0.3;
[x_grid, y_grid] = meshgrid([-10:stepsize:30],[-12.2:stepsize:12.2]);
x_grid = x_grid*1e-2;
y_grid = y_grid*1e-2;
r = sqrt(x_grid.^2 + y_grid.^2);
theta_L=60*pi/180;
n=Z0/Z_avg; % Z0/Z_avg
phi = 60*pi/180;
psi_rad = exp(j*k*x_grid*sin(theta_L)+j*phi);
psi_surf = exp(-j*k*n*r);
hold off
%plot(x, angle(psi_rad.*conj(psi_surf)));
%plot(x, real(psi_rad));
radial_amplitude = cos(k*n*r);
radiation_amplitude = cos(x_grid*k*sin(theta_L) + phi);
added_amplitude = cos(-k*n*r+k*x_grid*sin(theta_L));
Z_xy = 1j*(Z_avg+M*cos(-k*n*r+k*x_grid*sin(theta_L)));
subplot(311), imshow(radial_amplitude,[]);
subplot(312), imshow(radiation_amplitude,[]);
subplot(313), imshow(added_amplitude,[]);
max(Z_xy(:))
min(Z_xy(:))
