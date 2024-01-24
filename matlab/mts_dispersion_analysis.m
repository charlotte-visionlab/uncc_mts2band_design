clear;
close all;
clc;
filename.machine = {"visionlab34-pc","visionlab35-pc"};
filename.datetime ={"Dec11_19-37-48", ...
                    "Dec11_20-14-20", ... % GOOD parallel plate dispersion
                    "Dec11_20-38-24", ...
                    "Dec11_20-57-04", ...
                    "Dec11_21-09-38", ...
                    "Dec11_21-22-35", ...
                    "Dec12_15-35-15", ...
                    "Dec12_15-47-17", ... % GOOD parallel plate dispersion 1.0
                    "Dec12_15-57-05", ... % GOOD parallel plate dispersion 0.9
                    "Dec12_16-00-04", ... % GOOD parallel plate dispersion 0.5
                    "Dec12_21-15-11", ... % OK Sievenpiper Mushroom dispersion 
                    "Dec12_21-38-41", ... % GOOD Sievenpiper Mushroom dispersion 
                    "Dec13_15-39-30"
                    };
machine_index = 2;
datetime_index = length(filename.datetime);
data_filename = strcat("dispersion_data","_", ...
                        filename.datetime{datetime_index},"_", ...
                        filename.machine{machine_index},".mat");
data = load(data_filename);
dphase_x = abs(diff(data.phase_x));
dphase_y = abs(diff(data.phase_y));
dispersion_angle = [0, cumsum( max(dphase_x, dphase_y))];
dispersion_magnitude = real(data.mode_solutions/1e12);
dispersion_phase = angle(data.mode_solutions);
leak_factor = abs(data.q_solutions);
dispersion_angle = [dispersion_angle, dispersion_angle(end)+dphase_x(1)];
dispersion_magnitude = [dispersion_magnitude; dispersion_magnitude(1)];
dispersion_phase = [dispersion_phase; dispersion_phase(1)];
figure(1), plot(dispersion_angle, dispersion_magnitude,'bo-','LineWidth',3.0);
title("Brillouin Dispersion Diagram \\Gamma-X-M-\\Gamma - (0,180,360,540)")
xlabel("\\beta \\rho (deg)");
ylabel("Frequency (GHz)");
%ylim([0,4.5])
set(gca,'xtick',dispersion_angle(1:2:end));
xlim("tight");
legend(["TM mode (LH)"])
%figure(2), plot(dispersion_angle, dispersion_angle,'gx-','LineWidth',3.0);
%title("Brillouin Dispersion Diagram \\Gamma-X-M-\\Gamma - (0,180,360,540)")
%xlabel("\\phi Phase (deg)");
%ylabel("Frequency (GHz)");
%set(gca,'xtick',dispersion_angle(1:2:end));
%axis("tight");