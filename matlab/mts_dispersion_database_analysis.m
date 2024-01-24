clear;
close all;
clc;
filename.machine = {"visionlab34-pc","visionlab35-pc", ...
                    "ece-emag1", "ece-emag2", "ece-emag3", ...
                    "ENGR-MENCAGLI","ENGR-MENCAGLI2","ENGR-MENCAGLI3"};
filename.datetime ={"Dec14_12-59-11", ...
                    "Dec14_19-29-18", ...
                    "Dec15_20-29-28", ...
                    "Dec16_07-57-12", ...           % SQUARE PATCH
                    "Dec16_10-44-23", ...           % ELLIPTICAL DISC
                    "Dec16_10-52-18"                % CIRCULAR DISC
                    };
filename.prefix = "mts_databases/mts_dispersion_database";
machine_index = 3;
datetime_index = length(filename.datetime)-2;
%datetime_index = length(filename.datetime);
data_filename = strcat(filename.prefix, "_", ...
                        filename.datetime{datetime_index} , "_", ...
                        filename.machine{machine_index}, ".mat");
data = load(data_filename);
num_cell_designs = length(data.database);
legend_strs=cell(1,num_cell_designs);
for cell_index=1:num_cell_designs
  dphase_x = abs(diff(data.database{cell_index}.phase_x));
  dphase_y = abs(diff(data.database{cell_index}.phase_y));
  dispersion_angle = [0, cumsum( max(dphase_x, dphase_y))];
  dispersion_magnitude = real(data.database{cell_index}.mode_solutions/1e9);
  dispersion_phase = angle(data.database{cell_index}.mode_solutions);
  leak_factor = abs(data.database{cell_index}.q_solutions);
  dispersion_angle = [dispersion_angle, dispersion_angle(end)+dphase_x(1)];
  dispersion_magnitude = [dispersion_magnitude; dispersion_magnitude(1)];
  dispersion_phase = [dispersion_phase; dispersion_phase(1)];
  figure(1), plot(dispersion_angle, dispersion_magnitude,'LineWidth',3.0);
  hold on;
  if (cell_index==1)
    if (exist('OCTAVE_VERSION'))
      title("Brillouin Dispersion Diagram \\Gamma-X-M-\\Gamma - (0,180,360,540)")
    else
      title("Brillouin Dispersion Diagram $\Gamma$-X-M-$\Gamma$ - (0,180,360,540)",'Interpreter','latex')
    end
    
    xlabel("\\beta \\rho (deg)");
    ylabel("Frequency (GHz)");
  %ylim([0,4.5])
    set(gca,'xtick',dispersion_angle(1:2:end));
  end
  xlim("tight");
  legend_strs{cell_index} = sprintf("%0.2f mm",data.database{cell_index}.cell_size(1));
end
%legend(["TM mode (LH)"])
legend(legend_strs)

design_frequency = 15e9;
cell_size = 3e-3;
dielectric_relative_permeability = 2.2

for cell_idx=1:length(data.database)
  cell_data = data.database{cell_idx};
  x_phase_vals = cell_data.phase_x;
  y_phase_vals = cell_data.phase_y;
  cell_size(cell_idx) = cell_data.cell_size(1);
  phase_step = x_phase_vals(2) - x_phase_vals(1);
  resonant_frequencies = data.database{cell_idx}.mode_solutions(1:35);
  brillioun_phase = phase_step*(1:length(resonant_frequencies)); 
  phase_interp(cell_idx) = interp1(resonant_frequencies, brillioun_phase, design_frequency);
end
aaa = 1
delta_phi_deg = diff(phase_interp);
speed_of_light= 299792458 % m / s
% Z_s = sqrt((speed_of_light*delta_phi_deg/(2*pi*design_frequency*3.0e-3))-1)
epsilon0=8.854187812813e-12
mu0=1.2566370621219e-6
Z0 = 376.73031366857
k0=1/sqrt(epsilon0*mu0)

light_line = 2*pi*design_frequency/speed_of_light
Z_s = sqrt(dielectric_relative_permeability)*sqrt((real(phase_interp)*speed_of_light./(2*pi*design_frequency*3e-3)).^2-1)
figure(2);
plot(cell_size, Z_s)

%figure(2), plot(dispersion_angle, dispersion_angle,'gx-','LineWidth',3.0);
%title("Brillouin Dispersion Diagram \\Gamma-X-M-\\Gamma - (0,180,360,540)")
%xlabel("\\phi Phase (deg)");
%ylabel("Frequency (GHz)");
%set(gca,'xtick',dispersion_angle(1:2:end));
%axis("tight");
max_freq = max(real(resonant_frequencies))
num_steps = 100
freq_samples = linspace(0:max_freq/(num_steps-1):max_freq);
light_line = 2*pi*freq_samples/speed_of_light;

coeffs = polyfit([10;20;30], dispersion_magnitude(1:3)*1e9, 1)/1e9
scalef = coeffs(1)*speed_of_light/(2*pi)