function optstatus = fgo_gnss(datapath, setting, initflag)
%% Factor graph optimization using GNSS only
% Author: Taro Suzuki
arguments
    datapath string  % Dataset path
    setting table    % Setting data from setting_train.csv or setting_test.csv
    initflag = false % Initialization flag, default = false
end

%% Path
if exist("setup_local_env", "file")
    setup_local_env();
end
addpath ./functions/
if ispc
    addpath C:\'Program Files (x86)'\GTSAM\gtsam_toolbox\
else
    addpath /usr/local/gtsam_toolbox/
end

%% Load data
course = setting.Course; phone = setting.Phone;
fprintf('Course: %s, Phone: %s\n', course, phone);

% Load preprocessed smartphone data
load(datapath+course+"/"+phone+"/"+"phone_data.mat");

%% Setting
is = setting.IdxStart; % Start index for optimization
ie = setting.IdxEnd;   % End index for optimization
n = obs.n;             % Number of total epochs
nsat = obs.nsat;       % Number of satellites
FTYPE = ["L1","L5"];   % Frequency type

prm = parameters(setting, initflag); % Processing parameter

%% Initial position/velocity/clk/dclk/
if initflag
    % Single point positioning results
    posini = posbl.copy();
    velini = posbl.gradient(obs.dt);
    clk = [obs.clk zeros(n,6)];
    dclk = obs.dclk;
else
    % Previous estimation
    load(datapath+course+"/"+phone+"/"+"result_gnss.mat");
    posini = posest.copy();
    velini = velest.copy();
    clk = clkest;
    dclk = dclkest;
end

%% Compute residuals
% Exclude outliers
obsr = exobs(obs, prm);

% Observation residuals
satr = gt.Gsat(obsr, nav);
satr.setRcvPosVel(posini, velini);
obsr = obsr.residuals(satr);

% Exclude outliers from residuals
obsr = exobs_residuals(obsr, satr, clk(:,1), dclk, prm);
obsr = obsr.residuals(satr);

% ECEF to ENU
for j=1:nsat
    exyz = [-satr.ex(:,j) -satr.ey(:,j) -satr.ez(:,j)]; % Line-of-sight vector in ECEF
    eenu = rtklib.ecef2enu(exyz, posini.orgllh); % Line-of-sight vector in ENU
    ee(:,j) = eenu(:,1);
    en(:,j) = eenu(:,2);
    eu(:,j) = eenu(:,3);
end

%% Pseudorange compensation using base observation
for f=FTYPE
    if ~isempty(obsr.(f))
        % Pseudorange correction
        pc = correct_pseudorange(datapath, obsr, obsb, nav, f, prm);
        obsr.(f).resPc = obsr.(f).resPc-pc;
    end
end

%% Observation error model
obserr = obserrmodel(obsr,satr,prm);

%% Parameters for graph optimization
noise_sigmas = @gtsam.noiseModel.Diagonal.Sigmas;
noise_robust = @gtsam.noiseModel.Robust.Create;
sym = @gtsam.symbol;

% Initial state
x_ini  = posini.enu'; % x (position) in ENU
v_ini  = velini.enu'; % v (velocity) in ENU
c_ini  = clk'; % c (clock)
d_ini  = dclk'; % d (clock drift)

% Motion factor
sigma_motion = prm.sigma_motion*ones(3,1);
noise_motion = noise_sigmas(sigma_motion);

% Clock factor
noise_clk = noise_sigmas([prm.sigma_motion_clk; zeros(6,1)]);
noise_clkjump = noise_sigmas([Inf; zeros(6,1)]);

%% Graph Construction
% Create a factor graph container
graph = gtsam.NonlinearFactorGraph;

% gnss_gpu parity export hook: keep Taroz/GTSAM GNSS factors and metadata.
gnss_export_factors = {};
gnss_export_field = strings(0, 1);
gnss_export_factor_model = strings(0, 1);
gnss_export_freq = strings(0, 1);
gnss_export_epoch = zeros(0, 1);
gnss_export_utc = zeros(0, 1);
gnss_export_next_epoch = zeros(0, 1);
gnss_export_next_utc = zeros(0, 1);
gnss_export_sys = zeros(0, 1);
gnss_export_svid = zeros(0, 1);
gnss_export_sat_col = zeros(0, 1);
gnss_export_sigtype = zeros(0, 1);
gnss_export_sigma = zeros(0, 1);
gnss_export_measurement = zeros(0, 1);
gnss_export_dt = zeros(0, 1);
gnss_export_los = zeros(0, 3);
gnss_export_org1 = zeros(0, 3);
gnss_export_org2 = zeros(0, 3);

% Initial factor/state
initials = gtsam.Values;
for i=is:ie
    % Initial values
    initials.insert(sym('x',i), x_ini(:,i));
    initials.insert(sym('v',i), v_ini(:,i));
    initials.insert(sym('c',i), c_ini(:,i));
    initials.insert(sym('d',i), d_ini(:,i));

    % Initial factor
    graph.add(gtsam.PriorFactorVector(sym('x',i), x_ini(:,i), noise_sigmas(Inf*ones(3,1))));
    graph.add(gtsam.PriorFactorVector(sym('v',i), v_ini(:,i), noise_sigmas(Inf*ones(3,1))));
    graph.add(gtsam.PriorFactorVector(sym('c',i), c_ini(:,i), noise_sigmas(Inf*ones(7,1))));
    graph.add(gtsam.PriorFactorVector(sym('d',i), d_ini(:,i), noise_sigmas(Inf*ones(1,1))));
end

% Pseudorange/Doppler factor
for i=progress(is:ie)
    keyX = sym('x',i);
    keyV = sym('v',i);
    keyC = sym('c',i);
    keyD = sym('d',i);
    orgx = posini.enu(i,:)';
    orgv = velini.enu(i,:)';

    for j=1:nsat
        losvec = [ee(i,j) en(i,j) eu(i,j)]';
        for f=FTYPE
            if ~isempty(obsr.(f))
                sigtype = sysfreq2sigtype(obsr.sys,f);
                % Pseudorange factor
                if ~isnan(obsr.(f).resPc(i,j))
                    noise = noise_sigmas(obserr.(f).P(i,j));
                    noise_rubust = noise_robust(prm.P_kernel, noise);
                    gnss_factor = gtsam_gnss.PseudorangeFactor_XC(keyX, keyC, losvec, obsr.(f).resPc(i,j), sigtype(j), orgx, noise_rubust);
                    graph.add(gnss_factor);
                    gnss_export_factors{end + 1, 1} = gnss_factor;
                    gnss_export_field(end + 1, 1) = "P";
                    gnss_export_factor_model(end + 1, 1) = "XC";
                    gnss_export_freq(end + 1, 1) = string(f);
                    gnss_export_epoch(end + 1, 1) = i;
                    gnss_export_utc(end + 1, 1) = obs.utcms(i);
                    gnss_export_next_epoch(end + 1, 1) = 0;
                    gnss_export_next_utc(end + 1, 1) = 0;
                    gnss_export_sys(end + 1, 1) = obs.sys(j);
                    gnss_export_svid(end + 1, 1) = obs.prn(j);
                    gnss_export_sat_col(end + 1, 1) = j;
                    gnss_export_sigtype(end + 1, 1) = sigtype(j);
                    gnss_export_sigma(end + 1, 1) = obserr.(f).P(i,j);
                    gnss_export_measurement(end + 1, 1) = obsr.(f).resPc(i,j);
                    gnss_export_dt(end + 1, 1) = 0;
                    gnss_export_los(end + 1, :) = losvec';
                    gnss_export_org1(end + 1, :) = orgx';
                    gnss_export_org2(end + 1, :) = [NaN NaN NaN];
                end
                % Doppler factor
                if ~isnan(obsr.(f).resD(i,j))
                    noise = noise_sigmas(obserr.(f).D(i,j));
                    noise_rubust = noise_robust(prm.D_kernel, noise);
                    gnss_factor = gtsam_gnss.DopplerFactor_VD(keyV, keyD, losvec, obsr.(f).resD(i,j), orgv, noise_rubust);
                    graph.add(gnss_factor);
                    gnss_export_factors{end + 1, 1} = gnss_factor;
                    gnss_export_field(end + 1, 1) = "D";
                    gnss_export_factor_model(end + 1, 1) = "VD";
                    gnss_export_freq(end + 1, 1) = string(f);
                    gnss_export_epoch(end + 1, 1) = i;
                    gnss_export_utc(end + 1, 1) = obs.utcms(i);
                    gnss_export_next_epoch(end + 1, 1) = 0;
                    gnss_export_next_utc(end + 1, 1) = 0;
                    gnss_export_sys(end + 1, 1) = obs.sys(j);
                    gnss_export_svid(end + 1, 1) = obs.prn(j);
                    gnss_export_sat_col(end + 1, 1) = j;
                    gnss_export_sigtype(end + 1, 1) = 0;
                    gnss_export_sigma(end + 1, 1) = obserr.(f).D(i,j);
                    gnss_export_measurement(end + 1, 1) = obsr.(f).resD(i,j);
                    gnss_export_dt(end + 1, 1) = 0;
                    gnss_export_los(end + 1, :) = losvec';
                    gnss_export_org1(end + 1, :) = orgv';
                    gnss_export_org2(end + 1, :) = [NaN NaN NaN];
                end
            end
        end
    end
end

% Motion/Clock/IMU/TDCP factor
for i=progress(is:ie-1)
    keyX1 = sym('x',i); keyX2 = sym('x',i+1);
    keyV1 = sym('v',i); keyV2 = sym('v',i+1);
    keyC1 = sym('c',i); keyC2 = sym('c',i+1);
    keyD1 = sym('d',i); keyD2 = sym('d',i+1);
    orgx1 = posini.enu(i,:)';
    orgx2 = posini.enu(i+1,:)';

    % Time difference
    dtgps = (obs.utcms(i+1)-obs.utcms(i))/1000;

    if dtgps<prm.time_diff_th
        % Motion factor
        graph.add(gtsam_gnss.MotionFactor_XXVV(keyX1, keyX2, keyV1, keyV2, dtgps, noise_motion));

        % Clock factor
        if ~ismember(phone,["sm-a205u","sm-a505u","samsunga325g"])
            if obs.clkjump(i+1)
                graph.add(gtsam_gnss.ClockFactor_CCDD(keyC1, keyC2, keyD1, keyD2, dtgps, noise_clkjump));
            else
                graph.add(gtsam_gnss.ClockFactor_CCDD(keyC1, keyC2, keyD1, keyD2, dtgps, noise_clk));
            end
        end
    end
    if ~ismember(setting.Phone,["sm-a325f","samsunga32"])
        for j=1:nsat
            losvec = [ee(i,j),en(i,j),eu(i,j)]';
            for f=FTYPE
                if ~isempty(obsr.(f))
                    % TDCP factor
                    if ~isnan(obsr.(f).resL(i,j)) && ~isnan(obsr.(f).resL(i+1,j)) && ~obs.clkjump(i+1)
                        tdcp =  obsr.(f).resL(i+1,j)-obsr.(f).resL(i,j);
                        noise = noise_sigmas(obserr.(f).L(i,j));
                        noise_rubust = noise_robust(prm.L_kernel, noise);
                        if ismember(phone,["sm-a205u","sm-a217m","sm-a505g","sm-a600t","sm-a505u"])
                            tdcp_measurement = tdcp + prm.Loffset;
                            gnss_factor = gtsam_gnss.TDCPFactor_XXDD(keyX1, keyX2, keyD1, keyD2, losvec, tdcp_measurement, dtgps, orgx1, orgx2, noise_rubust);
                            graph.add(gnss_factor);
                            gnss_export_factors{end + 1, 1} = gnss_factor;
                            gnss_export_field(end + 1, 1) = "L";
                            gnss_export_factor_model(end + 1, 1) = "XXDD";
                            gnss_export_freq(end + 1, 1) = string(f);
                            gnss_export_epoch(end + 1, 1) = i;
                            gnss_export_utc(end + 1, 1) = obs.utcms(i);
                            gnss_export_next_epoch(end + 1, 1) = i + 1;
                            gnss_export_next_utc(end + 1, 1) = obs.utcms(i + 1);
                            gnss_export_sys(end + 1, 1) = obs.sys(j);
                            gnss_export_svid(end + 1, 1) = obs.prn(j);
                            gnss_export_sat_col(end + 1, 1) = j;
                            gnss_export_sigtype(end + 1, 1) = 0;
                            gnss_export_sigma(end + 1, 1) = obserr.(f).L(i,j);
                            gnss_export_measurement(end + 1, 1) = tdcp_measurement;
                            gnss_export_dt(end + 1, 1) = dtgps;
                            gnss_export_los(end + 1, :) = losvec';
                            gnss_export_org1(end + 1, :) = orgx1';
                            gnss_export_org2(end + 1, :) = orgx2';
                        elseif ismember(phone,"samsunga325g")
                            tdcp_measurement = tdcp;
                            gnss_factor = gtsam_gnss.TDCPFactor_XXDD(keyX1, keyX2, keyD1, keyD2, losvec, tdcp_measurement, dtgps, orgx1, orgx2, noise_rubust);
                            graph.add(gnss_factor);
                            gnss_export_factors{end + 1, 1} = gnss_factor;
                            gnss_export_field(end + 1, 1) = "L";
                            gnss_export_factor_model(end + 1, 1) = "XXDD";
                            gnss_export_freq(end + 1, 1) = string(f);
                            gnss_export_epoch(end + 1, 1) = i;
                            gnss_export_utc(end + 1, 1) = obs.utcms(i);
                            gnss_export_next_epoch(end + 1, 1) = i + 1;
                            gnss_export_next_utc(end + 1, 1) = obs.utcms(i + 1);
                            gnss_export_sys(end + 1, 1) = obs.sys(j);
                            gnss_export_svid(end + 1, 1) = obs.prn(j);
                            gnss_export_sat_col(end + 1, 1) = j;
                            gnss_export_sigtype(end + 1, 1) = 0;
                            gnss_export_sigma(end + 1, 1) = obserr.(f).L(i,j);
                            gnss_export_measurement(end + 1, 1) = tdcp_measurement;
                            gnss_export_dt(end + 1, 1) = dtgps;
                            gnss_export_los(end + 1, :) = losvec';
                            gnss_export_org1(end + 1, :) = orgx1';
                            gnss_export_org2(end + 1, :) = orgx2';
                        else
                            tdcp_measurement = tdcp;
                            gnss_factor = gtsam_gnss.TDCPFactor_XXCC(keyX1, keyX2, keyC1, keyC2, losvec, tdcp_measurement, orgx1, orgx2, noise_rubust);
                            graph.add(gnss_factor);
                            gnss_export_factors{end + 1, 1} = gnss_factor;
                            gnss_export_field(end + 1, 1) = "L";
                            gnss_export_factor_model(end + 1, 1) = "XXCC";
                            gnss_export_freq(end + 1, 1) = string(f);
                            gnss_export_epoch(end + 1, 1) = i;
                            gnss_export_utc(end + 1, 1) = obs.utcms(i);
                            gnss_export_next_epoch(end + 1, 1) = i + 1;
                            gnss_export_next_utc(end + 1, 1) = obs.utcms(i + 1);
                            gnss_export_sys(end + 1, 1) = obs.sys(j);
                            gnss_export_svid(end + 1, 1) = obs.prn(j);
                            gnss_export_sat_col(end + 1, 1) = j;
                            gnss_export_sigtype(end + 1, 1) = 0;
                            gnss_export_sigma(end + 1, 1) = obserr.(f).L(i,j);
                            gnss_export_measurement(end + 1, 1) = tdcp_measurement;
                            gnss_export_dt(end + 1, 1) = dtgps;
                            gnss_export_los(end + 1, :) = losvec';
                            gnss_export_org1(end + 1, :) = orgx1';
                            gnss_export_org2(end + 1, :) = orgx2';
                        end
                    end
                end
            end
        end
    end
end

%% Optimization
optparameters = gtsam.LevenbergMarquardtParams;
optparameters.setVerbosity('TERMINATION');
optparameters.setMaxIterations(1000);
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initials, optparameters);

% Optimize!
disp('optimization... ');
fprintf('Initial Error: %.2f\n',optimizer.error);
tic;
results = optimizer.optimize();
fprintf('Error: %.2f Iter: %d\n',optimizer.error,optimizer.iterations);
toc;
optstatus.OptTime = toc;
optstatus.OptIter = optimizer.iterations;
optstatus.OptError = optimizer.error;

% gnss_gpu parity export hook: write Taroz/GTSAM GNSS factor residuals.
if exist("gnss_export_factors", "var") && ~isempty(gnss_export_factors)
    gnss_export_trip_dir = datapath + course + "/" + phone;
    gnss_export_mask_file = fullfile(gnss_export_trip_dir, "phone_data_gnss_factor_mask.csv");
    gnss_export_residual_file = fullfile(gnss_export_trip_dir, "phone_data_gnss_factor_residuals.csv");
    gnss_export_summary_file = fullfile(gnss_export_trip_dir, "phone_data_gnss_factor_summary.csv");
    gnss_export_graph_state_file = fullfile(gnss_export_trip_dir, "phone_data_gnss_graph_state.csv");
    gnss_export_initial_state_file = fullfile(gnss_export_trip_dir, "phone_data_gnss_initial_state.csv");

    gnss_state_epoch_col = (is:ie)';
    gnss_state_utc_col = zeros(numel(gnss_state_epoch_col), 1);
    gnss_state_position_col = zeros(numel(gnss_state_epoch_col), 3);
    gnss_state_rpy_col = NaN(numel(gnss_state_epoch_col), 3);
    gnss_state_velocity_col = zeros(numel(gnss_state_epoch_col), 3);
    gnss_state_clock_col = zeros(numel(gnss_state_epoch_col), 7);
    gnss_state_drift_col = zeros(numel(gnss_state_epoch_col), 1);
    for gnss_state_idx = 1:numel(gnss_state_epoch_col)
        gnss_state_epoch = gnss_state_epoch_col(gnss_state_idx);
        gnss_state_utc_col(gnss_state_idx) = obs.utcms(gnss_state_epoch);
        gnss_state_position_col(gnss_state_idx, :) = results.atVector(sym('x', gnss_state_epoch))';
        try
            gnss_state_pose = results.atPose3(sym('p', gnss_state_epoch));
            gnss_state_rpy_col(gnss_state_idx, :) = gnss_state_pose.rotation.rpy';
        catch
        end
        gnss_state_velocity_col(gnss_state_idx, :) = results.atVector(sym('v', gnss_state_epoch))';
        gnss_state_clock = results.atVector(sym('c', gnss_state_epoch))';
        gnss_state_clock_col(gnss_state_idx, 1:numel(gnss_state_clock)) = gnss_state_clock;
        gnss_state_drift_col(gnss_state_idx) = results.atVector(sym('d', gnss_state_epoch));
    end
    gnss_graph_state_table = table(gnss_state_epoch_col, gnss_state_utc_col, ...
        gnss_state_position_col(:, 1), gnss_state_position_col(:, 2), gnss_state_position_col(:, 3), ...
        gnss_state_rpy_col(:, 1), gnss_state_rpy_col(:, 2), gnss_state_rpy_col(:, 3), ...
        gnss_state_velocity_col(:, 1), gnss_state_velocity_col(:, 2), gnss_state_velocity_col(:, 3), ...
        gnss_state_clock_col(:, 1), gnss_state_clock_col(:, 2), gnss_state_clock_col(:, 3), ...
        gnss_state_clock_col(:, 4), gnss_state_clock_col(:, 5), gnss_state_clock_col(:, 6), ...
        gnss_state_clock_col(:, 7), gnss_state_drift_col, ...
        'VariableNames', {'epoch_index', 'utcTimeMillis', ...
        'position_x', 'position_y', 'position_z', 'roll', 'pitch', 'yaw', ...
        'velocity_x', 'velocity_y', 'velocity_z', ...
        'clock_bias_m_0', 'clock_bias_m_1', 'clock_bias_m_2', 'clock_bias_m_3', ...
        'clock_bias_m_4', 'clock_bias_m_5', 'clock_bias_m_6', 'clock_drift_mps'});
    writetable(gnss_graph_state_table, gnss_export_graph_state_file);

    gnss_initial_utc_col = zeros(numel(gnss_state_epoch_col), 1);
    gnss_initial_position_col = zeros(numel(gnss_state_epoch_col), 3);
    gnss_initial_rpy_col = NaN(numel(gnss_state_epoch_col), 3);
    gnss_initial_velocity_col = zeros(numel(gnss_state_epoch_col), 3);
    gnss_initial_clock_col = zeros(numel(gnss_state_epoch_col), 7);
    gnss_initial_drift_col = zeros(numel(gnss_state_epoch_col), 1);
    for gnss_state_idx = 1:numel(gnss_state_epoch_col)
        gnss_state_epoch = gnss_state_epoch_col(gnss_state_idx);
        gnss_initial_utc_col(gnss_state_idx) = obs.utcms(gnss_state_epoch);
        gnss_initial_position_col(gnss_state_idx, :) = initials.atVector(sym('x', gnss_state_epoch))';
        try
            gnss_initial_pose = initials.atPose3(sym('p', gnss_state_epoch));
            gnss_initial_rpy_col(gnss_state_idx, :) = gnss_initial_pose.rotation.rpy';
        catch
        end
        gnss_initial_velocity_col(gnss_state_idx, :) = initials.atVector(sym('v', gnss_state_epoch))';
        gnss_initial_clock = initials.atVector(sym('c', gnss_state_epoch))';
        gnss_initial_clock_col(gnss_state_idx, 1:numel(gnss_initial_clock)) = gnss_initial_clock;
        gnss_initial_drift_col(gnss_state_idx) = initials.atVector(sym('d', gnss_state_epoch));
    end
    gnss_initial_state_table = table(gnss_state_epoch_col, gnss_initial_utc_col, ...
        gnss_initial_position_col(:, 1), gnss_initial_position_col(:, 2), gnss_initial_position_col(:, 3), ...
        gnss_initial_rpy_col(:, 1), gnss_initial_rpy_col(:, 2), gnss_initial_rpy_col(:, 3), ...
        gnss_initial_velocity_col(:, 1), gnss_initial_velocity_col(:, 2), gnss_initial_velocity_col(:, 3), ...
        gnss_initial_clock_col(:, 1), gnss_initial_clock_col(:, 2), gnss_initial_clock_col(:, 3), ...
        gnss_initial_clock_col(:, 4), gnss_initial_clock_col(:, 5), gnss_initial_clock_col(:, 6), ...
        gnss_initial_clock_col(:, 7), gnss_initial_drift_col, ...
        'VariableNames', {'epoch_index', 'utcTimeMillis', ...
        'position_x', 'position_y', 'position_z', 'roll', 'pitch', 'yaw', ...
        'velocity_x', 'velocity_y', 'velocity_z', ...
        'clock_bias_m_0', 'clock_bias_m_1', 'clock_bias_m_2', 'clock_bias_m_3', ...
        'clock_bias_m_4', 'clock_bias_m_5', 'clock_bias_m_6', 'clock_drift_mps'});
    writetable(gnss_initial_state_table, gnss_export_initial_state_file);

    gnss_initial_residual_col = NaN(numel(gnss_export_factors), 1);
    gnss_residual_col = NaN(numel(gnss_export_factors), 1);
    gnss_initial_factor_error_col = NaN(numel(gnss_export_factors), 1);
    gnss_factor_error_col = NaN(numel(gnss_export_factors), 1);
    for gnss_export_idx = 1:numel(gnss_export_factors)
        gnss_epoch = gnss_export_epoch(gnss_export_idx);
        gnss_next_epoch = gnss_export_next_epoch(gnss_export_idx);
        if gnss_export_field(gnss_export_idx) == "P"
            gnss_initial_error = gnss_export_factors{gnss_export_idx}.evaluateError( ...
                initials.atVector(sym('x', gnss_epoch)), ...
                initials.atVector(sym('c', gnss_epoch)));
            gnss_error = gnss_export_factors{gnss_export_idx}.evaluateError( ...
                results.atVector(sym('x', gnss_epoch)), ...
                results.atVector(sym('c', gnss_epoch)));
        elseif gnss_export_field(gnss_export_idx) == "D"
            gnss_initial_error = gnss_export_factors{gnss_export_idx}.evaluateError( ...
                initials.atVector(sym('v', gnss_epoch)), ...
                initials.atVector(sym('d', gnss_epoch)));
            gnss_error = gnss_export_factors{gnss_export_idx}.evaluateError( ...
                results.atVector(sym('v', gnss_epoch)), ...
                results.atVector(sym('d', gnss_epoch)));
        elseif gnss_export_factor_model(gnss_export_idx) == "XXDD"
            gnss_initial_error = gnss_export_factors{gnss_export_idx}.evaluateError( ...
                initials.atVector(sym('x', gnss_epoch)), ...
                initials.atVector(sym('x', gnss_next_epoch)), ...
                initials.atVector(sym('d', gnss_epoch)), ...
                initials.atVector(sym('d', gnss_next_epoch)));
            gnss_error = gnss_export_factors{gnss_export_idx}.evaluateError( ...
                results.atVector(sym('x', gnss_epoch)), ...
                results.atVector(sym('x', gnss_next_epoch)), ...
                results.atVector(sym('d', gnss_epoch)), ...
                results.atVector(sym('d', gnss_next_epoch)));
        else
            gnss_initial_error = gnss_export_factors{gnss_export_idx}.evaluateError( ...
                initials.atVector(sym('x', gnss_epoch)), ...
                initials.atVector(sym('x', gnss_next_epoch)), ...
                initials.atVector(sym('c', gnss_epoch)), ...
                initials.atVector(sym('c', gnss_next_epoch)));
            gnss_error = gnss_export_factors{gnss_export_idx}.evaluateError( ...
                results.atVector(sym('x', gnss_epoch)), ...
                results.atVector(sym('x', gnss_next_epoch)), ...
                results.atVector(sym('c', gnss_epoch)), ...
                results.atVector(sym('c', gnss_next_epoch)));
        end
        gnss_initial_residual_col(gnss_export_idx) = gnss_initial_error(1);
        gnss_residual_col(gnss_export_idx) = gnss_error(1);
        try
            gnss_initial_factor_error_col(gnss_export_idx) = gnss_export_factors{gnss_export_idx}.error(initials);
            gnss_factor_error_col(gnss_export_idx) = gnss_export_factors{gnss_export_idx}.error(results);
        catch
            gnss_initial_factor_error_col(gnss_export_idx) = NaN;
            gnss_factor_error_col(gnss_export_idx) = NaN;
        end
    end

    gnss_factor_mask = table(gnss_export_field, gnss_export_freq, gnss_export_epoch, ...
        gnss_export_utc, gnss_export_next_epoch, gnss_export_next_utc, ...
        gnss_export_sys, gnss_export_svid, gnss_export_sat_col, gnss_export_factor_model, ...
        gnss_export_sigtype, gnss_export_sigma, gnss_export_measurement, gnss_export_dt, ...
        gnss_export_los(:, 1), gnss_export_los(:, 2), gnss_export_los(:, 3), ...
        gnss_export_org1(:, 1), gnss_export_org1(:, 2), gnss_export_org1(:, 3), ...
        gnss_export_org2(:, 1), gnss_export_org2(:, 2), gnss_export_org2(:, 3), ...
        'VariableNames', {'field', 'freq', 'epoch_index', 'utcTimeMillis', ...
        'next_epoch_index', 'nextUtcTimeMillis', 'sys', 'svid', 'sat_col', ...
        'factor_model', 'sigtype', 'sigma', 'measurement', 'dt_s', ...
        'los_e', 'los_n', 'los_u', 'origin1_e', 'origin1_n', 'origin1_u', ...
        'origin2_e', 'origin2_n', 'origin2_u'});
    writetable(gnss_factor_mask, gnss_export_mask_file);

    gnss_factor_residuals = table(gnss_export_field, gnss_export_freq, gnss_export_epoch, ...
        gnss_export_utc, gnss_export_next_epoch, gnss_export_next_utc, ...
        gnss_export_sys, gnss_export_svid, gnss_export_sat_col, gnss_export_factor_model, ...
        gnss_export_sigtype, gnss_export_sigma, gnss_export_measurement, gnss_export_dt, ...
        gnss_initial_residual_col, gnss_residual_col, ...
        gnss_residual_col - gnss_initial_residual_col, ...
        gnss_initial_factor_error_col, gnss_factor_error_col, ...
        'VariableNames', {'field', 'freq', 'epoch_index', 'utcTimeMillis', ...
        'next_epoch_index', 'nextUtcTimeMillis', 'sys', 'svid', 'sat_col', ...
        'factor_model', 'sigtype', 'sigma', 'measurement', 'dt_s', ...
        'initial_residual', 'residual', 'residual_delta', ...
        'initial_factor_error', 'factor_error'});
    writetable(gnss_factor_residuals, gnss_export_residual_file);

    try
        gnss_export_initial_graph_error = graph.error(initials);
        gnss_export_final_graph_error = graph.error(results);
    catch
        gnss_export_initial_graph_error = NaN;
        gnss_export_final_graph_error = optimizer.error;
    end
    gnss_factor_count = numel(gnss_export_factors);
    gnss_p_count = sum(gnss_export_field == "P");
    gnss_d_count = sum(gnss_export_field == "D");
    gnss_l_count = sum(gnss_export_field == "L");
    gnss_iteration_count = optimizer.iterations;
    gnss_summary = table(gnss_factor_count, gnss_p_count, gnss_d_count, gnss_l_count, ...
        gnss_iteration_count, gnss_export_initial_graph_error, gnss_export_final_graph_error, ...
        'VariableNames', {'factor_count', 'p_count', 'd_count', 'l_count', ...
        'iterations', 'initial_graph_error', 'final_graph_error'});
    writetable(gnss_summary, gnss_export_summary_file);
end

% Retrieving the estimated value
xest = NaN(n,3);
vest = NaN(n,3);
clkest = NaN(n,7);
dclkest = NaN(n,1);
for i=is:ie
    xest(i,:) = results.atVector(sym('x',i))';
    vest(i,:) = results.atVector(sym('v',i))';
    clkest(i,:) = results.atVector(sym('c',i))';
    dclkest(i,:) = results.atVector(sym('d',i))';
end

% Estimated position/velocity
posest = gt.Gpos(xest,'enu',posini.orgllh,'llh');
velest = gt.Gvel(vest,'enu',posini.orgllh,'llh');

%% Add position offset
rpy = vel2rpy(velest.enu, prm);
posest = add_position_offset(posest, rpy, phone);

%% Plot results
plot_eststate(clkest);

% Plot score
if contains(datapath,'train')
    load(datapath+course+"/"+phone+"/"+"gt.mat");
    optstatus.Score =  plot_score(posest, posbl, posgt);
else
    optstatus.Score = NaN;
end

%% Save results
fname = datapath+course+"/"+phone+"/"+"result_gnss.mat";
save(fname,"posest","clkest","velest","dclkest");
