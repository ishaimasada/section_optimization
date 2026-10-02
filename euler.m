clear; clc; close all;


%% Functions

% Load Bump Mesh
function [x, y, delta_x, delta_y, ni, nj] = read_plot3d_2d(filename)
    fid = fopen(filename, 'r');
    if fid == -1; error('Could not open file: %s', filename); end

    % Read grid dimensions
    dims = fscanf(fid, '%d', 2);
    ni = dims(1);
    nj = dims(2);
    fprintf('Grid Size: %d x %d\n', ni, nj);

    % Read coordinates (x first, then y)
    x_data = fscanf(fid, '%f', ni * nj);
    y_data = fscanf(fid, '%f', ni * nj);
    fclose(fid);

    % Reshape into (i,j) arrays  [i = faster index, consistent with PLOT3D]
    x = reshape(x_data, [ni, nj]);
    y = reshape(y_data, [ni, nj]);

    % Spatial Steps
    delta_x = zeros(ni-1, 1);
    delta_y = zeros(nj-1, 1);
    for i=1: ni-1; delta_x(i) = x(i+1, 1) - x(i, 1); end
    for i=1: nj-1; delta_y(i) = y(1, i+1) - y(1, i); end
end


% Conversion from primitive state vector to conservative state vector
function conservative = get_conservative(primitive, Cv, R)
    v = primitive;
    u1 = v(1);
    u2 = v(1) * v(2);
    u3 = v(1) * v(3);
    u4 = Cv*v(4)/R + (v(1)/2) * (v(2)^2 + v(3)^2);
    conservative = [u1, u2, u3, u4];
end


% Conversion from conservative state vector to primitive state vector
function primitive = get_primitive(conservative, gamma)
    u = conservative;
    v1 = u(1);
    v2 = u(2) / u(1);          % FIX #1: was u(1)/u(2); correct is (rho*Vx)/rho
    v3 = u(3) / u(1);
    v4 = (u(4) - (1/2)*((u(2)^2/u(1)) + (u(3)^2/u(1)))) * (gamma - 1);
    primitive = [v1, v2, v3, v4];
end


% (Velocity Inlet) Boundary Condition
function [u1, V1] = velocity_inlet_BC(u_inf, v_inf, rho_inf, V2, Cv, R)
    % Subsonic inlet (P1 = P2)
    P2 = V2(1,1,4);
    Vx1 = u_inf;
    Vy1 = v_inf;
    rho1 = rho_inf;
    P1 = P2;
    V1 = [rho1, Vx1, Vy1, P1];
    u1 = get_conservative(V1, Cv, R);
end


% (Pressure Outlet) Boundary Condition
function [u1, V1] = pressure_outlet_BC(V2, P_inf, Cv, R)
    rho1 = V2(1);
    Vx1  = V2(2);
    Vy1  = V2(3);
    P1   = P_inf;
    V1 = [rho1, Vx1, Vy1, P1];
    u1 = get_conservative(V1, Cv, R);
end


% (Inviscid Slip Wall / Symmetry) Boundary Condition
function [u1, V1] = wall_BC(V2, eta_x, eta_y, Cv, R)
    % V2 = interior point (i,2)
    % eta_x, eta_y = metrics at the wall point (i,1)
    rho2 = V2(1);
    u2   = V2(2);
    v2   = V2(3);
    P2   = V2(4);
    % wall normal (pointing into the domain for a bottom wall)
    % n = (eta_x, eta_y) / |grad eta|
    norm_factor = sqrt(eta_x^2 + eta_y^2);
    nx = eta_x / norm_factor;
    ny = eta_y / norm_factor;
    % remove normal component of velocity
    Vn = u2*nx + v2*ny;
    u1_vel = u2 - Vn*nx;
    v1_vel = v2 - Vn*ny;
    % thermodynamic variables extrapolated
    rho1 = rho2;
    P1   = P2;
    % updated states
    V1 = [rho1, u1_vel, v1_vel, P1];
    u1 = get_conservative(V1, Cv, R);
end


% (Pressure Farfield) Boundary Condition
function [u1, V1] = farfield_BC(u_inf, v_inf, P_inf, T_inf, Cv, R)
    Vx1  = u_inf;
    Vy1  = v_inf;
    rho1 = P_inf / (T_inf*R);
    P1   = P_inf;
    V1 = [rho1, Vx1, Vy1, P1];
    u1 = get_conservative(V1, Cv, R);
end


% Display Residuals
function display_residuals(L2_norm)
    fprintf('RESIDUALS\n');
    fprintf('Continuity = %.3e\n', L2_norm(1));
    fprintf('X-Momentum = %.3e\n', L2_norm(2));
    fprintf('Y-Momentum = %.3e\n', L2_norm(3));
    fprintf('Energy = %.3e\n', L2_norm(4));
end


% Calculate Flux Vectors
function [A,B] = solve_fluxes(v, freestream)
    rho = v(1);
    Vx = v(2);
    Vy = v(3);
    P = v(4);
    R = freestream.R;
    Cp = freestream.cp;
    ht = Cp*P/(R*rho) + 0.5*(Vx^2 + Vy^2);  % local total enthalpy
    A = [rho*Vx, rho*Vx^2+P, rho*Vx*Vy, rho*ht*Vx];
    B = [rho*Vy, rho*Vx*Vy, rho*Vy^2+P, rho*ht*Vy];

    if any(A == inf,"all") || any(isnan(A),"all") || any(B == inf,"all") || any(isnan(B),"all")
        error("Invalid Elements (Inf or NaN)");
    end
end


% Update Boundaries
function [u, V] = update_boundaries(V, u, freestream, num_i, num_j, eta_x, eta_y)
    u_inf = freestream.u;
    v_inf = freestream.v;
    rho_inf = freestream.rho;
    P_inf = freestream.P;
    T_inf = freestream.T;
    Cv = freestream.cv;
    R = freestream.R;

    for i = 1:num_i
        for j = 1:num_j
            switch true
                % Left Boundary (Velocity Inlet)
                case i == 1; [u(i,j,:), V(i,j,:)] = velocity_inlet_BC(u_inf, v_inf, rho_inf, V(i+1,j,:), Cv, R);
                
                % Right Boundary (Pressure Outlet)
                case i == num_i; [u(i,j,:), V(i,j,:)] = pressure_outlet_BC(V(i-1,j,:), P_inf, Cv, R);
                
                % Bottom Boundary (Wall)
                case j == 1; [u(i,j,:), V(i,j,:)] = wall_BC(squeeze(V(i,j+1,:)), eta_x(i,j), eta_y(i,j), Cv, R);
                
                % Top Boundary (Farfield)
                case j == num_j; [u(i,j,:), V(i,j,:)] = farfield_BC(u_inf, v_inf, P_inf, T_inf, Cv, R);
            end
        end
    end
end


%% Inputs

% Provided mesh
[x,y,delta_x,delta_y,num_i,num_j] = read_plot3d_2d('bump.x');

% Freestream Flow Properties
gamma = 1.4;
R = 287;
Cp = gamma * R / (gamma - 1);
Cv = R / (gamma - 1);
T_inf = 300;
P_inf = 10^5;
rho_inf = P_inf / (R*T_inf);
c = repmat(sqrt(gamma * R * T_inf), num_i, num_j);
M = 0.3;
ht = Cp * T_inf * (1 + (gamma-1)/2*M^2);
u_inf = M * c(1,1);
v_inf = 0;
freestream.T = T_inf;
freestream.P = P_inf;
freestream.rho = rho_inf;
freestream.u = u_inf;
freestream.v = v_inf;
freestream.cv = Cv;
freestream.R = R;
freestream.gamma = gamma;
freestream.cp = gamma*R / (gamma - 1);

% Solver Variables
iterations = 50000;
tolerance = 1e-3;
CFL = 1.5;
alpha = [0.5, 1];
epsilon = 0.05;
num_stages = 2;

% Check CFL (Courant-Friedrichs-Lewy) number
if CFL > 1.8
    error("CFL number must be less than 1.8.\n"); 
end


%% Initializing Arrays

% Primitive State Vector
V = repmat(reshape([rho_inf; u_inf; v_inf; P_inf], 1, 1, 4), num_i, num_j, 1);

% Conservative State Vector
u = zeros(num_i, num_j, 4);
u_old = zeros(size(V));
for i=1:num_i
    for j=1:num_j
        u_old(i,j,:) = get_conservative(squeeze(V(i,j,:)), Cv, R); 
    end
end

% Residuals — only store current and first iteration to avoid huge memory allocation
residuals_n = zeros(num_i, num_j, 4); % residual at iteration n
residuals_1 = zeros(num_i, num_j, 4); % residual at iteration 1

% Fluxes
A = zeros(size(u));
B = zeros(size(u));

% Grid Metrics
x_xi  = zeros(num_i, num_j);
x_eta = zeros(num_i, num_j);
y_xi  = zeros(num_i, num_j);
y_eta = zeros(num_i, num_j);

% Inverse Grid Metrics
xi_x  = zeros(num_i, num_j);
xi_y  = zeros(num_i, num_j);
eta_x = zeros(num_i, num_j);
eta_y = zeros(num_i, num_j);

% Jacobian
J = zeros(num_i, num_j);

% Time step (interior cells)
delta_t = zeros(num_i-1, num_j-1);


%% Grid Metrics
for j = 1:num_j
    for i = 1:num_i

        if i==1 && j==1
            % Bottom-Left corner
            x_xi(i,j)  = (-3*x(1,1) + 4*x(2,1) - x(3,1)) / 2;
            x_eta(i,j) = (-3*x(1,1) + 4*x(1,2) - x(1,3)) / 2;
            y_xi(i,j)  = (-3*y(1,1) + 4*y(2,1) - y(3,1)) / 2;
            y_eta(i,j) = (-3*y(1,1) + 4*y(1,2) - y(1,3)) / 2;

        elseif i==1 && j==num_j
            % Top-Left corner
            x_xi(i,j)  = (-3*x(1,j) + 4*x(2,j) - x(3,j)) / 2;
            y_xi(i,j)  = (-3*y(1,j) + 4*y(2,j) - y(3,j)) / 2;
            x_eta(i,j) = ( 3*x(1,j) - 4*x(1,j-1) + x(1,j-2)) / 2;
            y_eta(i,j) = ( 3*y(1,j) - 4*y(1,j-1) + y(1,j-2)) / 2;

        elseif i==num_i && j==1
            % Bottom-Right corner
            x_xi(i,j)  = ( 3*x(i,1) - 4*x(i-1,1) + x(i-2,1)) / 2;
            x_eta(i,j) = (-3*x(i,1) + 4*x(i,2) - x(i,3)) / 2;
            y_xi(i,j)  = ( 3*y(i,1) - 4*y(i-1,1) + y(i-2,1)) / 2;
            y_eta(i,j) = (-3*y(i,1) + 4*y(i,2) - y(i,3)) / 2;

        elseif i==num_i && j==num_j
            % Top-Right corner
            x_xi(i,j)  = ( 3*x(i,j) - 4*x(i-1,j) + x(i-2,j)) / 2;
            x_eta(i,j) = ( 3*x(i,j) - 4*x(i,j-1) + x(i,j-2)) / 2;
            y_xi(i,j)  = ( 3*y(i,j) - 4*y(i-1,j) + y(i-2,j)) / 2;
            y_eta(i,j) = ( 3*y(i,j) - 4*y(i,j-1) + y(i,j-2)) / 2;

        elseif j == 1
            % Bottom boundary (i = 2..num_i-1)
            x_xi(i,j)  = (x(i+1,j) - x(i-1,j)) / 2;
            x_eta(i,j) = (-3*x(i,1) + 4*x(i,2) - x(i,3)) / 2;
            y_xi(i,j)  = (y(i+1,j) - y(i-1,j)) / 2;
            y_eta(i,j) = (-3*y(i,1) + 4*y(i,2) - y(i,3)) / 2;

        elseif j == num_j
            % Top boundary
            x_xi(i,j)  = (x(i+1,j) - x(i-1,j)) / 2;
            y_xi(i,j)  = (y(i+1,j) - y(i-1,j)) / 2;
            x_eta(i,j) = ( 3*x(i,j) - 4*x(i,j-1) + x(i,j-2)) / 2;
            y_eta(i,j) = ( 3*y(i,j) - 4*y(i,j-1) + y(i,j-2)) / 2;

        elseif i == 1
            % Left boundary
            x_xi(i,j)  = (-3*x(1,j) + 4*x(2,j) - x(3,j)) / 2;
            x_eta(i,j) = (x(1,j+1) - x(1,j-1)) / 2;
            y_xi(i,j)  = (-3*y(1,j) + 4*y(2,j) - y(3,j)) / 2;
            y_eta(i,j) = (y(1,j+1) - y(1,j-1)) / 2;

        elseif i == num_i
            % Right boundary
            x_xi(i,j)  = ( 3*x(i,j) - 4*x(i-1,j) + x(i-2,j)) / 2;
            y_xi(i,j)  = ( 3*y(i,j) - 4*y(i-1,j) + y(i-2,j)) / 2;
            x_eta(i,j) = (x(i,j+1) - x(i,j-1)) / 2;
            y_eta(i,j) = (y(i,j+1) - y(i,j-1)) / 2;

        else
            % True interior
            x_xi(i,j)  = (x(i+1,j) - x(i-1,j)) / 2;
            x_eta(i,j) = (x(i,j+1) - x(i,j-1)) / 2;
            y_xi(i,j)  = (y(i+1,j) - y(i-1,j)) / 2;
            y_eta(i,j) = (y(i,j+1) - y(i,j-1)) / 2;
        end

        % Jacobian & inverse metrics (safe for all points)
        J(i,j) = x_xi(i,j)*y_eta(i,j) - x_eta(i,j)*y_xi(i,j);
        if J(i,j) <= 0
            error('Non-positive Jacobian at (%d,%d)',i,j);
        end
        xi_x(i,j)  =  y_eta(i,j) / J(i,j);
        xi_y(i,j)  = -x_eta(i,j) / J(i,j);
        eta_x(i,j) = -y_xi(i,j)  / J(i,j);
        eta_y(i,j) =  x_xi(i,j)  / J(i,j);
    end
end


%% Jameson's Runge-Kutta Method with 2nd Order Dissipation (Explicit Method)

% Update Boundary Conditions
[u_old, V] = update_boundaries(V, u_old, freestream, num_i, num_j, eta_x, eta_y);

% Iterate until Convergence
for n = 1:iterations
    delta_t = zeros(num_i, num_j);

    % Find local time steps (local delta-t)
    for i = 2:num_i-1
        for j = 2:num_j-1
            Vx = V(i,j,2);
            Vy = V(i,j,3);
            a = c(i,j);
            % contravariant velocities
            U  = Vx*xi_x(i,j) + Vy*xi_y(i,j);
            Vc = Vx*eta_x(i,j)+ Vy*eta_y(i,j);
            % spectral radii
            lambda_xi  = abs(U)  + a*sqrt(xi_x(i,j)^2 + xi_y(i,j)^2);
            lambda_eta = abs(Vc) + a*sqrt(eta_x(i,j)^2 + eta_y(i,j)^2);

            delta_t(i,j) = CFL / (lambda_xi + lambda_eta);
        end
    end

    % Flux Vectors
    for i = 1:num_i
        for j = 1:num_j
            [A(i,j,:), B(i,j,:)] = solve_fluxes(squeeze(V(i,j,:)), freestream);
        end
    end

    u_k = cell(1, num_stages);
    V_k = cell(1, num_stages);

    % Runge-Kutta Loop
    for k = 1:num_stages
        % Use appropriate prior state
        if k == 1
            u_previous = u_old;
            V_previous = V;
        else
            u_previous = u_k{k-1};
            V_previous = V_k{k-1};
        end
    
        u_k{k} = u_old;
        V_k{k} = V;
    
        % local speed of sound
        c = sqrt(gamma * V_previous(:,:,4) ./ V_previous(:,:,1));
    
        % velocity field for dissipation
        Vx_field = u_previous(:,:,2) ./ u_previous(:,:,1);
        Vy_field = u_previous(:,:,3) ./ u_previous(:,:,1);
    
        % Residuals for interior points
        for i = 2:num_i-1
            for j = 2:num_j-1
    
                AE = 0.5*(A(i+1,j,:) + A(i,j,:));
                AW = 0.5*(A(i-1,j,:) + A(i,j,:));
                BE = 0.5*(B(i+1,j,:) + B(i,j,:));
                BW = 0.5*(B(i-1,j,:) + B(i,j,:));
                AN = 0.5*(A(i,j+1,:) + A(i,j,:));
                AS = 0.5*(A(i,j-1,:) + A(i,j,:));
                BN = 0.5*(B(i,j+1,:) + B(i,j,:));
                BS = 0.5*(B(i,j-1,:) + B(i,j,:));
    
                y_eta_E = 0.5*(y_eta(i,j) + y_eta(i+1,j));
                x_eta_E = 0.5*(x_eta(i,j) + x_eta(i+1,j));
                y_eta_W = 0.5*(y_eta(i,j) + y_eta(i-1,j));
                x_eta_W = 0.5*(x_eta(i,j) + x_eta(i-1,j));
                y_xi_N  = 0.5*(y_xi(i,j)  + y_xi(i,j+1));
                x_xi_N  = 0.5*(x_xi(i,j)  + x_xi(i,j+1));
                y_xi_S  = 0.5*(y_xi(i,j)  + y_xi(i,j-1));
                x_xi_S  = 0.5*(x_xi(i,j)  + x_xi(i,j-1));
    
                AE_prime = AE.*y_eta_E - BE.*x_eta_E;
                AW_prime = AW.*y_eta_W - BW.*x_eta_W;
                BN_prime = AN.*y_xi_N  - BN.*x_xi_N;
                BS_prime = AS.*y_xi_S  - BS.*x_xi_S;
    
                % dissipation
                VxE = 0.5*(Vx_field(i,j)+Vx_field(i+1,j));
                VyE = 0.5*(Vy_field(i,j)+Vy_field(i+1,j));
                cE  = 0.5*(c(i,j)+c(i+1,j));
                lambda_E = abs(VxE*y_eta_E - VyE*x_eta_E) + cE*sqrt(y_eta_E^2 + x_eta_E^2);
                DE = epsilon * lambda_E * (u_previous(i+1,j,:) - u_previous(i,j,:));
    
                VxW = 0.5*(Vx_field(i,j)+Vx_field(i-1,j));
                VyW = 0.5*(Vy_field(i,j)+Vy_field(i-1,j));
                cW  = 0.5*(c(i,j)+c(i-1,j));
                lambda_W = abs(VxW*y_eta_W - VyW*x_eta_W) + cW*sqrt(y_eta_W^2 + x_eta_W^2);
                DW = epsilon * lambda_W * (u_previous(i,j,:) - u_previous(i-1,j,:));
    
                VxN = 0.5*(Vx_field(i,j)+Vx_field(i,j+1));
                VyN = 0.5*(Vy_field(i,j)+Vy_field(i,j+1));
                cN  = 0.5*(c(i,j)+c(i,j+1));
                lambda_N = abs(VxN*y_xi_N - VyN*x_xi_N) + cN*sqrt(y_xi_N^2 + x_xi_N^2);
                DN = epsilon * lambda_N * (u_previous(i,j+1,:) - u_previous(i,j,:));
    
                VxS = 0.5*(Vx_field(i,j)+Vx_field(i,j-1));
                VyS = 0.5*(Vy_field(i,j)+Vy_field(i,j-1));
                cS  = 0.5*(c(i,j)+c(i,j-1));
                lambda_S = abs(VxS*y_xi_S - VyS*x_xi_S) + cS*sqrt(y_xi_S^2 + x_xi_S^2);
                DS = epsilon * lambda_S * (u_previous(i,j,:) - u_previous(i,j-1,:));
    
                residuals_n(i,j,:) = -(AE_prime - AW_prime + BN_prime - BS_prime) + (DE - DW + DN - DS);
            end
        end
    
        % Update the solution for interior points
        u_k{k}(2:num_i-1, 2:num_j-1, :) = u_old(2:num_i-1, 2:num_j-1, :) + alpha(k) * delta_t(2:num_i-1, 2:num_j-1) ./ J(2:num_i-1, 2:num_j-1) .* residuals_n(2:num_i-1, 2:num_j-1, :);
    
        % Primitive states for interior points
        for i = 2:num_i-1
            for j = 2:num_j-1
                V_k{k}(i,j,:) = get_primitive(squeeze(u_k{k}(i,j,:)), gamma);
            end
        end
    
        % Boundary Conditions
        [u_k{k}, V_k{k}] = update_boundaries(V_k{k}, u_k{k}, freestream, num_i, num_j, eta_x, eta_y);
    
        % Fluxes
        for i = 1:num_i
            for j = 1:num_j
                [A(i,j,:), B(i,j,:)] = solve_fluxes(squeeze(V_k{k}(i,j,:)), freestream);
            end
        end
    
        % Store residual of first iteration for normalisation
        if n == 1 && k == num_stages
            residuals_1 = residuals_n;
        end
    end

    % Update Conservative State Vector with the last Runge-Kutta step
    u = u_k{end};

    % L2 Normalization of Residual
    interior_n = residuals_n(2:end-1, 2:end-1, :);
    interior_1 = residuals_1(2:end-1, 2:end-1, :);
    L2_norm = squeeze(sqrt(sum(sum(interior_n.^2, 1), 2))) ./ squeeze(sqrt(sum(sum(interior_1.^2, 1), 2)));   % 4-element vector

    % Display every ten iterations
    if mod(n,10) == 0
        fprintf('\nIteration %d\n', n); 
        display_residuals(L2_norm); 
    end

    % Check for Convergence
    if all(L2_norm < tolerance)
        fprintf('Converged at iteration %d\n', n); 
        display_residuals(L2_norm); 
        break; 
    end

    u_old = u;
    V = V_k{end};
end

% Check for Convergence
if any(L2_norm >= tolerance)
    fprintf('Did not converge.\n'); 
    display_residuals(L2_norm); 
end

%% Post-Processing

% Extract Properties
rho = squeeze(V(:,:,1));
Vx = squeeze(V(:,:,2));
Vy = squeeze(V(:,:,3));
P = squeeze(V(:,:,4));

% Velocity magnitude
V = sqrt(Vx.^2 + Vy.^2);

% Pressure Coefficient
Cp_plot = 1 - (V ./ u_inf).^2;

% Temperature
T = P ./ (rho * R);

% Mach number
M = V ./ c;


%% Plots

% Convergence History
L2_norm = L2_norm(L2_norm > 0);
figure;
semilogy(L2_norm, 'LineWidth', 1.5);
grid on;
title('Residual Convergence History');
xlabel('Iteration');
ylabel('L2 Norm (log scale)');

% Velocity Magnitude
figure;
contourf(x, y, V, 40, 'LineColor', 'none');
colorbar;
title('Velocity Magnitude');
xlabel('x'); ylabel('y');
axis equal tight;

% Pressure Coefficient
figure;
contourf(x, y, Cp_plot, 30, 'LineColor', 'none');
colorbar;
title('Pressure Contours');
xlabel('x'); ylabel('y');
axis equal tight;

% Mach Number
figure;
contourf(x, y, M, 30, 'LineColor', 'none');
colorbar;
title('Mach Number Contours');
xlabel('x'); ylabel('y');
axis equal tight;

% Streamlines
% Create uniform grid for stable streamline computation
xi = linspace(min(x(:)), max(x(:)), 200);
yi = linspace(min(y(:)), max(y(:)), 200);
[XI, YI] = meshgrid(xi, yi);

% Interpolate velocity field onto uniform grid
Vx_i = griddata(x, y, Vx, XI, YI);
Vy_i = griddata(x, y, Vy, XI, YI);

% Clean any NaN/Inf values (prevents streamline crashes)
Vx_i(~isfinite(Vx_i)) = 0;
Vy_i(~isfinite(Vy_i)) = 0;

% Interpolated velocity magnitude
V_i = sqrt(Vx_i.^2 + Vy_i.^2);

% Define streamline seed points (left boundary)
nSeeds = 25;
x_start = xi(1) * ones(1, nSeeds);
y_start = linspace(min(yi), max(yi), nSeeds);

figure;
contourf(XI, YI, V_i, 30, 'LineColor', 'none');
colorbar;
hold on;
h = streamline(XI, YI, Vx_i, Vy_i, x_start, y_start);
set(h, 'Color', 'k');
title('Velocity Magnitude with Streamlines');
xlabel('x'); ylabel('y');
axis equal tight;

