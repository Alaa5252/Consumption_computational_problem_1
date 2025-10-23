%% households' consumption and savings decesions under income risk (VFI and simulation)

% cleanup
clear; clc; close all;

%% parameters (annual)
% defining the parameters. 
beta  = 0.96;        
gamma = 1.3;        
r     = 0.04;        
rho   = 0.90;        
sigma = 0.04;        
mu    = 0.0;        

%% building the grids for income and assets
% income grid: Tauchen
Ny = 9;  % this approximates the continuous income process with 9 discrete levels (I choose 9 income states).           
m  = 3;  % the income grid covers values within +/- 3 σ around the mean, capturing about 99% of possible shocks.           
[yg, P] = Tauchen(Ny, mu, rho, sigma, m);  % this creates a 9×1 vector of all possible log-income levels and a 9×9 matrix of transition probabilities.
y_level = exp(yg);  % this converts log-income to actual income.  

% the above code lines use the Tauchen method to create a 9 point Markov chain for income and turn the continuous stochastic process of income to something can be handled numerically in the VFI.                     

% asset grid: no borrowing; a' is greater than or equals 0
Na   = 500;  % this creates 500 asset grid points.                                
amin = 0.0;  % this is the minimum asset level (the lower bound). No borrowing: the household cannot have negative wealth.                                
amax = 50.0; % this is the maximum asset level (the upper bound).                               
agrid = linspace(amin, amax, Na)'; % this creates the actual grid of asset values. This generates N evenly spaced points between a and b. The apostrophe turns into a column vector.        

% for speed:
R = 1 + r;

%%  defining the utility function (vectorized)
% in this section I define how utility is calculated and how to handle negative consumption. 
% if consumption is negative or zero, MATLAB replaces it with tiny positive number (1e-12 = 0.000000000001).
u = @(c) ((gamma==1) .* log(max(c,1e-12))) + ...
         ((gamma~=1) .* ((max(c,1e-12)).^(1-gamma)) ./ (1-gamma));
 
BIGNEG = -1e12; %if consumption is <=0, this assigns a value of BIGNEG instead of computing real utility. 



%% value function iteration (using brute force over a' on the grid)

% setup 

V  = zeros(Na, Ny);          % this is my initial guess. The iteration below will gradually improve this guess until it converges.

% this Creates a matrix of zeros with size (Na × Ny); one row for each asset level and one column for each income level.
% each element in the matrix represents the value (lifetime utility) starting from asset ai and income yi. 

Vp = V;                      % this creates another matrix Vp with the same size as V.

% during each iteration: Vp stores the new (or updated) value function while V keeps the old one for comparison.
% this way checks how much the value function changed after each iteration to test convergence.

pol_a_ix = ones(Na, Ny);     % index of chosen a' for each (a,y)

% this creates a matrix to record the policy choices.
% it will store, for every combination of a and y the index (position) of the optimal next-period asset a` on the agrid.
% starting it with ones just fills it with placeholder values, we will overwrite them as soon as the algorithm finds the optimal decisions.

% initial setup values for the VFI loop

tol   = 1e-9; % when the change in the value function between iterations is smaller than this, MATLAB will stop iterating.
maxit = 5000; % this is the maximum number of iterations allowed.
diff  = inf; % sets the starting difference to infinity so that the while-loop condition is definitely true the first time, ensuring the loop runs at least once.
iter  = 0; % each time the loop runs, we'll add 1 to iter. 

% precomputing a matrix for a and a' terms to vectorize c = a + y - a'/R

A   = agrid;                      

% A is the current asset vector, copied from agrid.
% it's a column vector of size Na × 1.
% each row represents one possible current asset level ai.

Apr = agrid';                     

% Apr is the transpose of agrid, so it becomes a row vector of size 1 × Na.
% each column now represents a possible next-period asset choice a`k.
% later, when I compute the consumption matrix: c = a + y - a'/R MATLAB automatically broadcasts the two vectors:
% A (Na×1) gets repeated across columns,
% Apr (1×Na) gets repeated down rows,
% so C becomes an Na × Na matrix where:
% each row i corresponds to current asset ai
% each column k corresponds to next asset a`k 
% each entry gives the feasible consumption if I start with ai and save a`k. 

% These two lines set up the current asset vector (A) and next asset vector (Apr) in a shape that lets MATLAB
%  efficiently compute all possible consumption c = a + y - a'/R combinations in one step where no loops needed which much faster.


%% iteration loop
% this is where MATLAB repeatedly improves the household's value function 
% until it converges to the true optimal one.

while diff > tol && iter < maxit % this keeps improving the guess until it stabilizes or we hit the max number of loops
    iter = iter + 1; % counts the number of iterations and starts from 0 and increases by 1 each round.
    Vold = V; % this stores the current value function as the "old" version. It is used to calculate the new value function and compare how much it changes.

    EV = Vold * P';  
% the above line calculates the expected future value for each possible next asset choice a`.
% Vold is (Na × Ny) and P' is (Ny × Ny), giving EV (Na × Ny). 
% EV(:, j) tells the expected continuation value if today's income is yj
% this integrates over all possible income transitions for next period.          

    for j = 1:Ny % this loops over each income state yj. Since there are only 9 income states, this loop is computationally cheap.
        income = y_level(j); % this picks the actual income level corresponding to income state yj. Used in the budget constraint below:


        C = (A + income) - (Apr / R); 
% the above computes consumption for every combination of current assets ai and next assets a`k.
% this results in a matrix of size (Na × Na). 
% each element is how much someone can consume if he/she starts with ai and plan to save a`k

        U = u(C);
% the above calculates instantaneous utility from consumption using the utility function.
%this gives the same (Na × Na) matrix, now with utility values instead of consumption.

        U(C <= 0) = BIGNEG;
% any case where consumption ≤ 0 is infeasible. Those entries are replaced with a very large negative number.

        cont = repmat(EV(:, j)', Na, 1);  
% this copies the expected future value for each next-asset choice (EV(:, j)) across all current asset levels.
% this also makes sure matrix dimensions match so we can add U and β × cont. 

        VV = U + beta * cont;

% the above adds today's utility and discounted expected future value.
% Each element VV(i,k) = total lifetime value from choosing a`k when you have ai and income yj

        [Vp(:, j), pol_a_ix(:, j)] = max(VV, [], 2); % for each current asset level ai (each row), find the best next asset a`k that maximizes total value.% the maximum value for each row becomes the new value function V`(ai,yj) % the column index of that maximum (pol_a_ix) records the optimal saving choice.


    end

    % convergence
    diff = max(abs(Vp(:) - Vold(:)));
% after finishing all income states, compare the new (Vp) and old (Vold) value functions.
% finds the largest absolute difference; the convergence measure.

    V = Vp; % updates the value function for the next iteration.

    if mod(iter,10)==0 || iter==1 % this prints progress every 10 iterations (or on the first one).

        fprintf('Iter %4d | sup-norm diff = %.3e\n', iter, diff); % this displays how far the value function changed this round; helps monitor convergence.
    end
end

fprintf('VFI finished in %d iterations. Final diff = %.3e\n\n', iter, diff);

% this line prints a confirmation that the algorithm converged.

% Recovering policy levels

pol_a = agrid(pol_a_ix);  
% this gives the saving policy function; how much to save depending on assets and income.
                        
pol_c = (agrid + y_level') - pol_a / R;  
% Each cell pol_c(i,j) gives the optimal consumption given current assets ai and income yj.        


%% plotting the value function and policies

% I pick a few income states: low, middle, high
ix_lo = 2; ix_mid = ceil(Ny/2); ix_hi = Ny-1;

figure('Name','Value Function by Income State');
plot(agrid, V(:,ix_lo), 'LineWidth',1.5); hold on;
plot(agrid, V(:,ix_mid),'LineWidth',1.5);
plot(agrid, V(:,ix_hi), 'LineWidth',1.5);
grid on; xlabel('Assets a'); ylabel('V(a,y)');
legend(sprintf('Low y (%.2f)', y_level(ix_lo)), ...
       sprintf('Mid y (%.2f)', y_level(ix_mid)), ...
       sprintf('High y (%.2f)', y_level(ix_hi)),'Location','southeast');
title('Value Function');

figure('Name','Policy: Next-Period Assets a''(a,y)');
plot(agrid, pol_a(:,ix_lo), 'LineWidth',1.5); hold on;
plot(agrid, pol_a(:,ix_mid),'LineWidth',1.5);
plot(agrid, pol_a(:,ix_hi), 'LineWidth',1.5);
grid on; xlabel('Assets a'); ylabel('a''(a,y)');
legend('Low y','Mid y','High y','Location','southeast');
title('Saving Policy');

figure('Name','Policy: Consumption c(a,y)');
plot(agrid, pol_c(:,ix_lo), 'LineWidth',1.5); hold on;
plot(agrid, pol_c(:,ix_mid),'LineWidth',1.5);
plot(agrid, pol_c(:,ix_hi), 'LineWidth',1.5);
grid on; xlabel('Assets a'); ylabel('c(a,y)');
legend('Low y','Mid y','High y','Location','southeast');
title('Consumption Policy');

%% simulation (1000 periods, burning the first 500)
T     = 1000;
burn  = 500;

a_path = zeros(T,1);
y_ix   = zeros(T,1);
c_path = zeros(T,1);

% start at a0 = 0, y0 = closest to mean (yg ~ 0)
[~, y0_ix] = min(abs(yg - 0));
[~, a0_ix] = min(abs(agrid - 0));
y_ix(1) = y0_ix;
a_ix    = a0_ix;

rng(123); % reproducible simulation

for t = 1:T

    c_path(t) = pol_c(a_ix, y_ix(t));
    a_next    = pol_a(a_ix, y_ix(t));
    
    [~, a_ix] = min(abs(agrid - a_next));
    a_path(t) = agrid(a_ix);

    u = rand;
    cumP = cumsum(P(y_ix(t), :));
    y_ix(t+1) = find(u <= cumP, 1, 'first');
end
y_ix = y_ix(1:T);                         

% keeping post-burn samples
a_sim = a_path(burn+1:end);
c_sim = c_path(burn+1:end);
y_sim = y_level(y_ix(burn+1:end));

% reporting std of consumption
std_c = std(c_sim);
fprintf('Std of consumption after burn-in (T=%d, burn=%d): %.4f\n\n', T, burn, std_c);

%% "Tile" style time-series visualization (stacked)
t = (1:length(a_sim))';
figure('Name','Simulated Paths (post burn-in)');
tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(t, y_sim, 'LineWidth',1.1); grid on;
ylabel('Income level (exp y)'); title('Income');

nexttile;
plot(t, a_sim, 'LineWidth',1.1); grid on;
ylabel('Assets a');

nexttile;
plot(t, c_sim, 'LineWidth',1.1); grid on;
ylabel('Consumption c'); xlabel('Time (periods)');
