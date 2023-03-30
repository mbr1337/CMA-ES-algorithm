%% sprzątanie
clc;
clear;
close all;
%% Ustawienia problemów
      % Funkcja kosztów | @Ackley
     
CostFunction = @Sphere
nVar = 50;                % Liczba nieznanych (decyzyjnych) zmiennych

VarSize = [1 nVar];       % Rozmiar macierzy zmiennych decyzyjnych

VarMin = -100;             % Dolna granica zmiennych decyzyjnych 
VarMax = 100;             % Górna granica zmiennych decyzyjnych 

%% ustawienia

% Maksymalna liczba iteracji
MaxIt = 1000;

% Wielkość populacji (i liczba potomstwa)
lambda = (4+round(3*log(nVar)))*10; %(4+round(3*log(nVar)))*10;

% Liczba rodzicow
mu = round(lambda/2);

% Wagi rodzicow
w = log(mu+0.5)-log(1:mu);
w = w/sum(w);

% Liczba skutecznych rozwiązań
mu_eff = 1/sum(w.^2);

% Parametry kontroli rozmiaru kroku (c_sigma i d_sigma);
sigma0 = 0.5*(VarMax-VarMin); %def 0.3
cs = (mu_eff+2)/(nVar+mu_eff+5); %def +2 i +5
ds = 1+cs+2*max(sqrt((mu_eff-1)/(nVar+1))-1, 0); %def 2*max
ENN = sqrt(nVar)*(1-1/(4*nVar)+1/(21*nVar^2));

% Parametry aktualizacji kowariancji
cc = (4+mu_eff/nVar)/(4+nVar+2*mu_eff/nVar);
c1 = 2/((nVar+1.3)^2+mu_eff);
alpha_mu = 2;
cmu = min(1-c1, alpha_mu*(mu_eff-2+1/mu_eff)/((nVar+2)^2+alpha_mu*mu_eff/2));
hth = (1.4+2/(nVar+1))*ENN;

%% Inicializacja

% cell() - Tablica komórek z indeksowanymi kontenerami danych zwanymi komórkami,
% gdzie każda komórka może zawierać dowolny typ danych.

ps = cell(MaxIt, 1);
pc = cell(MaxIt, 1);
C = cell(MaxIt, 1);
sigma = cell(MaxIt, 1);

ps{1} = zeros(VarSize);
pc{1} = zeros(VarSize);
C{1} = eye(nVar);
sigma{1} = sigma0;

empty_individual.Position = [];
empty_individual.Step = [];
empty_individual.Cost = [];

%repmat() - Powtórz kopie tablicy
M = repmat(empty_individual, MaxIt, 1);
M(1).Position = unifrnd(VarMin, VarMax, VarSize);
M(1).Step = zeros(VarSize);
M(1).Cost = CostFunction(M(1).Position);

BestSol = M(1);

BestCost = zeros(MaxIt, 1);

%% CMA-ES główna pętla

for g = 1:MaxIt
    
    % Generuj próbki
    pop = repmat(empty_individual, lambda, 1);
    for i = 1:lambda

        % Generuje próbki
        pop(i).Step = mvnrnd(zeros(VarSize), C{g});
        pop(i).Position = M(g).Position + sigma{g}*pop(i).Step;
        
        % Stosowanie ograniczeń
        pop(i).Position = max(pop(i).Position, VarMin);
        pop(i).Position = min(pop(i).Position, VarMax);
        
        % ewaluacja
        pop(i).Cost = CostFunction(pop(i).Position);
        
        % Zaktualizuj najlepsze rozwiązanie, jakie kiedykolwiek znaleziono
        if pop(i).Cost<BestSol.Cost
            BestSol = pop(i);
        end
    end
    
    % Sortuj populacje
    Costs = [pop.Cost];
    [Costs, SortOrder] = sort(Costs);
    pop = pop(SortOrder);
  
    % Zapisz wyniki
    BestCost(g) = BestSol.Cost;
    
    % Pokaz wyniki
    disp(['Iteration ' num2str(g) ': Best Cost = ' num2str(BestCost(g))]);
    
    % Wyjdź w ostatniej iteracji
    if g == MaxIt
        break;
    end
    
    % Zaktualizuj średnią
    M(g+1).Step = 0;
    for j = 1:mu
        M(g+1).Step = M(g+1).Step+w(j)*pop(j).Step;
    end
    M(g+1).Position = M(g).Position + sigma{g}*M(g+1).Step;
    
    % Stosowanie ograniczeń
    M(g+1).Position = max(M(g+1).Position, VarMin);
    M(g+1).Position = min(M(g+1).Position, VarMax);
    
    % Ewaluacja
    M(g+1).Cost = CostFunction(M(g+1).Position);
    
    % Zaktualizuj najlepsze rozwiązanie, jakie znaleziono
    if M(g+1).Cost < BestSol.Cost
        BestSol = M(g+1);
    end
    
    % Zaktualizuj rozmiar kroku
    ps{g+1} = (1-cs)*ps{g}+sqrt(cs*(2-cs)*mu_eff)*M(g+1).Step/chol(C{g})';
    sigma{g+1} = sigma{g}*exp(cs/ds*(norm(ps{g+1})/ENN-1))^0.3;
    
    % Zaktualizuj macierz kowariancji
    if norm(ps{g+1})/sqrt(1-(1-cs)^(2*(g+1)))<hth
        hs = 1;
    else
        hs = 0;
    end
    delta = (1-hs)*cc*(2-cc);
    pc{g+1} = (1-cc)*pc{g}+hs*sqrt(cc*(2-cc)*mu_eff)*M(g+1).Step;
    C{g+1} = (1-c1-cmu)*C{g}+c1*(pc{g+1}'*pc{g+1}+delta*C{g});
    for j = 1:mu
        C{g+1} = C{g+1}+cmu*w(j)*pop(j).Step'*pop(j).Step;
    end
    
    % Jeśli macierz kowariancji nie jest dodatnio określona lub blisko liczby pojedynczej
    [V, E] = eig(C{g+1});
    if any(diag(E)<0)
        E = max(E, 0);
        C{g+1} = V*E/V;
    end
    
end

%% Pokaż wyniki

figure;
% plot(BestCost, 'LineWidth', 2);
semilogy(BestCost, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best Cost');
grid on;

