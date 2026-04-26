%% =======================
%   ЗАГРУЗКА ДАННЫХ
% =======================

FileName = '2D_SCAN_21.16.26.txt';
DATA = importdata(FileName,'\t',0);

% Диапазон длин волн монохроматора (в мкм)
mono = (4750:5:5000)/1000;

% Отображаемый диапазон (нм)
wl_1 = 4750;
wl_2 = 5000;

c = 299792458; % скорость света


%% =======================
%   ОСНОВНОЙ ЦИКЛ ПО СКАНАМ
% =======================

N_scans = min(size(DATA))/4;

for k = 1:N_scans
    
    fprintf('Processing scan %d\n', k);
    
    %% -----------------------
    % 1. ИЗВЛЕЧЕНИЕ КАНАЛОВ
    % -----------------------
    
    Phase  = DATA(:,k*4-3); % фаза интерферометра
    Signal = DATA(:,k*4-2); % нелинейный сигнал
    AI3    = DATA(:,k*4-1); % референс (опорный канал)
    AI4    = DATA(:,k*4);   % дополнительный канал
    
    %% -----------------------
    % 2. УДАЛЕНИЕ "ЗАГЛУШКИ"
    % -----------------------
    
    valid = Signal < 900;
    
    Phase  = Phase(valid);
    Signal = Signal(valid);
    AI3    = AI3(valid);
    AI4    = AI4(valid);
    
    
    %% -----------------------
    % 3. РАЗДЕЛЕНИЕ OPEN / CLOSE (чоппер)
    % -----------------------
    
    if max(AI3(1:2:end)) > max(AI3(2:2:end))
        idx_open  = 1:2:length(AI3);
        idx_close = 2:2:length(AI3);
    else
        idx_open  = 2:2:length(AI3);
        idx_close = 1:2:length(AI3);
    end
    
    Phase_open  = Phase(idx_open);
    Phase_close = Phase(idx_close);
    
    Signal_open  = Signal(idx_open);
    Signal_close = Signal(idx_close);
    
    AI3_open = AI3(idx_open);
    AI4_open = AI4(idx_open);
    
    
    %% -----------------------
    % 4. ФАЗА → ВРЕМЯ (delay)
    % -----------------------
    
    lambda_ref = 632.8e-9; % HeNe
    
    t_open  = Phase_open  / (2*pi) * lambda_ref / c;
    t_close = Phase_close / (2*pi) * lambda_ref / c;
    
    
    %% -----------------------
    % 5. УДАЛЕНИЕ ДУБЛЕЙ
    % -----------------------
    
    [t_open,  ia_open]  = unique(t_open);
    [t_close, ia_close] = unique(t_close);
    
    Signal_open  = Signal_open(ia_open);
    Signal_close = Signal_close(ia_close);
    
    AI3_open = AI3_open(ia_open);
    AI4_open = AI4_open(ia_open);
    
    
    %% -----------------------
    % 6. СГЛАЖИВАНИЕ (Gaussian)
    % -----------------------
    
    tau = 7e-14;
    
    Signal_open_smooth  = Signal_open;
    Signal_close_smooth = Signal_close;
    AI3_smooth          = AI3_open;
    
    for i = 1:length(t_open)
        g = exp(-(t_open - t_open(i)).^2 / tau^2);
        Signal_open_smooth(i) = sum(Signal_open .* g) / sum(g);
        AI3_smooth(i)         = sum(AI3_open .* g) / sum(g);
    end
    
    for i = 1:length(t_close)
        g = exp(-(t_close - t_close(i)).^2 / tau^2);
        Signal_close_smooth(i) = sum(Signal_close .* g) / sum(g);
    end
    
    % Быстрая компонента (high-pass)
    Signal_open_fast  = Signal_open  - Signal_open_smooth;
    Signal_close_fast = Signal_close - Signal_close_smooth;
    AI3_fast          = AI3_open     - AI3_smooth;
    
    
    %% -----------------------
    % 7. ОПРЕДЕЛЕНИЕ t0
    % -----------------------
    
    [~, idx_t0] = max(AI3_fast);
    t0 = t_open(idx_t0);
    
    
    %% -----------------------
    % 8. ВЫДЕЛЕНИЕ ОКНА ВОКРУГ t0
    % -----------------------
    
    T_AI3 = 500e-13;
    
    [~, n1] = min(abs(t_open - (t0 - T_AI3)));
    [~, n2] = min(abs(t_open - (t0 + T_AI3)));
    
    t_direct  = t_open(n1:n2);
    AI3_direct = AI3_fast(n1:n2)';
    
    
    %% -----------------------
    % 9. ВЕСА ДЛЯ ИНТЕГРИРОВАНИЯ
    % -----------------------
    
    dt = diff(t_direct);
    WeightT = zeros(size(t_direct));
    
    WeightT(2:end-1) = (dt(1:end-1) + dt(2:end))/2;
    WeightT(1)  = dt(1)/2;
    WeightT(end)= dt(end)/2;
    
    
    %% -----------------------
    % 10. ЧАСТОТНАЯ СЕТКА
    % -----------------------
    
    N_freq = 51;
    Fmin = 5.996e13;
    Fmax = 6.311e13;
    
    freq = linspace(Fmin, Fmax, N_freq);
    
    
    %% -----------------------
    % 11. ОЦЕНКА ФАЗЫ (AI3)
    % -----------------------
    
    Spec_sin = zeros(1,N_freq);
    Spec_cos = zeros(1,N_freq);
    
    for i = 1:N_freq
        w = 2*pi*freq(i);
        
        s = sin(w*(t_direct - t0));
        c = cos(w*(t_direct - t0));
        
        Spec_sin(i) = sum(AI3_direct .* s .* WeightT);
        Spec_cos(i) = sum(AI3_direct .* c .* WeightT);
    end
    
    AI3_phase = unwrap(angle(Spec_cos + 1i*Spec_sin));
    
    % Линейная аппроксимация фазы
    Alpha = polyfit(freq, AI3_phase, 1);
    phase_fit = polyval(Alpha, freq);
    
    
    %% -----------------------
    % 12. ВЕСА ДЛЯ OPEN/CLOSE
    % -----------------------
    
    Weight_open  = computeWeights(t_open);
    Weight_close = computeWeights(t_close);
    
    
    %% -----------------------
    % 13. ОБНУЛЕНИЕ ПОСЛЕ t0
    % -----------------------
    
    Signal_open_fast(idx_t0:end) = 0;
    
    [~, idx_close_t0] = min(abs(t_close - t0));
    Signal_close_fast(idx_close_t0:end) = 0;
    
    
    %% -----------------------
    % 14. СПЕКТРЫ (OPEN / CLOSE)
    % -----------------------
    
    Spec_open  = zeros(1,N_freq);
    Spec_close = zeros(1,N_freq);
    
    for i = 1:N_freq
        
        w = 2*pi*freq(i);
        phi = phase_fit(i);
        
        % open
        s = sin(w*(t_open - t0) - phi);
        c = cos(w*(t_open - t0) - phi);
        
        Spec_open(i) = sum(Signal_open_fast .* (c + 1i*s) .* Weight_open);
        
        % close
        s = sin(w*(t_close - t0) - phi);
        c = cos(w*(t_close - t0) - phi);
        
        Spec_close(i) = sum(Signal_close_fast .* (c + 1i*s) .* Weight_close);
    end
    
    % Нормировка
    norm_level = mean(Signal_open);
    Spec_open  = Spec_open  / norm_level;
    Spec_close = Spec_close / norm_level;
    
    
    %% -----------------------
    % 15. СОХРАНЕНИЕ
    % -----------------------
    
    if k == 1
        R_open  = zeros(N_scans, N_freq);
        R_close = zeros(N_scans, N_freq);
    end
    
    R_open(k,:)  = Spec_open;
    R_close(k,:) = Spec_close;
    
end


%% =======================
%   ФИНАЛЬНЫЙ СИГНАЛ
% =======================

R = R_open - R_close;


%% =======================
%   ВИЗУАЛИЗАЦИЯ
% =======================

R_plot = R;

% Нормализация отдельно для + и -
R_plot(real(R_plot)>0) = R_plot(real(R_plot)>0) / max(real(R_plot(:)));
R_plot(real(R_plot)<0) = R_plot(real(R_plot)<0) / abs(min(real(R_plot(:))));

freq_mono = 1./mono * 1e4;

figure;
imagesc(freq/c/100, freq_mono, -real(R_plot), [-1 1]);

set(gca,'YDir','normal');
xlabel('Excitation (cm^{-1})');
ylabel('Detection (cm^{-1})');
axis square;
colormap jet;


%% =======================
%   ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ
% =======================

function W = computeWeights(t)
    dt = diff(t);
    W = zeros(size(t));
    
    W(2:end-1) = (dt(1:end-1) + dt(2:end))/2;
    W(1)  = dt(1)/2;
    W(end)= dt(end)/2;
end