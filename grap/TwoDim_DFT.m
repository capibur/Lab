%%
% JACOBIAN??????

FileName = '2D_SCAN_21.16.26.txt'; % Имя файла с сырыми данными 2D-спектра
DATA = importdata(FileName,'\t',0); % Импорт данных

%%
% mono=(5330:2:5410)/1000; % Массив длин волн, сканируемых монохроматором
mono=(4750:5:5000)/1000; % Массив длин волн, сканируемых монохроматором
wl_1=4750; % нижняя граница спектра для отображения в нм
wl_2=5000; % верхняя граница спектра для отображения в нм
c=299792458; % скорость света м/c

for k=1:min(size(DATA))/4
% for k=[1,20]
% for k=3
k
    % выделяем сигналы
    Phase=DATA(:,k*4-3);  % Фаза с заглушкой
    Signal=DATA(:,k*4-2); % Полезный сигнал с заглушкой
    AI3=DATA(:,k*4-1 );    % Сигнал AI3 с заглушкой
    AI4=DATA(:,k*4);      % Сигнал AI4 с заглушкой

    % удаляем заглушку
    Phase=Phase(Signal<900);   % Фаза без заглушки
    AI3=AI3(Signal<900);       % Сигнал AI3 без заглушки
    AI4=AI4(Signal<900);       % Сигнал AI4 без заглушки
    Signal=Signal(Signal<900); % Нелинейный сигнал без заглушки
    
    % разделяем строки с открытым/закрытым чоппером
    if max(AI3(1:2:end))>max(AI3(2:2:end))
        Phase_open=Phase(1:2:end);
        Phase_close=Phase(2:2:end);
        Signal_open=Signal(1:2:end);
        Signal_close=Signal(2:2:end);
        AI3_open=AI3(1:2:end);
        AI4_open=AI4(1:2:end);
    else
        Phase_open=Phase(2:2:end);
        Phase_close=Phase(1:2:end);
        Signal_open=Signal(2:2:end);
        Signal_close=Signal(1:2:end);
        AI3_open=AI3(2:2:end);
        AI4_open=AI4(2:2:end);
    end
    
    t_open=Phase_open/2/pi*632.8E-9/c;       % время в секундах для открытых строк
    t_close=Phase_close/2/pi*632.8E-9/c;     % время в секундах для закрытых строк
    [t_open_unique,ia_open,~] = unique(t_open);            % Массив времени без повторяющихся t для открытых строк
    [t_close_unique,ia_close,~] = unique(t_close);          % Массив времени без повторяющихся t для закрытых строк
    AI3_unique=AI3_open(ia_open);                         % Соответствующий массив для сигнала AI3
    AI4_unique=AI4_open(ia_open);                         % Соответствующий массив для сигнала AI4
    Signal_open_unique=Signal_open(ia_open);                % Соответствующий массив для открытых строк полезного сигнала
    Signal_close_unique=Signal_close(ia_close);              % Соответствующий массив для закрытых строк полезного сигнала
    
Signal_open_unique_smoothed=Signal_open_unique;
Signal_close_unique_smoothed=Signal_close_unique;
AI3_unique_smoothed=AI3_unique;
tau=7e-14;
for kk=1:max(size(t_open_unique))
    g=exp(-(t_open_unique-t_open_unique(kk)).^2/tau^2);
    Signal_open_unique_smoothed(kk)=sum(Signal_open_unique.*g)/sum(g);
    AI3_unique_smoothed(kk)=sum(AI3_unique.*g)/sum(g);
end
for kk=1:max(size(t_close_unique))
    g=exp(-(t_close_unique-t_close_unique(kk)).^2/tau^2);
    Signal_close_unique_smoothed(kk)=sum(Signal_close_unique.*g)/sum(g);
end
Signal_open_unique_fast=Signal_open_unique-Signal_open_unique_smoothed;
Signal_close_unique_fast=Signal_close_unique-Signal_close_unique_smoothed;
AI3_unique_fast=AI3_unique-AI3_unique_smoothed;

% figure(2);
% plot(t_close_unique,Signal_close_unique_fast);
% 
    
    [~,n_t0]=max(AI3_unique_fast);
    t0=t_open_unique(n_t0);
    T_AI3=500e-13;
    [~,n1]=min(abs(t0-T_AI3-t_open_unique));
    [~,n2]=min(abs(t0+T_AI3-t_open_unique));
    t_direct=t_open_unique(n1:n2);
    AI3_direct=AI3_unique_fast(n1:n2);
    AI3_direct_size=size(AI3_direct);
    if AI3_direct_size(1)>AI3_direct_size(2)
        AI3_direct=AI3_direct';
    end
%     figure(30);
%     plot(t_direct,AI3_direct);

    WeightT=zeros(size(t_direct));
    deltaTdirect=t_direct(2:end)-t_direct(1:end-1);
    for kk=1:max(size(t_direct))-2
        WeightT(kk+1)=(deltaTdirect(kk+1)+deltaTdirect(kk))/2;
    end
    WeightT(1)=deltaTdirect(1)/2;
    WeightT(end)=deltaTdirect(end)/2;
    
%     t0=t0-16e-15;
    
    N_freq=51; % EDIT: num of freq
    Fmin=5.996e13; % EDIT: freq range
    Fmax=6.311e13; % EDIT: freq range
%     Fmin=5.958e13;
%     Fmax=6.272e13;
    freq=(1:N_freq)/N_freq*(Fmax-Fmin)+Fmin;
    % freq=[5.75:0.01:5.9,6.3:0.01:6.45]*1e13;
    matrix_sin=zeros(max(size(t_direct)),max(size(freq)));
    matrix_cos=zeros(max(size(t_direct)),max(size(freq)));
    Norma_sin=zeros(size(freq));
    Norma_cos=zeros(size(freq));
    for kk=1:max(size(freq))
        matrix_sin(:,kk)=sin(2*pi*(t_direct-t0)*freq(kk));
        matrix_cos(:,kk)=cos(2*pi*(t_direct-t0)*freq(kk));
        Norma_sin(kk)=sum(sin(2*pi*(t_direct-t0)*freq(kk)).^2.*WeightT);
        Norma_cos(kk)=sum(cos(2*pi*(t_direct-t0)*freq(kk)).^2.*WeightT);
    end
    Spec_sin=((AI3_direct.*WeightT')*matrix_sin)./Norma_sin;
    Spec_cos=((AI3_direct.*WeightT')*matrix_cos)./Norma_cos;
    AI3_phase=unwrap(phase(Spec_cos+1j*Spec_sin));
    Alpha=polyfit(freq, AI3_phase, 1); % Аппроксимируем фазу прямой МНК
%     Alpha(1)/2/pi*1e15;       % Углы наклона прямых для разных значений t0
    Lin_fit=freq*Alpha(1)+Alpha(2);
    
    % figure(31);
    % plot(freq,AI3_phase,freq,Lin_fit);    
    % figure(32);
    % plot(freq,Spec_cos,freq,Spec_sin,freq,abs(Spec_cos+1j*Spec_sin));

    WeightT_open=zeros(size(t_open_unique));
    WeightT_close=zeros(size(t_close_unique));
    deltaTopen=t_open_unique(2:end)-t_open_unique(1:end-1);
    deltaTclose=t_close_unique(2:end)-t_close_unique(1:end-1);
    WeightT_open(2:end-1)=(deltaTopen(1:end-1)+deltaTopen(2:end))/2;
    WeightT_open(1)=deltaTopen(1)/2;
    WeightT_open(end)=deltaTopen(end)/2;
    WeightT_close(2:end-1)=(deltaTclose(1:end-1)+deltaTclose(2:end))/2;
    WeightT_close(1)=deltaTclose(1)/2;
    WeightT_close(end)=deltaTclose(end)/2;
   
    matrix_cos_open=zeros(max(size(t_open_unique)),max(size(freq)));
    matrix_cos_close=zeros(max(size(t_close_unique)),max(size(freq)));
    Norma_cos_open=zeros(size(freq));
    Norma_cos_close=zeros(size(freq));
    matrix_sin_open=zeros(max(size(t_open_unique)),max(size(freq)));
    matrix_sin_close=zeros(max(size(t_close_unique)),max(size(freq)));
    Norma_sin_open=zeros(size(freq));
    Norma_sin_close=zeros(size(freq));
    
    % Lin_fit=freq*Alpha(1)+Alpha(2);
    
    for kk=1:max(size(freq))
        % matrix_cos_open(:,kk)=cos(2*pi*(t_open_unique-t0)*freq(kk)-Lin_fit(kk));
        % matrix_cos_close(:,kk)=cos(2*pi*(t_close_unique-t0)*freq(kk)-Lin_fit(kk));
        % Norma_cos_open(kk)=sum(cos(2*pi*(t_open_unique-t0)*freq(kk)-Lin_fit(kk)).^2.*WeightT_open);
        % Norma_cos_close(kk)=sum(cos(2*pi*(t_close_unique-t0)*freq(kk)-Lin_fit(kk)).^2.*WeightT_close);
        % matrix_sin_open(:,kk)=sin(2*pi*(t_open_unique-t0)*freq(kk)-Lin_fit(kk));
        % matrix_sin_close(:,kk)=sin(2*pi*(t_close_unique-t0)*freq(kk)-Lin_fit(kk));
        % Norma_sin_open(kk)=sum(sin(2*pi*(t_open_unique-t0)*freq(kk)-Lin_fit(kk)).^2.*WeightT_open);
        % Norma_sin_close(kk)=sum(sin(2*pi*(t_close_unique-t0)*freq(kk)-Lin_fit(kk)).^2.*WeightT_close);
        matrix_cos_open(:,kk)=cos(2*pi*(t_open_unique-t0)*freq(kk)-AI3_phase(kk));
        matrix_cos_close(:,kk)=cos(2*pi*(t_close_unique-t0)*freq(kk)-AI3_phase(kk));
        Norma_cos_open(kk)=sum(cos(2*pi*(t_open_unique-t0)*freq(kk)-AI3_phase(kk)).^2.*WeightT_open);
        Norma_cos_close(kk)=sum(cos(2*pi*(t_close_unique-t0)*freq(kk)-AI3_phase(kk)).^2.*WeightT_close);
        matrix_sin_open(:,kk)=sin(2*pi*(t_open_unique-t0)*freq(kk)-AI3_phase(kk));
        matrix_sin_close(:,kk)=sin(2*pi*(t_close_unique-t0)*freq(kk)-AI3_phase(kk));
        Norma_sin_open(kk)=sum(sin(2*pi*(t_open_unique-t0)*freq(kk)-AI3_phase(kk)).^2.*WeightT_open);
        Norma_sin_close(kk)=sum(sin(2*pi*(t_close_unique-t0)*freq(kk)-AI3_phase(kk)).^2.*WeightT_close);
    end

    Signal_open_unique_fast(n_t0:end)=zeros(size(Signal_open_unique_fast(n_t0:end)));
    [~,n_t0_close]=min(abs(t_close_unique-t0));
    Signal_close_unique_fast(n_t0_close:end)=zeros(size(Signal_close_unique_fast(n_t0_close:end)));
    Spec_cos_open=((Signal_open_unique_fast'.*WeightT_open')*matrix_cos_open)./Norma_cos_open;
    Spec_sin_open=((Signal_open_unique_fast'.*WeightT_open')*matrix_sin_open)./Norma_sin_open;
    Spec_cos_close=((Signal_close_unique_fast'.*WeightT_close')*matrix_cos_close)./Norma_cos_close;
    Spec_sin_close=((Signal_close_unique_fast'.*WeightT_close')*matrix_sin_close)./Norma_sin_close;
    Uroven_signala=mean(Signal_open_unique);
    Spec_open=(Spec_cos_open+1j*Spec_sin_open)/Uroven_signala;
    Spec_close=(Spec_cos_close+1j*Spec_sin_close)/Uroven_signala;
%     Spec=Spec_open-Spec_close;
%     Spec=Spec_open;
    
%     figure(33);
%     plot(freq,real(Spec_open),freq,real(Spec_close));  

    if k==1
        R_open=zeros(min(size(DATA))/4,N_freq);
        R_close=zeros(min(size(DATA))/4,N_freq);
    end
    R_open(k,:)=Spec_open;
    R_close(k,:)=Spec_close;

end

R=R_open-R_close;

%% Отрисовка 2D спектра

R_shift=R;
R_shift(real(R_shift)>0)=R_shift(real(R_shift)>0)/abs(max(max(real(R_shift))));
R_shift(real(R_shift)<0)=R_shift(real(R_shift)<0)/abs(min(min(real(R_shift))));
freq_mono=1./mono*10^4;
fontsize=18;
figure(13);
imagesc(freq/c/100,freq_mono,real(R_shift)*(-1),[-1 1]);
set(gca,'YDir','normal');
set(gca,'FontSize',fontsize);
ylabel('Спектр детектирования, см^{-1}', 'FontSize', fontsize);
xlabel('Спектр возбуждения, см^{-1}', 'FontSize', fontsize);
axis square;
colormap jet;