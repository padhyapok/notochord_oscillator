clear all;clc;close all
tspan=[0 80];
figure_counter=1;
alpha_list=linspace(0.235,0.44,15);%for chosen parameter set, confirmed first occurence of bifurcation at 0.226
alpha_2=0.4;
beta_1=3;
n1=8;
k_1=0.42;
k_3=0.47;
gamma_1 = 2.5;
beta_2=1.0;
beta_3=0.3;
erk_r=linspace(0.0,1.,100);
spry_r=linspace(0.,1.,100);
col_tm_pks=length(alpha_list);
col_tm_trs=length(alpha_list);
col_amp=length(alpha_list);
for al =1:length(alpha_list)
    alpha_1=alpha_list(al);
    alpha_val_str=num2str(alpha_1);
    alpha_val_str=replace(alpha_val_str,'.','p');
    
    [t,sol]=ode23s(@(t,y) odefunc(t,y,alpha_1,alpha_2,beta_1,n1,k_1,k_3,gamma_1,beta_2,beta_3),tspan,hist_fn);
    %uncomment for trajectories
    % figure(1)
    % plot(t,sol(:,1),LineWidth=3,color='#BB5566');
    % hold on;
    % plot(t,sol(:,2),LineWidth=3,color='#9970AB')
    % ylim([0,1.0])
    % xlim([0,80])
    % set(gca, 'linewidth',2,'fontsize',40,'fontname','Helvetica')
    % %add peaks or troughs
    [pks,loc_pks]=findpeaks(sol(:,1));
    [trs,loc_trs]=findpeaks(-sol(:,1));
    
    %amplitude
    mean_pks=mean(sol(loc_pks,1));
    mean_trs=mean(sol(loc_trs,1));
    amplitude=mean_pks-mean_trs
    %time period
    tm_pks=mean(diff(t(loc_pks)));
    
    tm_trs=mean(diff(t(loc_trs)));
    
    col_tm_pks(al)=tm_pks;
    col_tm_trs(al)= tm_trs;
    col_amp(al)=amplitude;
    
    % ax=gca;
    % exportgraphics(ax,strcat(base_path+sv_path,'time_pt_alpha_',alpha_val_str,'_gamma_2p5.png'))

    %plot additional nullclines
    erk_func=(erk_r.^n1)./(erk_r.^n1+k_1^n1);

    sef_act=gamma_1*(1-erk_r).*erk_func;
    spry_func=(alpha_1-beta_2*erk_r+sef_act)./(erk_r*(beta_1));
    erk_null=spry_func;
    
    erk_func2=(erk_r.^n1)./(erk_r.^n1+k_3^n1);
    spry_null=alpha_2*erk_func2/beta_3;
    %uncomment for nullclines
    % figure(2)
    % plot(erk_r,erk_null,linewidth=3,color='#BB5566')
    % hold on
    % plot(erk_r,spry_null,linewidth=3,color='#9970AB')
    % 
    % scatter(sol(:,1),sol(:,2),15,'k','filled');
    % set(gca, 'linewidth',2,'fontsize',40,'fontname','Helvetica')
    % xlim([0.,1])
    % ylim([-0.05,1])
    % % ax=gca;
    % exportgraphics(ax,strcat(base_path+sv_path,'nullclines_',alpha_val_str,'_gamma_2p5.png'))
    
end    
%plot all collected values
f=figure()
plot(alpha_list,col_tm_pks,'o-','Color','#BB5566',LineWidth=3)
set(gca, 'linewidth',2,'fontsize',30,'fontname','Helvetica')
ylim([0 30])
f.Position=[100,100,700,500];
%ax=gca;
%exportgraphics(ax,strcat(base_path+sv_path,'alpha_vs_time_per.png'),'Resolution',600)


f=figure()
plot(alpha_list,col_amp,'o-','Color','#BB5566',LineWidth=3)
set(gca, 'linewidth',2,'fontsize',30,'fontname','Helvetica')
ylim([0. 1])
f.Position=[100,100,700,500];
%ax=gca;
%exportgraphics(ax,strcat(base_path+sv_path,'alpha_vs_amp.png'),'Resolution',600)