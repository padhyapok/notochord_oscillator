function dydt=odefunc(t,y,alpha_1,alpha_2,beta_1,n1,k_1,k_3,gamma_1,beta_2,beta_3)
  erk_func_self=(y(1)^n1)/(k_1^n1+y(1)^n1);
  dydt(1)=alpha_1-y(1)*beta_1*y(2)+gamma_1*(1-y(1))*erk_func_self-beta_2*y(1);
  erk_func=(y(1)^n1)/(k_3^n1+y(1)^n1);
  dydt(2)=alpha_2*erk_func-beta_3*y(2);
  dydt=[dydt(1);dydt(2)];
end    