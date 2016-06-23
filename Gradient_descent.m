function [optTheta,functionVal,exitFlag]=Gradient_descent( )  
  
 options = optimset('GradObj','on','MaxIter',100);  
 initialTheta = [0;0];  
 [optTheta,functionVal,exitFlag] = fminunc(@costFunction3,initialTheta,options);  
  
end  