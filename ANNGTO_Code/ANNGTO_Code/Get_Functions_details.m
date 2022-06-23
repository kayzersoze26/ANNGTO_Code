%  Traning Feed-forward Neural Networks using Grey Wolf Optimizer   %
%                                                                   %
%  Developed in MATLAB R2011b(7.13)                                 %
%                                                                   %
%  Author and programmer: Seyedali Mirjalili                        %
%                                                                   %
%         e-Mail: ali.mirjalili@gmail.com                           %
%                 seyedali.mirjalili@griffithuni.edu.au             %
%                                                                   %
%       Homepage: http://www.alimirjalili.com                       %
%                                                                   %
%   Main paper: S. Mirjalili,How effective is the Grey Wolf         %
%               optimizer in training multi-layer perceptrons       %
%              Applied Intelligece, in press, 2015,                 %
%               http://dx.doi.org/10.1007/s10489-014-0645-7         %
%                                                                   %

% This function containts full information and implementations of the
% datasets

% lb is the lower bound: lb=[lb_1,lb_2,...,lb_d]
% up is the uppper bound: ub=[ub_1,ub_2,...,ub_d]
% dim is the number of variables (dimension of the problem)

function [lb,ub,dim,fobj,pred_tr] = Get_Functions_details(F)
global pred_tr;
switch F       
   case 'F1'
        fobj=@MLP_XOR
        lb=-10;
        ub=10;
        dim=36;
        
    case 'F2'
        fobj = @MLP_Baloon;
        lb=-10;
        ub=10;
        dim=55;   
        
    case 'F3'
        fobj=@MLP_Iris
        lb=-10;
        ub=10;
        dim=75;
        
    case 'F4'
        fobj=@MLP_Cancer
        lb=-10;
        ub=10;
        dim=209;
        
     case 'F5'
        fobj=@MLP_Heart
        lb=-10;
        ub=10;
        dim=1081;       
        
     case 'F6'
        fobj=@MLP_Sigmoid
        lb=-10;
        ub=10;
        dim=46; 
        
     case 'F7'
        fobj=@MLP_Cosine
        lb=-10;
        ub=10;
        dim=46;    
        
     case 'F8'
         fobj=@MLP_Sine
         lb=-10;
         ub=10;
         dim=46;
        
     case 'F9'
         fobj=@MLP_Sphere
         lb=-10;
         ub=10;
         dim=61;

      case 'F10'
         fobj=@MLP_Gumus
         lb=-1;
         ub=1;
         dim=480;
         
end

end


