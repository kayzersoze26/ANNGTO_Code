
clear all 
clc

IterNo=1;

PopSize=30; % Number of search agents

% classification datasets
load ALRT_T1.mat
Data=data;
X = Data(:,3:5);
Y = Data(:,2);

DataNum = size(X,1);
InputNum = size(X,2);
OutputNum = size(Y,2);
% nP=20;
HiddenNeuronSize=20;
% MaxIt=250;
%% Normalization
MinX = min(X);
MaxX = max(X);

MinY = min(Y);
MaxY = max(Y);

XN = X;
YN = Y;
% 
for i = 1:InputNum
    XN(:,i) = Normalize_Fcn(X(:,i),MinX(i),MaxX(i));
end

for i = 1:OutputNum
    YN(:,i) = Normalize_Fcn(Y(:,i),MinY(i),MaxY(i));
end

%% Test and Train Data
TrPercent = 60;
TrNum = round(DataNum * TrPercent / 100);


Xtr = XN(1 : TrNum,:);
Ytr = YN(1 : TrNum,:);



Xts = XN(TrNum+1:end,:);
Yts = YN(TrNum+1:end,:);
Ytr1=Denormalize_Fcn(Ytr,MinY, MaxY);
Yts1=Denormalize_Fcn(Yts,MinY, MaxY);


TrainActual=Ytr1;
TestActual=Yts1;
TrainPredicted=[];
TestPredicted=[];
Max_iteration=500;
global pred_tr;
for k=1:30

[lb,ub,dim,fobj,pred_tr]=Get_Functions_details(Function_name);

[Best_score,Best_pos,GWO_cg_curve]=GTO(PopSize,Max_iteration,lb,ub,dim,fobj);
  
pred_tr2=Denormalize_Fcn(pred_tr,MinY, MaxY);

TrainPredicted=[TrainPredicted pred_tr2]; 
testsize = size(Xts,1);

if Function_name=='F10'

Hnode=20;
dim = 4*Hnode+Hnode*Hnode;
 

  
  
%   zNN=zeros(1,10);
  test_error=zeros(1,IterNo);
    for i=1:IterNo
        for ww=1:2*Hnode+Hnode*Hnode
            W(ww)=Best_pos(i,ww);
        end
%         for bb=2*Hnode+Hnode*Hnode+1: 4*Hnode+Hnode*Hnode
%             B(bb-2*Hnode+Hnode*Hnode)=Best_pos(i,bb);
%         end

        for bb=441:480
            B(bb-440)=Best_pos(i,bb);
        end 



        for pp=1:testsize %% 
            actualvalue=TestPhase(W,B,Xts(pp,1),Xts(pp,2),Xts(pp,3),Hnode);
             actualvalue1(pp)=actualvalue;
  end

pred_test=Denormalize_Fcn(actualvalue1,MinY, MaxY);
Yts1=Denormalize_Fcn(Yts,MinY, MaxY);
TestPredicted=[TestPredicted pred_test'];
n=size(Yts,1);
p=dim;
RMSE_ts(k)= sqrt(mse(pred_test- Yts1'));
TRAINRES=[TrainActual TrainPredicted];
TESTRES=[TestActual TestPredicted];

    end
end
end
