clear all, close all, clc
load value.mat
img = imread('intersection.png');

nActions = 24;
actions = linspace(0,2*pi,nActions+1);
actions(end) = [];
x = [1.1; 100.0];
v = 1;
dt = 1;
X = x;
u = [];
for k = 1 : 2000
  xNew = Dynamics(x,actions,dt);
  [maxVal,idx] = max(interp2(value,xNew(1,:),xNew(2,:)));
  x = xNew(:,idx);
  X = [X x];
  if norm(x-[100;100])<1
    disp('breaking')
    break
  end
  u = [u actions(idx)];
end


plot(X(1,1),X(2,1),'ko','markersize',5), hold on
plot(100,100,'ro','markersize',5), hold on
imshow(img,[],'initialmagnification','fit')
hold on
plot(X(1,:),X(2,:),'k','linewidth',2),

shg