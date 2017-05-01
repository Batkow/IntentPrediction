close all, clear all, clc
gridHeight = 100;
gridWidth = 100;
opt.gridHeight = gridHeight;
opt.gridWidth = gridWidth;
nActions = 24;
discount = 0.999  ;
     
actions = linspace(0,2*pi,nActions+1);
actions(end) = [];

img = imread('intersection.png');
rChannel = find(img(:,:,1));
gChannel = find(img(:,:,2));
bChannel = find(img(:,:,3));

R = zeros(size(img(:,:,1)));
R(rChannel) = -2;
R(gChannel) = -0.3;

xStates = 1:gridWidth;
yStates = 1:gridHeight;

value = R;
%%
for k = 1:100
  k
  for xi = 1:gridWidth
    for yi = 1:gridHeight
        if (xi == 100 && yi == 100)
          continue
        end
        Xnew = Dynamics([xi;yi],actions,1);
        total = R(yi,xi) + discount * max(interp2(value,Xnew(1,:),Xnew(2,:)));
        value(yi,xi) =  total;
    end
  end
end
value
%%

imshow(value,[])
shg




%%
x = [1.1; 2.0];
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

plot(X(1,:),X(2,:))
shg