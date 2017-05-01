function [xNeighbours,yNeighbours] = Interpolate(X,opt)

  width = opt.gridWidth;
  heigh = opt.gridHeight;
  
  x = X(1);
  y = X(2);
  
  xNeighbours = [];
  yNeighbours = [];
  if mod(x,1) == 0
    xNeighbours = round(x);
  else
    xNeighbours = [floor(x) ceil(x)];
  end
    
  if mod(y,1) == 0
    yNeighbours = round(y);
  else
    yNeighbours = [floor(y) ceil(y)];
  end
  
  idx = find((xNeighbours > opt.gridWidth) | (xNeighbours < 1));
  xNeighbours(idx) = [];
  
  idx = find((yNeighbours > opt.gridHeight) | (yNeighbours < 1));
  yNeighbours(idx) = [];
end