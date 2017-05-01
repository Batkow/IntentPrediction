function xNext = Dynamics(x,u,dt)

  xNext = [ x(1) + dt * cos(u);
        x(2) + dt * sin(u)];


end