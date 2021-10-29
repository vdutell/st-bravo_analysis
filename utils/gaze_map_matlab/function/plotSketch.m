%% PLOT SKETCH FIGURE
function h = plotSketch(sketchFig,EYE_POS,P,EYE_COLOR)

h = figure(sketchFig);hold on,axis equal
set(gca,'xdir','reverse')
plot3(0,0,0,'ko')
quiver3(0,0,0,0,0,25,'k','linewidth',2)
quiver3(0,0,0,0,25,0,'k','linewidth',2)
quiver3(0,0,0,25,0,0,'k','linewidth',2)

[Xeye,Yeye,Zeye] = sphere() ;
surf(12*Xeye + EYE_POS(1),12*Zeye + EYE_POS(3),12*Yeye + EYE_POS(2),...
    'facecolor',EYE_COLOR,'facelighting','gouraud','edgecolor','none')

plot3(P(:,1),P(:,3),P(:,2),'+','color',EYE_COLOR,'linewidth',2)
plot3([P(:,1),EYE_POS(1).*ones(length(P),1)]',...
      [P(:,3),EYE_POS(3).*ones(length(P),1)]',...
      [P(:,2),EYE_POS(2).*ones(length(P),1)]','k:');

