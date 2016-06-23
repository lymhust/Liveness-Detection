function plot_features(finalf,pred)
figure1 = figure;
axes1 = axes('Parent',figure1,'YGrid','on','XGrid','on','LineWidth',2,...
    'FontSize',12,...
    'FontName','times new roman');
box(axes1,'on');
hold(axes1,'all');
marksize = [8 8 10 10 10 8 8];%fliplr([8 8 10 10 10 8 8]);
marks = {'<','v','*','+','x','square','o'};%fliplr({'<','v','*','+','x','square','o'});
col = [0.24705882370472 0.24705882370472 0.24705882370472;
    0.749019622802734 0.749019622802734 0;
    0 0.749019622802734 0.749019622802734;
    0 0.498039215803146 0;
    0.749019622802734 0 0.749019622802734;
    1 0 0;
    0 0 1];
col = flipud(col);
unipred = unique(pred);
names = cell(1,length(unipred));

dim = size(finalf,1);

if dim == 2
    for j = 1:length(unipred)
        plot(finalf(1,pred==unipred(j)),finalf(2,pred==unipred(j)),'MarkerSize',marksize(j),'Marker',marks{j},'LineWidth',2,'LineStyle','none',...
            'Color',col(j,:));
        names{j} = num2str(unipred(j));
        hold on;
    end
    led = legend(names);
    set(led,'Location','SouthWest','LineWidth',2,'FontSize',10);
    hold off;
else
    view(axes1,[41.5 32]);
    box(axes1,'on');
    grid(axes1,'on');
    hold(axes1,'all');
    
    for j = 1:length(unipred)
        plot3(finalf(1,pred==unipred(j)),finalf(2,pred==unipred(j)),finalf(3,pred==unipred(j)),'MarkerSize',marksize(j),'Marker',marks{j},'LineWidth',2,'LineStyle','none',...
            'Color',col(j,:));
        names{j} = num2str(unipred(j));
        hold on;
    end
    led = legend(names);
    set(led,'Location','SouthWest','LineWidth',2,'FontSize',10);
    hold off;
end


end
