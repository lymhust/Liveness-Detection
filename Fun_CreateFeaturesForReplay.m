function [TrainImg,TrainLabels,TestImg,TestLabels,DevImg,DevLabels] = Fun_CreateFeaturesForReplay(str,FakeType,classnum)
if FakeType ~= 0
    for i = 1:3
        load(str{i});
        IMGFEATURE = IMGFEATURE(:,find(LABELS == 1 | LABELS == FakeType));
        LABELS = LABELS(find(LABELS == 1 | LABELS == FakeType));
        LABELS(LABELS == FakeType) = 2;
        switch i
            case 1
                TrainImg = IMGFEATURE;
                TrainLabels = LABELS;
            case 2
                TestImg = IMGFEATURE;
                TestLabels = LABELS;
            case 3
                DevImg = IMGFEATURE;
                DevLabels = LABELS;
        end
    end
else
    for i = 1:3
        load(str{i});
        if classnum == 2
            LABELS(find(LABELS > 1)) = 2;
        end
        
        switch i
            case 1
                TrainImg = IMGFEATURE;
                TrainLabels = LABELS;
            case 2
                TestImg = IMGFEATURE;
                TestLabels = LABELS;
            case 3
                DevImg = IMGFEATURE;
                DevLabels = LABELS;
        end
    end
end
