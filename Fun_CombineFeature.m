function [IMGFEATURES_T, LABELS_T] = Fun_CombineFeature(str,FakeType)
len = size(str,2);
IMGFEATURE_T = cell(1,len);
LABELS_T = cell(1,len);
if FakeType ~= 0
    for i = 1:len
        load(str{i});
        IMGFEATURE = IMGFEATURE(:,find(LABELS == 1 | LABELS == FakeType));
        LABELS = LABELS(find(LABELS == 1 | LABELS == FakeType));
        IMGFEATURES_T{i} = IMGFEATURE;
        LABELS_T{i} = LABELS;
    end
else
    for i = 1:len
        load(str{i});
        IMGFEATURES_T{i} = IMGFEATURE;
        LABELS_T{i} = LABELS;
    end
end
