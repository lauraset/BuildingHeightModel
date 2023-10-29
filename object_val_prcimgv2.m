function res=object_val_prcimgv2(img,ratio)
%% FUNCTIONS
%           feature [rows*cols,bands] -> im [rows,cols]
%INPUT ARGUMENTS
%           im: offer the binary results
%           feature:offer the continous values
%OUTPUT ARGUMENTS
%           res:result
% Author： Yinxia Cao
% Update: 2020.8.25  return image
f1=img(:,:,1); % feature
im=img(:,:,2);% mask

cc=bwconncomp(im);
num=cc.NumObjects;
res=zeros(size(im));

for i=1:num
    id=cc.PixelIdxList{i};
%     res{i,1}=id; %id位置
%     res{i,2}=f1(id); %对应的特征值
    res(id)=prctile(f1(id),ratio);%each column
end
end