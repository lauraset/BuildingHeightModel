% iroot = 'D:\yinxcao\height\pred_deep\';
iroot = '';

fcode ='ningbo';
% fcode='shenzhen1';
% fcode='shenzhen2';

for i=1%3:num
    tic;
    file = [iroot, fcode, '\pred'];%fullfile(filelist(i).folder, filelist(i).name);
    ipath=[file,'\']; % the path of data
    respath=[file,'\'];
    % if ~isfolder(respath)
    %     mkdir(respath)
    % end
    % path0= [respath, 'predtlcnetu_200_segh'];
    path1 = [respath, 'predtlcnetu_200_obj1.tif'];
    % path2 = [respath, 'predtlcnetu_200_obj1000'];
    % path3 =[respath, 'predtlcnetu_200_obj500'];
    % path4 = [respath, 'predtlcnetu_200_obj100'];
    % if isfile(path0)
    %     continue;
    % end
    % 0. read 2.5 m deep prediction
    [predheight, R]=geotiffread([ipath, 'predtlcnetu_200.tif']);%predheight=predheight';
    predseg=imread([ipath, 'predtlcnetu_seg.tif']);
    %predseg=predseg';
    info = geotiffinfo([ipath, 'predtlcnetu_200.tif']);

    pred=cat(3,predheight,single(predseg));
    clear('predseg','predheight');
    % 0. seg mask height
    % if ~isfile(path0)
    %     disp('process seg.*height');
    %     nenviwrite(pred(:,:,1).*pred(:,:,2),path0);
    % end
    % 1. object processing
    pvalue=98; % 75
    ifun=@(block_struct) object_val_prcimgv2(block_struct.data,pvalue);
    predobject=blockproc(pred, [800,800],ifun);
    % add 
    predobject = uint8(predobject);
    % 2. save object result
    if ~isfile(path1)
        disp('process object75 2.5');
        geotiffwrite(path1, predobject, R,"GeoKeyDirectoryTag", ...
            info.GeoTIFFTags.GeoKeyDirectoryTag)
        % nenviwrite(predobject, path1);
        % imwrite(predobject, [path1,'.tif']);
    end
    % % 3. downsampling to any scale
    % if ~isfile(path2)
    %     disp('process object75 1000');
    %     scale=1000;
    %     preobj1000=func_resize(predobject, pred(:,:,2), 1, 2.5/scale);
    %     nenviwrite(preobj1000, path2);
    % end
    % 
    % %500m
    % if ~isfile(path3)
    %     disp('process object75 500');
    %     scale=500;
    %     preobj500=func_resize(predobject, pred(:,:,2), 1, 2.5/scale);
    %     nenviwrite(preobj500, path3);
    % end
    % % 100m
    % if ~isfile(path4)
    %     disp('process object75 100');
    %     scale=100;
    %     preobj100=func_resize(predobject, pred(:,:,2), 1, 2.5/scale);
    %     nenviwrite(preobj100, path4);
    % end

    toc;
end

