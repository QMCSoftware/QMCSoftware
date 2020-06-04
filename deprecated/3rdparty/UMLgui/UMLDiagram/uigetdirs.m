function [dirList] = uigetdirs(start_path, dialog_title)
% Pick multiple directories with the Java widgets instead of uigetdir

import javax.swing.JFileChooser;

if (nargin == 0 || strcmp(start_path,''))
    start_path = pwd;
end

jchooser = javaObjectEDT('javax.swing.JFileChooser', start_path);

%jchooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);

jchooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
jchooser.setMultiSelectionEnabled(true);

if (nargin > 1)
    jchooser.setDialogTitle(dialog_title);
end

status = jchooser.showOpenDialog([]);

if (status == JFileChooser.APPROVE_OPTION)
    jFile = jchooser.getSelectedFiles();
    dirList{size(jFile, 1)}=[];
    for i=1:size(jFile, 1)
        dirList{i} = char(jFile(i).getAbsolutePath);
    end

elseif status == JFileChooser.CANCEL_OPTION
    dirList = [];
else
    error('Error occured while picking file.');
end