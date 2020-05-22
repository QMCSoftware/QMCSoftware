function [output] = uigetfileordir(start_path, dialog_title)
% Pick multiple directories with the Java widgets instead of uigetdir

import javax.swing.JFileChooser;

if (nargin == 0 || strcmp(start_path,''))
    start_path = pwd;
end

jchooser = javaObjectEDT('javax.swing.JFileChooser', start_path);

jchooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);

jchooser.setMultiSelectionEnabled(false);

if (nargin > 1)
    jchooser.setDialogTitle(dialog_title);
end

status = jchooser.showOpenDialog([]);

if (status == JFileChooser.APPROVE_OPTION)
    jFile = jchooser.getSelectedFile();
    output = char(jFile.getAbsolutePath);
elseif status == JFileChooser.CANCEL_OPTION
    output = [];
else
    error('Error occured while picking file.');
end