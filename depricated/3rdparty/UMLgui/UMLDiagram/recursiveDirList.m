function dirList = recursiveDirList(thisDir)

dirList = {};

files = dir(thisDir);
if isempty(files)
  return
end

dirList = [dirList; thisDir];

% set logical vector for subdirectory entries in thisDir
isdir = logical(cat(1,files.isdir));

% Recursively descend through directories

dirs = files(isdir); % select only directories
classsep = '@';      % ignore class directories
packagesep = '+';    % ignore package directories

for i=1:length(dirs)
   dirname = dirs(i).name;
   if(~strcmp( dirname,'.') && ~strcmp( dirname,'..') ...
           && ~strncmp(dirname,classsep,1) ...
           && ~strncmp(dirname,packagesep,1) ...
           && ~strcmp(dirname,'private') ...
           && ~strcmp(dirname,'.git'))
      dirList = [dirList ; recursiveDirList(fullfile(thisDir,dirname))]; %#ok recursive calling of this function.
   end
end

end