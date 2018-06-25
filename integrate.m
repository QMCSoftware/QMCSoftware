function [solution, dataObj] = integrate(funObj, distribObj, stopCritObj)
%§\mcommentfont Specify and generate values $f(\vx)$ for $\vx \in \cx$§
% funObj = an object from class fun
% distribObj = an object from class discrete_distribution
% stopcritObj = an object from class stopping_criterion

%Initialize the accumData object and other crucial objects
[stopCritObj, dataObj, distribObj] = stopYet(stopCritObj, [], funObj, distribObj);
while ~strcmp(dataObj.stage, 'done') %the dataObj.stage property tells us where we are in the process
   dataObj = updateData(dataObj, distribObj, funObj); %compute additional data
   [stopCritObj, dataObj] = stopYet(stopCritObj, dataObj, funObj); %update the status of the computation
end
solution = dataObj.solution; %assign outputs
dataObj.timeUsed = toc(dataObj.timeStart);

