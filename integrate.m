function [solution, dataObj] = integrate(funObj, distribObj, stopCritObj)
%§\mcommentfont Specify and generate values $f(\vx)$ for $\vx \in \cx$§
% funObj = an object from class fun
% distribObj = an object from class discrete_distribution
% stopcritObj = an object from class stopping_criterion


[stopCritObj, dataObj, distribObj] = stopYet(stopCritObj, [], funObj, distribObj);
while ~strcmp(dataObj.stage, 'done')
   dataObj = updateData(dataObj, distribObj, funObj);
   [stopCritObj, dataObj] = stopYet(stopCritObj, dataObj, funObj);
end
solution = dataObj.solution;
dataObj.timeUsed = toc(dataObj.timeStart);

