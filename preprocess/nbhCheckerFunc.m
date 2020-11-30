function final_traj = nbhCheckerFunc(traj, tracks)
%% This file only change the info saved in 'traj'
%  @@@@@ Finally to make sure 
%       1. Neighbour vehicles (Actuall the neightbours of ego vehicle, i.e. targets to be predicted) 
%          eixst in tracks, with at least 3s history
%       2. centVehicle is also included in the gird of all of its prediction targets (some case of overlapping)
%       3. The neighbours around each target, i.e. subneighbours, are included in tracks.

rowNum = size(traj, 1);
columnNum = size(traj, 2);
vehIdxLimit = size(tracks,2);
rowSavedIn = 0;
processed_traj = single(zeros(size(traj)));

fprintf('checkerFunc processing: ')

% Loop on each row of traj.
for k = 1: rowNum       
    ds = traj(k,1);     % the dataset of this row.  
    ego = traj(k,2);    % the Id of ego vehicle.
    t = traj(k,3);      % the Frame Id for this row.
    keepThisRow = 1;
    
    % Check if there exists neighbours
    neighbours = nonzeros(traj(k, 14:columnNum));
    if isempty(neighbours)
        continue
    elseif any(neighbours>vehIdxLimit)
        % ### May not be needed after merging tracks 
        continue
    end
    
    %Loop on all neighbour vehicles------------------------
    for i = 1:length(neighbours)
       nbID =  neighbours(i);
       % Check if the neighbour exists in tracks
       if isempty(tracks{ds, nbID})
           keepThisRow = 0;
           break
       end
       % Check if it has more than 30 frames history
       fram31th = tracks{ds, nbID}(1, 31);
       if t < fram31th
           keepThisRow = 0;
           break
       end
       % Check if ego car is continued in its sub_neighbours
       sub_neighbours = nonzeros(tracks{ds, nbID}(12:(columnNum-2), find(tracks{ds, nbID}(1, :)==t)));
       if isempty(sub_neighbours)
           keepThisRow = 0;  
           break
       elseif any(sub_neighbours>vehIdxLimit)
           % ### May not be needed after merging tracks 
           keepThisRow = 0;  
           break
       elseif all(sub_neighbours~=ego)
           keepThisRow = 0;  
           break
       end
       % Check if the sub_neighbours exists in tracks
       for ii = 1:length(sub_neighbours)
           if isempty(tracks{ds, sub_neighbours(ii)})
               keepThisRow = 0;  
               break
           end
       end
       if not(keepThisRow)
           break
       end
    end
    
    % Keep applicable rows
    if keepThisRow
        rowSavedIn = rowSavedIn + 1;
        processed_traj(rowSavedIn, :) = traj(k,:);
    end
    
    %Count time------------------------
    if mod(k,100000)==0
        fprintf( '%.2f ... ', k/rowNum);
    end
end

final_traj = processed_traj( processed_traj(:,1)~=0, : );
fprintf( '\nPast #data: %d ===>>> Now #data: %d \n\n', size(traj, 1), size(final_traj, 1));
end