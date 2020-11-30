%% Process highD dataset into required datasets %%
clear;
clc;

%% WORKFLOW
%{
1) Reading csv files 
2) Parse fields including sptail grid and maneuver labels
3) Using unique vehicle ids to spilit train(70%)/validation(10%)/test(20%)
4) Only reserve those data sample with at least 3s history and 5s future
%}

%% Hyperparameters:
%{
30ms for history traj
50ms for future traj
past and future 40ms for lateral behavior detaction.
grid is splitted with size 25*5 (8x7 feet for each grid)
                           25*7 (8x5 feet for each grid)
%}

grid_length=25; grid_width=5; cell_length=8; cell_width=7;

% Save location
raw_folder = sprintf('./dataset/highD/%dx%d_raw/', grid_length, grid_width);
post_folder = sprintf('./dataset/highD/%dx%d/', grid_length, grid_width);
mkdir(raw_folder); 
mkdir(post_folder);

% Other variable dependent on grid.
grid_cells = grid_length * grid_width;
grid_cent_location = ceil(grid_length*grid_width*0.5);


%% Fields in the final result:
% traj  : (data number)*(13+grid_num)
%{
1: Dataset Id
2: Vehicle Id
(Column in tracks)
|3 : Frame Id
|4 : Local X
|5 : Local Y
|6 : Lane Id
|7*: Lateral maneuver
|8*: Longitudinal maneuver
|9 : Length
|10: Width
|11: Class label
|12: Velocity
|13: Accerlation
|14-end*: Neighbor Car Ids at grid location
%}

% tracks: includes {Dataset_Id*Vehicle_Id}, each cell with (11+grid)*totalFramNum
%{
|1 : Frame Id
|2 : Local X
|3 : Local Y
|4 : Lane Id
|5 : Lateral maneuver
|6 : Longitudinal maneuver
|7 : Length
|8 : Width
|9 : Class label
|10: Velocity
|11: Accerlation
|12-end*: Neighbor Car Ids at grid location
%}


%% 1.Load data 
dataset_to_use = 120;
disp('Loading data...')

for k = 1:dataset_to_use
    if mod(k,2)
        % forward side:
        dataset_name = sprintf('./raw_highd_ngsim_format/%02d-fwd.csv', ceil(k/2));
    else
        % backward side:
        dataset_name = sprintf('./raw_highd_ngsim_format/%02d-bck.csv', ceil(k/2));
    end
    csv{k}  = readtable(dataset_name);
    traj{k} = csv{k}{:,1:14};
    % Add dataset id at the 1st column
    traj{k} = single([ k*ones(size(traj{k},1),1), traj{k} ]);
    % Finally 1:dataset id, 2:Vehicle id, 3:Frame index, 
    %         6:Local X, 7:Local Y, 15:Lane id.
    %         10:v_length, 11:v_Width, 12:v_Class
    %         13:Velocity (feet/s), 14:Acceleration (feet/s2).
    traj{k} = traj{k}(:,[1,2,3,6,7,15,10,11,12,13,14]);
    % Leave space for maneuver labels (2 columns) and grid (grid_cells columns)
    traj{k} = [ traj{k}(:,1:6), zeros(size(traj{k},1),2), traj{k}(:,7:11), zeros(size(traj{k},1),grid_cells) ];
    
    lane_num = size(unique(traj{k}(:, 6)), 1);
end

% Use the vehilce's center as its location
offset = zeros(1,dataset_to_use);
for k = 1:dataset_to_use
    traj{k}(:,5) = traj{k}(:,5) - 0.5*traj{k}(:,9);
    offset(k) = min(traj{k}(:,5));
    if offset(k) < 0
        % To make coordinate Y > 0
        traj{k}(:,5) = traj{k}(:,5) - offset(k);
    end
end


%% 2.Parse fields (listed above: maneuver label, neighbour grid)
disp('Parsing fields...')

poolobj = parpool( min(8, dataset_to_use) );
parfor ii = 1:dataset_to_use   % Loop on each dataset.
% for ii = 1:dataset_to_use
    tic;
    disp(['Now process dataset ', num2str(ii)])
   
    % Loop on each row.
    for k = 1:length(traj{ii}(:,1)) 
        % Refresh the process every 1 mins
        if toc > 60  
            fprintf( 'Dataset-%d: Complete %.3f%% \n', ii, k/length(traj{ii}(:,1))*100 );
            tic;
        end

        dsId = ii;
        vehId = traj{ii}(k,2);
        time = traj{ii}(k,3);
        % Get all rows about this vehId
        vehtraj = traj{ii}(traj{ii}(:,2)==vehId, : );  
    
        % Get the row index of traj at this frame.
        ind = find(vehtraj(:,3)==time);                                      
        ind = ind(1);
        lane = traj{ii}(k,6);
        
        % Lateral maneuver in Column 7:
        ub = min(size(vehtraj,1),ind+40);                                %Upper boundary (+40 frame)
        lb = max(1, ind-40);                                             %Lower boundary (-40 frame)
        if vehtraj(ub,6)>vehtraj(ind,6) || vehtraj(ind,6)>vehtraj(lb,6)  %(prepate to turn or stablize after turn)
            traj{ii}(k,7) = 3;   % Turn Right==>3. 
        elseif vehtraj(ub,6)<vehtraj(ind,6) || vehtraj(ind,6)<vehtraj(lb,6)
            traj{ii}(k,7) = 2;   % Turn Left==>2.
        else
            traj{ii}(k,7) = 1;   % Keep lane==>1.
        end
        
        % Longitudinal maneuver in Column 8:
        ub = min(size(vehtraj,1),ind+50); % Future boundary  (+50 frame)
        lb = max(1, ind-30);              % History boundary (-30 frame)
        if ub==ind || lb ==ind
            traj{ii}(k,8) = 1;   % Normal==>1
        else
            vHist = (vehtraj(ind,5)-vehtraj(lb,5))/(ind-lb);
            vFut = (vehtraj(ub,5)-vehtraj(ind,5))/(ub-ind);
            if vFut/vHist <0.8
                traj{ii}(k,8) = 2; % Brake==> 2
            else
                traj{ii}(k,8) = 1; % Normal==>1
            end
        end
        
        % Get grid locations in Column 14~13+grid_length*grid_width (grid_cells, each with cell_length*cell_width): 
        centVehX = traj{ii}(k,4);
        centVehY = traj{ii}(k,5);
        gridMinX = centVehX - 0.5*grid_width*cell_width;
        gridMinY = centVehY - 0.5*grid_length*cell_length;
        otherVehsAtTime = traj{ii}( traj{ii}(:,3)==time , [2,4,5]);  % Only keep the (vehId, localX, localY)        
        otherVehsInSizeRnage = otherVehsAtTime( abs(otherVehsAtTime(:,3)-centVehY)<(0.5*grid_length*cell_length) ...
                                              & abs(otherVehsAtTime(:,2)-centVehX)<(0.5*grid_width*cell_width) , :);
        if ~isempty(otherVehsInSizeRnage)
            % Lateral and Longitute grid location. Finally exact location is saved in the 3rd column;
            otherVehsInSizeRnage(:,2) = ceil((otherVehsInSizeRnage(:,2) - gridMinX) / cell_width); 
            otherVehsInSizeRnage(:,3) = ceil((otherVehsInSizeRnage(:,3) - gridMinY) / cell_length); 
            otherVehsInSizeRnage(:,3) = otherVehsInSizeRnage(:,3) + (otherVehsInSizeRnage(:,2)-1) * grid_length; 
            for l = 1:size(otherVehsInSizeRnage, 1)
                exactGridLocation = otherVehsInSizeRnage(l,3);
                if exactGridLocation ~= grid_cent_location % The center gird location is kept to NONE
                    traj{ii}(k,13+exactGridLocation) = otherVehsInSizeRnage(l,1);
                end
            end   
        end
        
    end
end
delete(poolobj);


%% 3.Merge and Split train, validation, test
disp('Splitting into train, validation and test sets...')

% Merge all datasets together.
trajAll = [];
for i = 1:dataset_to_use
    trajAll = [trajAll; traj{i}];
    fprintf( 'Now merge %d rows of data from traj{%d} \n', size(traj{i},1), i);
end
clear traj;

% Training, Validation and Test dataset (Everything together)
trajTr = [];
trajVal = [];
trajTs = [];
for k = 1:dataset_to_use  % Split the vehilce's trajectory in all dataset_to_use
    uniqueVehIds = sort( unique(trajAll(trajAll(:,1)==k,2)) );
    % Cutting point: Vehicle Id with index of 0.7* length(candidate vehicles) (70% Training set)
    ul1 = uniqueVehIds( round(0.7*length(uniqueVehIds)) ); 
    % Cutting point: Vehicle Id with index of 0.8* length(candidate vehicles) (20% Test set)
    ul2 = uniqueVehIds( round(0.8*length(uniqueVehIds)) ); 
    % Extract according to the vehicle ID 
    trajTr =  [trajTr;  trajAll(trajAll(:,1)==k & trajAll(:,2)<=ul1, :) ]; 
    trajVal = [trajVal; trajAll(trajAll(:,1)==k & trajAll(:,2)>ul1 & trajAll(:,2)<=ul2, :) ];
    trajTs =  [trajTs;  trajAll(trajAll(:,1)==k & trajAll(:,2)>ul2, :) ];
end

% Merging all info together in tracks
% The neighbour existence problem is addressed
tracks = {};
for k = 1:dataset_to_use
    trajSet = trajAll(trajAll(:,1)==k,:); 
    carIds = unique(trajSet(:,2));      % Unique Vehicle ID, then get a cell for each car.    
    for l = 1:length(carIds)
        % The cell in {datasetID, carID} is placed with (11+grid_cells)*TotalFram.
        tracks{k,carIds(l)} = trajSet( trajSet(:,2)==carIds(l), 3:end )';                      
    end
end


%% 4.Filter edge cases: 
disp('Filtering edge cases...')

% Flag for whether to discard this row of dataraw_folder
indsTr = zeros(size(trajTr,1),1); 
indsVal = zeros(size(trajVal,1),1);
indsTs = zeros(size(trajTs,1),1);

% Since the model uses 3 sec of trajectory history for prediction, and 5s
% future for planning, therefore the reserve condition for each row of data: 
    % 1) this frame t should be larger than the 31st id, and
    % 2) has at least 5s future.
    
for k = 1: size(trajTr,1)   % Loop on each row of traj.
    t = trajTr(k,3);    
    if size(tracks{trajTr(k,1),trajTr(k,2)}, 2) > 30
        if tracks{trajTr(k,1),trajTr(k,2)}(1,31) <= t && tracks{trajTr(k,1),trajTr(k,2)}(1,end)>= t+50
            indsTr(k) = 1;
        end
    end
end
trajTr = trajTr(find(indsTr),:);
for k = 1: size(trajVal,1)
    t = trajVal(k,3);
    if size(tracks{trajVal(k,1),trajVal(k,2)}, 2) > 30
        if tracks{trajVal(k,1),trajVal(k,2)}(1,31) <= t && tracks{trajVal(k,1),trajVal(k,2)}(1,end)>= t+50
            indsVal(k) = 1;
        end
    end
end
trajVal = trajVal(find(indsVal),:);
for k = 1: size(trajTs,1)
    t = trajTs(k,3);
    if size(tracks{trajTs(k,1),trajTs(k,2)}, 2) > 30
        if tracks{trajTs(k,1),trajTs(k,2)}(1,31) <= t && tracks{trajTs(k,1),trajTs(k,2)}(1,end)>= t+50
            indsTs(k) = 1;
        end
    end
end
trajTs = trajTs(find(indsTs),:);


% % ## If filter those files with only 2 lanes
% files_with_dual_lanes = [1 2 3 15 16 17 18 19 20 21 22 23 24];
% dsIds_with_dual_lanes = sort([files_with_dual_lanes*2 files_with_dual_lanes*2-1]);
% keep_this_row = false( size(trajTr, 1),1 );
% for i = 1:size(trajTr, 1)
%     keep_this_row(i) = ~any(trajTr(i, 1)==dsIds_with_dual_lanes);
% end
% trajTr = trajTr(keep_this_row, :);
% fprintf( '%d rows of data with only 2-lanes from train are filtered \n', length(keep_this_row) - sum(keep_this_row));
% 
% keep_this_row = false( size(trajVal, 1),1 );
% for i = 1:size(trajVal, 1)
%     keep_this_row(i) = ~any(trajVal(i, 1)==dsIds_with_dual_lanes);
% end
% trajVal = trajVal(keep_this_row, :);
% fprintf( '%d rows of data with only 2-lanes from train are filtered \n', length(keep_this_row) - sum(keep_this_row));
% 
% keep_this_row = false( size(trajTs, 1),1 );
% for i = 1:size(trajTs, 1)
%     keep_this_row(i) = ~any(trajTs(i, 1)==dsIds_with_dual_lanes);
% end
% trajTs = trajTs(keep_this_row, :);
% fprintf( '%d rows of data with only 2-lanes from train are filtered \n', length(keep_this_row) - sum(keep_this_row));


%% Save mat files:
% traj  : n*(13+grid_cells), n is the data number.
% tracks: 6*maxVehicleId, each cell is specified for (datasetId, vehicleId), with size (11+grid_cells)*totalFramNum.
disp('Saving mat files...')

% Save raw data
save(strcat(raw_folder,'highdTrainRaw'), 'trajTr', 'tracks','-v7.3');
save(strcat(raw_folder,'highdValRaw'),  'trajVal','tracks','-v7.3');
save(strcat(raw_folder,'highdTestRaw'), 'trajTs', 'tracks','-v7.3');

% Save post-processed data
fprintf( '### Train data: \n');
traj = nbhCheckerFunc(trajTr, tracks);
save(strcat(post_folder,'highdTrainAround'),'traj','tracks','-v7.3');
traj = traj(1:50:size(traj,1), :);
save(strcat(post_folder,'tinyTrainAround'),'traj','tracks','-v7.3');

fprintf( '### Validation data: \n');
traj = nbhCheckerFunc(trajVal, tracks);
save(strcat(post_folder,'highdValAround'),'traj','tracks','-v7.3');
traj = traj(1:50:size(traj,1), :);
save(strcat(post_folder,'tinyValAround'),'traj','tracks','-v7.3');

fprintf( '### Test data: \n');
traj = nbhCheckerFunc(trajTs, tracks);
save(strcat(post_folder,'highdTestAround'),'traj','tracks','-v7.3');
traj = traj(1:50:size(traj,1), :);
save(strcat(post_folder,'tinyTestAround'),'traj','tracks','-v7.3');

fprintf('Complete');